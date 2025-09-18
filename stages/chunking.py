#!/usr/bin/env python3
"""
Chunk a JSONL of documents using a selectable strategy + params supplied in a config file.

Inputs:
- JSONL file with one JSON object per line containing the full text (configurable key).
- Config (YAML or JSON) specifying the chunking strategy & parameters.

Outputs:
- JSONL file where each line is a chunk record with: source_doc_id, chunk_index, text, stats.

Notes:
- This script *imports* the chunkers you provided. Paths are made tolerant via try/except blocks.
- Token counting is optional; if `tiktoken` isn’t installed, we just skip token counts.
"""

import os
import sys
import json
import math
import uuid
import argparse
from typing import Any, Dict, Optional, Iterable
from pathlib import Path

# ---------- Robust imports for your provided chunkers ----------
# Try likely layouts; adjust if your repo structure differs.
# (No need to edit your chunker classes—this script adapts to them.)
try:
    from chunking_evaluation.chunkers.fixed_token_chunker import FixedTokenChunker  # type: ignore
except Exception:
    try:
        from fixed_token_chunker import FixedTokenChunker  # type: ignore
    except Exception:
        FixedTokenChunker = None  # type: ignore

try:
    from chunking_evaluation.chunkers.recursive_token_chunker import RecursiveTokenChunker  # type: ignore
except Exception:
    try:
        from recursive_token_chunker import RecursiveTokenChunker  # type: ignore
    except Exception:
        RecursiveTokenChunker = None  # type: ignore

try:
    from chunking_evaluation.chunkers.kamradt_modified_chunker import KamradtModifiedChunker  # type: ignore
except Exception:
    try:
        from kamradt_modified_chunker import KamradtModifiedChunker  # type: ignore
    except Exception:
        KamradtModifiedChunker = None  # type: ignore

try:
    from chunking_evaluation.chunkers.cluster_semantic_chunker import ClusterSemanticChunker  # type: ignore
except Exception:
    try:
        from cluster_semantic_chunker import ClusterSemanticChunker  # type: ignore
    except Exception:
        ClusterSemanticChunker = None  # type: ignore

try:
    from chunking_evaluation.chunkers.llm_semantic_chunker import LLMSemanticChunker  # type: ignore
except Exception:
    try:
        from llm_semantic_chunker import LLMSemanticChunker  # type: ignore
    except Exception:
        LLMSemanticChunker = None  # type: ignore

# Optional libs
try:
    import yaml  # for YAML config
except Exception:
    yaml = None

try:
    import tiktoken  # for token counts
except Exception:
    tiktoken = None

from tqdm import tqdm


# ---------- Helpers ----------

import re

def locate_chunk_offsets(original_text: str, chunks: list[str]) -> list[tuple[int, int]]:
    """
    Return [(start, end_exclusive), ...] for each chunk within original_text.
    Uses a forward cursor to disambiguate repeated content and tolerate
    some whitespace trimming differences from the splitters.
    """
    offsets: list[tuple[int, int]] = []
    n = len(original_text)
    cursor = 0

    for chunk in chunks:
        if not chunk:
            # Empty chunk: pin to current cursor.
            offsets.append((cursor, cursor))
            continue

        # 1) Nudge cursor past any whitespace (helps when splitter stripped)
        while cursor < n and original_text[cursor].isspace():
            cursor += 1

        # 2) Try exact match from cursor
        idx = original_text.find(chunk, cursor)

        # 3) If not found, try stripped chunk (common when splitter strips ends)
        used = chunk
        if idx == -1:
            stripped = chunk.strip()
            if stripped:
                idx = original_text.find(stripped, cursor)
                if idx != -1:
                    used = stripped

        # 4) If still not found, try a fuzzy: collapse internal whitespace in both
        if idx == -1:
            def squeeze_ws(s: str) -> str:
                return re.sub(r'\s+', ' ', s.strip())
            sq_chunk = squeeze_ws(chunk)
            # Scan forward in small windows for performance; worst-case fallback to full search
            window_start = cursor
            step = 4096  # reasonable chunk window scan to avoid O(N^2) on huge texts
            found = False
            while window_start < n:
                window_end = min(n, window_start + 1_000_000)  # 1MB-ish window
                sq_window = squeeze_ws(original_text[window_start:window_end])
                j = sq_window.find(sq_chunk)
                if j != -1:
                    # Map back to original indices by expanding around match
                    # Find the start position in original by matching first non-ws char sequence
                    head = sq_chunk.split(' ', 1)[0]
                    # locate 'head' in original text from window_start
                    guess = original_text.find(head, window_start)
                    if guess != -1:
                        idx = guess
                        used = chunk  # best effort
                        found = True
                        break
                if window_end == n:
                    break
                window_start = window_end
            if not found:
                # As a last resort, anchor at cursor with chunk length (avoids crash, flags mismatch)
                idx = cursor

        start = max(idx, 0)
        end = start + len(used)
        # Advance cursor to end to keep matching forward
        cursor = max(end, cursor)
        offsets.append((start, end))

    return offsets




def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("YAML config provided but PyYAML is not installed. `pip install pyyaml`")
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Try JSON first, then YAML
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            if yaml is None:
                raise
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e


def get_token_encoder(encoding_name: str = "cl100k_base", model_name: Optional[str] = None):
    if tiktoken is None:
        return None
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fallback to cl100k_base
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, encoder=None) -> Optional[int]:
    if encoder is None:
        return None
    try:
        return len(encoder.encode(text))
    except Exception:
        return None


# ---------- Chunker Factory ----------

class ChunkerFactory:
    """
    Build the chosen chunker with params from config.
    Names supported:
      - "fixed_token"        -> FixedTokenChunker
      - "recursive_token"    -> RecursiveTokenChunker
      - "kamradt"            -> KamradtModifiedChunker
      - "cluster_semantic"   -> ClusterSemanticChunker
      - "llm"                -> LLMSemanticChunker
    """
    @staticmethod
    def create(name: str, params: Dict[str, Any]):
        name = (name or "").lower().strip()

        if name == "fixed_token":
            if FixedTokenChunker is None:
                raise ImportError("FixedTokenChunker not importable. Check your module path.")
            # Pass through params directly (supports: encoding_name, model_name, chunk_size, chunk_overlap, etc.)
            return FixedTokenChunker(**params)

        elif name == "recursive_token":
            if RecursiveTokenChunker is None:
                raise ImportError("RecursiveTokenChunker not importable. Check your module path.")
            return RecursiveTokenChunker(**params)

        elif name == "kamradt":
            if KamradtModifiedChunker is None:
                raise ImportError("KamradtModifiedChunker not importable. Check your module path.")
            return KamradtModifiedChunker(**params)

        elif name == "cluster_semantic":
            if ClusterSemanticChunker is None:
                raise ImportError("ClusterSemanticChunker not importable. Check your module path.")
            return ClusterSemanticChunker(**params)

        elif name == "llm":
            if LLMSemanticChunker is None:
                raise ImportError("LLMSemanticChunker not importable. Check your module path.")
            # For LLMSemanticChunker, allow api_key via env if not provided
            organisation = params.get("organisation", "openai")
            api_key = params.get("api_key")
            if api_key is None:
                api_env = params.get("api_key_env") or ("OPENAI_API_KEY" if organisation == "openai" else "ANTHROPIC_API_KEY")
                api_key = os.getenv(api_env)
            model_name = params.get("model_name")
            return LLMSemanticChunker(organisation=organisation, api_key=api_key, model_name=model_name)

        else:
            raise ValueError(f"Unknown strategy name '{name}'. Valid: fixed_token, recursive_token, kamradt, cluster_semantic, llm.")


# ---------- Main processing ----------

def process_docs(
    input_path: str,
    output_path: str,
    text_field: str = "text",
    id_field: Optional[str] = None,
    strategy_name: str = "fixed_token",
    strategy_params: Optional[Dict[str, Any]] = None,
    token_count_encoding: Optional[str] = "cl100k_base",
    token_count_model: Optional[str] = None,
    max_docs: Optional[int] = None,
    include_source_line_number_as_id: bool = True,
):
    ensure_dir(output_path)
    strategy_params = strategy_params or {}

    # Build chunker
    chunker = ChunkerFactory.create(strategy_name, strategy_params)

    # Token counter (optional)
    encoder = get_token_encoder(
        encoding_name=token_count_encoding or "cl100k_base",
        model_name=token_count_model,
    )

    # Streaming read + write
    num_docs = 0
    num_chunks = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, obj in enumerate(tqdm(iter_jsonl(input_path), desc="Chunking docs"), start=1):
            if max_docs is not None and num_docs >= max_docs:
                break

            if text_field not in obj:
                raise KeyError(f"Missing '{text_field}' in document #{i}")

            raw_text = obj[text_field]
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue

            if id_field and id_field in obj:
                source_id = obj[id_field]
            elif include_source_line_number_as_id:
                source_id = i
            else:
                source_id = str(uuid.uuid4())

            try:
                chunks = chunker.split_text(raw_text)
            except Exception as e:
                raise RuntimeError(f"Chunker failed for document #{i} (source_id={source_id}): {e}") from e

            # Compute (start, end) offsets for each chunk in the original text
            try:
                offsets = locate_chunk_offsets(raw_text, chunks)
            except Exception:
                # Fail-safe: if locator fails, produce None offsets per chunk
                offsets = [(None, None) for _ in chunks]

            for idx, (chunk_text, (start_char, end_char)) in enumerate(zip(chunks, offsets)):
                rec = {
                    "source_doc_id": source_id,
                    "chunk_index": idx,
                    "chunk_id": f"{source_id}:{idx}",
                    "text": chunk_text,
                    "n_chars": len(chunk_text),
                    "start_char": start_char,
                    "end_char": end_char,  # end is exclusive
                }
                tok = count_tokens(chunk_text, encoder)
                if tok is not None:
                    rec["n_tokens"] = tok
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                num_chunks += 1

            num_docs += 1

    print(f"\nDone. Processed {num_docs} docs → wrote {num_chunks} chunks to {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Chunk a JSONL of documents with a selectable strategy.")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    input_path = cfg.get("input_path")
    output_path = cfg.get("output_path")
    if not input_path or not output_path:
        raise ValueError("Config must include 'input_path' and 'output_path'.")

    text_field = cfg.get("text_field", "text")
    id_field = cfg.get("id_field", None)
    token_count_encoding = cfg.get("token_count_encoding", "cl100k_base")
    token_count_model = cfg.get("token_count_model", None)
    max_docs = cfg.get("max_docs", None)
    include_source_line_number_as_id = cfg.get("include_source_line_number_as_id", True)

    strategy_cfg = cfg.get("strategy", {})
    strategy_name = strategy_cfg.get("name", "fixed_token")
    strategy_params = strategy_cfg.get("params", {})

    process_docs(
        input_path=input_path,
        output_path=output_path,
        text_field=text_field,
        id_field=id_field,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        token_count_encoding=token_count_encoding,
        token_count_model=token_count_model,
        max_docs=max_docs,
        include_source_line_number_as_id=include_source_line_number_as_id,
    )


if __name__ == "__main__":
    main()

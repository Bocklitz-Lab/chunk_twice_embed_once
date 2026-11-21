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
    from pipeline_lib.fixed_token_chunker import FixedTokenChunker  # type: ignore
except Exception:
    try:
        from fixed_token_chunker import FixedTokenChunker  # type: ignore
    except Exception:
        FixedTokenChunker = None  # type: ignore

try:
    from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker  # type: ignore
except Exception:
    try:
        from recursive_token_chunker import RecursiveTokenChunker  # type: ignore
    except Exception:
        RecursiveTokenChunker = None  # type: ignore


try:
    from pipeline_lib.semantic_chunker import SemanticSplitter  # type: ignore
except Exception:
    try:
        from semantic_chunker import SemanticSplitter  # type: ignore
    except Exception:
        SemanticSplitter = None  # type: ignore



try:
    from pipeline_lib.hierarchical_section_chunker import HierarchicalSectionChunker  # type: ignore
except Exception:
    try:
        from hierarchical_section_chunker import HierarchicalSectionChunker  # type: ignore
    except Exception:
        HierarchicalSectionChunker = None  # type: ignore


# in your main script (the chunking runner)
try:
    from pipeline_lib.hybrid_multi_chunker import HybridMultiGranularityChunker  # type: ignore
except Exception:
    try:
        from hybrid_multi_chunker import HybridMultiGranularityChunker  # type: ignore
    except Exception:
        HybridMultiGranularityChunker = None  # type: ignore


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

def locate_chunk_offsets(
    original_text: str, chunks: list[str]
) -> list[tuple[Optional[int], Optional[int]]]:
    """
    Return [(start, end_exclusive), ...] for each chunk, in document order.

    A position is returned ONLY if it is exact: original_text[start:end] == chunk.
    Otherwise (None, None) is returned. Nothing is guessed, because a wrong position
    would silently mislabel chunks in any span-based relevance labelling.

    Each chunk is searched forward from one character after the START of the previous
    located chunk, not from its end. With overlap > 0 a chunk begins before the previous
    chunk ends, so searching from the previous end skips the true position (the earlier
    version did this and returned wrong offsets for almost every chunk when overlap > 0).

    Chunks that are not verbatim substrings of the document (for example the detokenised
    or whitespace-collapsed output of the semantic splitters) get (None, None).
    """
    offsets: list[tuple[Optional[int], Optional[int]]] = []
    prev_start = -1  # start of the last located chunk; the next chunk starts strictly after it

    for chunk in chunks:
        if not chunk:
            offsets.append((None, None))
            continue
        idx = original_text.find(chunk, prev_start + 1)
        if idx == -1:
            offsets.append((None, None))
            continue
        offsets.append((idx, idx + len(chunk)))
        prev_start = idx

    return offsets




class ChunkOffsetError(RuntimeError):
    """A chunker returned a position that does not reproduce its chunk (or breaks document order)."""


def check_chunk_offsets(full_text: str, triples, source_id=None) -> None:
    """Hard gate for chunkers that supply their own positions.

    Every (text, start, end) must satisfy ``full_text[start:end] == text`` and the starts must not
    decrease. Anything else raises ChunkOffsetError: the run stops instead of writing positions
    that would silently mislabel chunks in a span-based relevance labelling.
    """
    prev_start = -1
    for i, item in enumerate(triples):
        try:
            text, start, end = item
        except (TypeError, ValueError):
            raise ChunkOffsetError(f"doc {source_id!r} chunk {i}: expected (text, start, end), got {type(item)}")
        where = f"doc {source_id!r} chunk {i}"
        if not isinstance(text, str) or not text:
            raise ChunkOffsetError(f"{where}: empty or non-string chunk text")
        if not (isinstance(start, int) and isinstance(end, int)) or not (0 <= start < end <= len(full_text)):
            raise ChunkOffsetError(f"{where}: invalid span ({start}, {end}) for a text of {len(full_text)} characters")
        if full_text[start:end] != text:
            raise ChunkOffsetError(f"{where}: text != full_text[{start}:{end}]")
        if start < prev_start:
            raise ChunkOffsetError(f"{where}: start {start} < previous start {prev_start} (chunks must be in document order)")
        prev_start = start


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
      - "semantic"           -> SemanticSplitter (embedding-based)
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

        elif name == "semantic":
            if SemanticSplitter is None:
                raise ImportError("SemanticSplitter not importable.")

            model_name = params.get("model_name") or params.get("embedding_model")
            if not model_name:
                raise ValueError("Strategy 'semantic' needs params.model_name (no default model).")

            return SemanticSplitter(
                model_name=model_name,
                use_recursive=params.get("use_recursive", False),
                chunk_size=params.get("chunk_size", 4000),
                chunk_overlap=params.get("chunk_overlap", 200),
                breakpoint_threshold_type=params.get("breakpoint_threshold_type", "percentile"),
                breakpoint_threshold_amount=params.get("breakpoint_threshold_amount", 95.0),
                buffer_size=params.get("buffer_size", 1),
                min_chunk_size=params.get("min_chunk_size"),
                add_start_index=params.get("add_start_index", False),
                strip_whitespace=params.get("strip_whitespace", True),
                model_kwargs=params.get("model_kwargs"),
                encode_kwargs=params.get("encode_kwargs"),
                tokenizer_kwargs=params.get("tokenizer_kwargs"),
            )



        elif name in {"hierarchical", "section", "section_hier", "hierarchical_section"}:
            if HierarchicalSectionChunker is None:
                raise ImportError("HierarchicalSectionChunker not importable. Check your module path.")
            return HierarchicalSectionChunker(**params)

        elif name in {"hybrid_multi", "hybrid"}:
            if HybridMultiGranularityChunker is None:
                raise ImportError("HybridMultiGranularityChunker not importable. Check your module path.")
            return HybridMultiGranularityChunker(
                model_name=params.get("model_name"),
                levels=params.get("levels", ["section", "paragraph", "sentence"]),
                chunk_sizes=params.get("chunk_sizes"),
                chunk_overlaps=params.get("chunk_overlaps"),
                strip_whitespace=params.get("strip_whitespace", True),
                approx_chars_per_token=params.get("approx_chars_per_token", 4),
                sectioner=params.get("sectioner"),  # pass-through for section headings/merge behavior
            )

        else:
            raise ValueError(f"Unknown strategy name '{name}'. Valid: fixed_token, recursive_token, cluster_semantic, llm.")


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
    compute_offsets: bool = False,
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
    offset_sources: Dict[str, int] = {}

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

            native_offsets = None  # [(start, end)] when the chunker supplies its own positions
            try:
                if hasattr(chunker, "split_text_with_offsets"):
                    triples = chunker.split_text_with_offsets(raw_text)
                    chunks = [t for t, _, _ in triples]
                    native_offsets = [(st, en) for _, st, en in triples]
                else:
                    chunks = chunker.split_text(raw_text)
            except Exception as e:
                raise RuntimeError(f"Chunker failed for document #{i} (source_id={source_id}): {e}") from e

            if native_offsets is not None:
                # Hard gate: no search, no guessing. Wrong positions stop the run.
                check_chunk_offsets(raw_text, triples, source_id)

            # ---- Normalize chunks to list[dict] with text + offsets + metadata ----
            norm_chunks: list[dict] = []

            if not chunks:
                pass

            elif native_offsets is not None:
                for ctext, (s, e) in zip(chunks, native_offsets):
                    norm_chunks.append({
                        "text": ctext,
                        "start_char": s,
                        "end_char": e,
                        "granularity": None,
                        "offset_source": "native",
                    })

            elif isinstance(chunks[0], str):
                # legacy splitters (strings only)
                if compute_offsets:
                    try:
                        offsets = locate_chunk_offsets(raw_text, chunks)
                    except Exception:
                        offsets = [(None, None) for _ in chunks]
                else:
                    offsets = [(None, None) for _ in chunks]

                for ctext, (s, e) in zip(chunks, offsets):
                    norm_chunks.append({
                        "text": ctext,
                        "start_char": s,
                        "end_char": e,
                        "granularity": None,  # unknown for legacy splitters
                    })

            elif isinstance(chunks[0], dict):
                # dict-based chunkers (e.g., hybrid_multi) — preserve their metadata
                for item in chunks:
                    ctext = item.get("text")
                    if not isinstance(ctext, str):
                        continue
                    s = item.get("start_char")
                    e = item.get("end_char")

                    # A position supplied by the chunker is trusted only if it is exact.
                    if s is None or e is None or raw_text[s:e] != ctext:
                        s, e = None, None
                        # Dict chunkers (e.g. hybrid_multi) emit chunks of several granularities,
                        # so document order cannot be assumed. Accept only an unambiguous match.
                        if compute_offsets and raw_text.count(ctext) == 1:
                            s = raw_text.find(ctext)
                            e = s + len(ctext)

                    norm_item = {**item, "text": ctext, "start_char": s, "end_char": e}
                    norm_chunks.append(norm_item)

            else:
                raise TypeError(f"Unsupported chunk item type: {type(chunks[0])}")

            # ---- Write out normalized chunks ----
            for idx, item in enumerate(norm_chunks):
                chunk_text = item["text"]
                start_char = item.get("start_char")
                end_char = item.get("end_char")

                # Final guard: never write a position that does not reproduce the chunk text.
                offset_exact = (
                    start_char is not None
                    and end_char is not None
                    and raw_text[start_char:end_char] == chunk_text
                )
                if not offset_exact:
                    if item.get("offset_source") == "native":
                        raise ChunkOffsetError(
                            f"doc {source_id!r} chunk {idx}: native position does not reproduce the chunk text"
                        )
                    start_char, end_char = None, None

                rec = {
                    "source_doc_id": source_id,
                    "chunk_index": idx,
                    "chunk_id": f"{source_id}:{idx}",
                    "text": chunk_text,
                    "n_chars": len(chunk_text),
                    "start_char": start_char,
                    "end_char": end_char,  # end is exclusive; None if no exact position exists
                    # True only when raw_text[start_char:end_char] == text. Chunk files written
                    # before this field existed have unreliable positions when overlap > 0:
                    # treat a missing flag as untrusted.
                    "offset_exact": offset_exact,
                    # "native": position computed by the chunker itself; "search": found by
                    # locate_chunk_offsets (legacy string-only chunkers); "none": no exact position.
                    "offset_source": (item.get("offset_source", "search") if offset_exact else "none"),
                }

                # pass-through useful metadata when present
                for key in (
                    "granularity",
                    "section_title", "section_index",
                    "parent_section_start", "parent_section_end",
                    "parent_par_start", "parent_par_end",
                ):
                    if key in item:
                        rec[key] = item[key]

                tok = count_tokens(chunk_text, encoder)
                if tok is not None:
                    rec["n_tokens"] = tok

                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                num_chunks += 1
                offset_sources[rec["offset_source"]] = offset_sources.get(rec["offset_source"], 0) + 1

            num_docs += 1

    stats = {
        "n_docs": num_docs,
        "n_chunks": num_chunks,
        "offset_sources": offset_sources,
        # chunks whose text matched more than once inside the window where they can lie
        # (recursive splitters); the first match was taken
        "n_ambiguous_locations": getattr(chunker, "n_ambiguous", 0),
        "n_relaxed_locations": getattr(chunker, "n_relaxed", 0),
    }
    with open(os.path.join(os.path.dirname(output_path) or ".", "chunking_stats.json"), "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)

    print(f"\nDone. Processed {num_docs} docs → wrote {num_chunks} chunks to {output_path}")
    print(f"Offsets: {offset_sources}; ambiguous locations: {stats['n_ambiguous_locations']}")

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
    compute_offsets = cfg.get("compute_offsets", False)

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
        compute_offsets=compute_offsets,

    )


if __name__ == "__main__":
    main()

"""One row per Phase 1 model: pooling, dimension, maximum length, tokenizer, revision, prompt (input of Table S1).

Reads the PINNED local Hugging Face snapshots (``stage1.models`` of the main config) from the cache: CPU only, offline, no weights loaded.
Models whose pinned snapshot is not in the cache are skipped and listed (``table_model_properties_skipped.csv`` and on screen).

The facts come from the files of the snapshot (``pipeline_lib/model_info.describe_snapshot``): the pooling is the Pooling module of
``modules.json``; a checkpoint WITHOUT ``modules.json`` is a plain Hugging Face encoder that SentenceTransformer wraps in Transformer +
Pooling(mean): ``pooling_is_library_default`` is true for it. ``max_seq_length`` and ``embedding_dim`` follow the library rules; the values of
the run itself are in ``run_manifest.json: model_properties`` (written by the evaluation), which is the authority if the two ever differ.

  python -m tools.model_properties --out-dir results/v2-revision
  python -m tools.model_properties --cache-dir ~/.cache/huggingface/hub --shortlist artifacts/embedding_models_screening/passed_per_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import model_info  # noqa: E402
from pipeline_lib.config_params import DEFAULT_MAIN_CONFIG  # noqa: E402

COLUMNS = ["model", "revision", "pooling_mode", "pooling_is_library_default", "modules", "embedding_dim", "max_seq_length", "tokenizer_class",
           "architecture", "model_type", "query_prompt", "passage_prompt", "instruction_masked", "shortlisted"]


def load_shortlist(path: Optional[str]) -> Optional[set]:
    """Model names of a stage 2 file (``model_name`` column); None if no file is given."""
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return {r["model_name"] for r in csv.DictReader(f)}


def load_tokenizer_facts(snap: Path) -> Tuple[Optional[str], Optional[int]]:
    """(class name, model_max_length) of the tokenizer AutoTokenizer builds from the snapshot (offline); (None, None) if it cannot be built."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        return type(tok).__name__, int(tok.model_max_length)
    except Exception:
        return None, None


def build_rows(models: Sequence[Dict[str, Any]], prompts: Dict[str, Any], *, use_prompts: bool = True, mask_instruction: bool = True,
               cache_dir: Optional[str] = None, shortlist: Optional[set] = None, load_tokenizers: bool = True
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(rows of the cached models, skipped models). ``models``: dicts with ``name`` and ``revision`` (stage1.models)."""
    rows, skipped = [], []
    for m in models:
        name, rev = m["name"], m["revision"]
        snap = model_info.snapshot_dir(name, rev, cache_dir)
        if snap is None:
            skipped.append({"model": name, "revision": rev, "reason": "pinned snapshot not in the local cache"})
            continue
        tok_cls, tok_max = load_tokenizer_facts(snap) if load_tokenizers else (None, None)
        d = model_info.describe_snapshot(snap, tok_max)
        entry = (prompts or {}).get(name) or {}
        tok = tok_cls or d.get("tokenizer_class")
        rows.append({
            "model": name, "revision": rev, "pooling_mode": d["pooling_mode"], "pooling_is_library_default": d["pooling_is_library_default"],
            "modules": "+".join(d["modules"]), "embedding_dim": d["embedding_dim"], "max_seq_length": d["max_seq_length"],
            "tokenizer_class": tok, "architecture": d.get("architecture"), "model_type": d.get("model_type"),
            "query_prompt": (entry.get("query") or "") if use_prompts else "", "passage_prompt": (entry.get("passage") or "") if use_prompts else "",
            "instruction_masked": bool(use_prompts and mask_instruction and entry.get("mask_instruction")),
            "shortlisted": (name in shortlist) if shortlist is not None else "",
        })
    return rows, skipped


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-config", default=str(DEFAULT_MAIN_CONFIG))
    ap.add_argument("--cache-dir", help="Hugging Face hub cache (default: the library default)")
    ap.add_argument("--shortlist", help="stage 2 file (passed_per_dataset*.csv): adds the column `shortlisted`")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-tokenizers", action="store_true", help="read the tokenizer class from the files only (faster)")
    args = ap.parse_args(argv)
    doc = yaml.safe_load(Path(args.main_config).read_text(encoding="utf-8"))
    p = doc["params"]
    rows, skipped = build_rows(doc["stage1"]["models"], p.get("model_prompts") or {}, use_prompts=bool(p.get("use_prompts", False)),
                               mask_instruction=bool(p.get("instructor_mask_instruction", False)), cache_dir=args.cache_dir,
                               shortlist=load_shortlist(args.shortlist), load_tokenizers=not args.no_tokenizers)
    out = Path(args.out_dir)
    write_csv(out / "table_model_properties.csv", rows, COLUMNS)
    if skipped:
        write_csv(out / "table_model_properties_skipped.csv", skipped, ["model", "revision", "reason"])
    n_default = sum(1 for r in rows if r["pooling_is_library_default"])
    print(f"{len(rows)} models -> {out / 'table_model_properties.csv'} | library-default pooling (no modules.json): {n_default} | "
          f"skipped (not cached): {len(skipped)}")
    for s in skipped:
        print(f"  skipped: {s['model']} @ {s['revision'][:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

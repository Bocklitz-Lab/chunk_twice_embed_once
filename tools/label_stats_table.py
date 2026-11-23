"""Corpus and label statistics of every prepared configuration in one table (``make report`` -> ``table_label_stats.csv``).

Reads, for each prepared folder (``<artifacts>/prepared/<strategy>_c<L>_o<O>__<key>/``):
  * ``coverage_diagnostics.csv``   chunks, chunk length in reference tokens, % of spans inside one chunk, relevant chunks per question
  * ``mteb_task/label_stats_c<c>.json``   questions per labelling step (1 / 2 / 3 / no overlapping chunk) of every cover level present
  * ``chunks/chunks.jsonl``        only when the median chunk length is missing from the diagnostics (older folders): the lengths are then
                                   counted again with the pinned reference tokenizer named in ``manifest.json``

One row per configuration and cover level:
  chunker, size, overlap, cover_level, n_chunks, chunk_tokens_median / mean / max (reference tokens, no special tokens),
  n_questions, questions_step1 / step2 / step3 / step0 (Step 1: one chunk holds >= c of the span; Step 2: several chunks together reach c;
  Step 3: c is not reachable, every overlapping chunk is labelled; step0: no chunk overlaps the span),
  pct_split_over_chunks (= 100 - % of spans lying entirely inside one chunk: the evidence is split whatever c is),
  pct_step2_or_3 (needs more than one chunk at this c), mean / max relevant chunks per question, n_empty_questions.

  python -m tools.label_stats_table --prepared-root artifacts/prepared --out results/v2-revision/table_label_stats.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FOLDER = re.compile(r"^(?P<chunker>.+)_c(?P<size>\d+)_o(?P<overlap>\d+)__(?P<key>[0-9a-f]+)$")

COLUMNS = ["configuration", "chunker", "size", "overlap", "cover_level", "n_chunks", "chunk_tokens_median", "chunk_tokens_mean", "chunk_tokens_max",
           "n_questions", "questions_step1", "questions_step2", "questions_step3", "questions_step0", "pct_split_over_chunks", "pct_step2_or_3",
           "mean_relevant_per_question", "max_relevant_per_question", "n_empty_questions"]


def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def chunk_length_stats(folder: Path) -> Dict[str, Optional[float]]:
    """Median / mean / max chunk length in reference tokens of a prepared folder, counted from ``chunks/chunks.jsonl`` with the pinned tokenizer
    of its manifest (used only when the diagnostics lack the median)."""
    from tools import coverage_diagnostics as cd
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8")) if (folder / "manifest.json").is_file() else {}
    texts = [json.loads(l)["text"] for l in open(folder / "chunks" / "chunks.jsonl", encoding="utf-8") if l.strip()]
    lens = cd._token_lengths(texts, man.get("tokenizer"), man.get("tokenizer_revision"))
    if not lens:
        return {"median": None, "mean": None, "max": None}
    return {"median": float(np.median(lens)), "mean": float(np.mean(lens)), "max": float(max(lens))}


def folder_rows(folder: Path, recompute_lengths: bool = True) -> List[Dict[str, Any]]:
    """The table rows of one prepared folder (one per cover level that has a label_stats file)."""
    diag_path = folder / "coverage_diagnostics.csv"
    diag = next(csv.DictReader(open(diag_path, newline="", encoding="utf-8")), {}) if diag_path.is_file() else {}
    m = _FOLDER.match(folder.name)
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8")) if (folder / "manifest.json").is_file() else {}
    chunker = man.get("chunker") or (m["chunker"] if m else folder.name)
    size = man.get("size", int(m["size"]) if m else None)
    overlap = man.get("overlap", int(m["overlap"]) if m else None)
    median, mean, mx = _num(diag.get("median_chunk_tokens")), _num(diag.get("mean_chunk_tokens")), _num(diag.get("max_chunk_tokens"))
    if median is None and recompute_lengths and (folder / "chunks" / "chunks.jsonl").is_file():
        st = chunk_length_stats(folder)
        median, mean, mx = st["median"], mean if mean is not None else st["mean"], mx if mx is not None else st["max"]
    inside = _num(diag.get("pct_span_inside_one_chunk"))
    rows = []
    for p in sorted((folder / "mteb_task").glob("label_stats_c*.json")):
        ls = json.loads(p.read_text(encoding="utf-8"))
        steps = ls["questions_per_step"]
        n = ls["n_questions"]
        rows.append({"configuration": folder.name, "chunker": chunker, "size": size, "overlap": overlap, "cover_level": ls["cover_level"],
                     "n_chunks": ls.get("n_chunks") or _num(diag.get("n_chunks")), "chunk_tokens_median": median, "chunk_tokens_mean": mean,
                     "chunk_tokens_max": mx, "n_questions": n, "questions_step1": steps["step1"], "questions_step2": steps["step2"],
                     "questions_step3": steps["step3"], "questions_step0": steps["step0_no_overlapping_chunk"],
                     "pct_split_over_chunks": (100.0 - inside) if inside is not None else None,
                     "pct_step2_or_3": 100.0 * (steps["step2"] + steps["step3"]) / n if n else None,
                     "mean_relevant_per_question": ls["labels_per_question"]["mean"], "max_relevant_per_question": ls["labels_per_question"]["max"],
                     "n_empty_questions": ls["n_empty_questions"]})
    return rows


def build_table(prepared_root: Path, recompute_lengths: bool = True) -> List[Dict[str, Any]]:
    """Rows of every prepared folder below ``prepared_root`` (folders without a manifest.json are incomplete and skipped), sorted by configuration."""
    rows: List[Dict[str, Any]] = []
    root = Path(prepared_root)
    for d in sorted(p for p in root.glob("*") if (p / "manifest.json").is_file()) if root.is_dir() else []:
        rows += folder_rows(d, recompute_lengths)
    return sorted(rows, key=lambda r: (str(r["chunker"]), int(r["size"] or 0), int(r["overlap"] or 0), float(r["cover_level"])))


def write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared-root", required=True, help="folder with the prepared configurations (<artifacts>/prepared)")
    ap.add_argument("--out", required=True, help="table_label_stats.csv")
    ap.add_argument("--no-recompute", action="store_true", help="leave the median empty if the diagnostics lack it (no tokenizer needed)")
    args = ap.parse_args(argv)
    rows = build_table(Path(args.prepared_root), not args.no_recompute)
    write_table(Path(args.out), rows)
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One CSV row per run folder: metrics under the span and the lexical labels, span coverage, truncation, timing,
prompts, and the diagnostics of the prepared configuration.

  python -m tools.collect_runs --tests-root artifacts/pilot/tests --out pilot_runs.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def run_row(run: Path) -> Optional[Dict[str, Any]]:
    info = _load(run / "run_info.json")
    outs = [d for d in (run / "results").glob("*") if d.is_dir() and (d / "metrics_summary.json").exists()]
    if not info or len(outs) != 1:
        return None
    out = outs[0]
    ms, man, tr = _load(out / "metrics_summary.json"), _load(out / "run_manifest.json"), _load(out / "truncation_report.json")
    prepared = Path(info["prepared_dir"])
    diag = {}
    dcsv = prepared / "coverage_diagnostics.csv"
    if dcsv.exists():
        with open(dcsv, newline="", encoding="utf-8") as f:
            diag = next(csv.DictReader(f), {})
    agg, lex = ms.get("aggregate", {}), (ms.get("other_label_sets") or {}).get("qrels_lexical.jsonl", {})
    t = man.get("timing_s") or {}
    pm = _load(prepared / "manifest.json").get("timing_s", {})
    row: Dict[str, Any] = {
        "model": info["model"], "chunker": info["chunker"], "size": info["size"], "overlap": info["overlap"],
        "labels": ms.get("qrels_file"), "prepared": prepared.name,
        "n_chunks": diag.get("n_chunks"), "mean_chunk_tokens_ref": diag.get("mean_chunk_tokens"),
        "pct_span_inside_one_chunk": diag.get("pct_span_inside_one_chunk"),
        "mean_relevant_span": diag.get("mean_relevant_span"), "mean_relevant_lexical": diag.get("mean_relevant_lexical"),
        "n_zero_relevant_span": diag.get("n_zero_relevant_span"), "n_zero_relevant_lexical": diag.get("n_zero_relevant_lexical"),
    }
    for k in ("ndcg_at_10", "recall_at_10", "geom_at_10", "mrr_at_10", "map_at_10", "ndcg_at_1", "recall_at_100"):
        row[f"span_{k}"] = agg.get(k)
        row[f"lexical_{k}"] = lex.get(k)
    for k, v in (ms.get("span_coverage") or {}).items():
        row[k] = v
    if tr:
        row.update({"max_seq_length": tr["max_seq_length"], "n_chunks_truncated": tr["passages"]["n_truncated"],
                    "pct_chunks_truncated": tr["passages"]["pct_truncated"], "mean_tokens_own_tokenizer": tr["passages"]["mean_tokens"]})
    row.update({"use_prompts": man.get("use_prompts"), "prompt_query": (man.get("prompts") or {}).get("query"),
                "prompt_passage": (man.get("prompts") or {}).get("passage"),
                "mteb_evaluation_time_s": t.get("mteb_evaluation_time_s"), "model_load_s": t.get("model_load_s"),
                "encode_passage_s": t.get("encode_passage_s"), "encode_query_s": t.get("encode_query_s"),
                "search_metrics_and_dump_s": t.get("search_metrics_and_dump_s"),
                "bm25_index_s": t.get("bm25_index_s"), "bm25_search_s": t.get("bm25_search_s"),
                "chunking_s": pm.get("chunking_s"), "qrels_span_s": pm.get("qrels_span_s"), "qrels_lexical_s": pm.get("qrels_lexical_s"),
                "git_commit": (man.get("git") or {}).get("commit"), "gpu": (man.get("hardware") or {}).get("device_in_use"),
                "batch_size": man.get("batch_size")})
    return row


def collect(tests_root: Path) -> List[Dict[str, Any]]:
    rows = [r for d in sorted(Path(tests_root).iterdir()) if d.is_dir() for r in [run_row(d)] if r]
    return sorted(rows, key=lambda r: (r["chunker"], r["size"], r["overlap"], r["model"]))


def write_csv(path, rows: List[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = collect(Path(args.tests_root))
    write_csv(args.out, rows)
    print(f"{len(rows)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

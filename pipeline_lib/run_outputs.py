"""Per-query outputs of a run, from its saved predictions: gzipped TREC run, per-query metrics (primary label set and
every other qrels file of the folder), span coverage, and the cross-check against the metrics MTEB stored."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import coverage_diagnostics as qcov  # noqa: E402
from tools import metrics as qmetrics  # noqa: E402


def write_per_query_outputs(model_out_dir: Path, data_dir: Path, qrels_name: str, depth: int = 100,
                            run_name: str = "run") -> Optional[Dict[str, Any]]:
    """Writes into ``model_out_dir``: run_top<depth>.trec.gz, per_query_metrics.csv (``qrels_name``),
    per_query_metrics__<other qrels>.csv, span_coverage.csv, metrics_summary.json (with the MTEB cross-check).
    Needs exactly one ``*_predictions.json`` in ``model_out_dir``; returns the summary (or None without one)."""
    model_out_dir, data_dir = Path(model_out_dir), Path(data_dir)
    preds = sorted(model_out_dir.glob("*_predictions.json"))
    if len(preds) != 1:
        return None
    run = qmetrics.load_run(preds[0])
    qrels = qmetrics.load_qrels(data_dir / qrels_name)
    qids = qmetrics.load_query_ids(data_dir / "queries.jsonl")
    n_lines = qmetrics.write_trec_run(run, model_out_dir / f"run_top{depth}.trec.gz", tag=run_name, depth=depth)
    pq = qmetrics.per_query_metrics(run, qrels, qmetrics.DEFAULT_KS, qids)
    qmetrics.write_per_query_csv(model_out_dir / "per_query_metrics.csv", pq)
    summary: Dict[str, Any] = {"qrels_file": qrels_name, "aggregate": qmetrics.aggregate(pq), "run_lines": n_lines}
    # every other label set of the folder (e.g. the lexical qrels next to the span qrels): same ranking, other labels
    for other in sorted(data_dir.glob("qrels_*.jsonl")):
        if other.name == qrels_name:
            continue
        opq = qmetrics.per_query_metrics(run, qmetrics.load_qrels(other), qmetrics.DEFAULT_KS, qids)
        qmetrics.write_per_query_csv(model_out_dir / f"per_query_metrics__{other.stem}.csv", opq)
        summary.setdefault("other_label_sets", {})[other.name] = qmetrics.aggregate(opq)
    # chunking-independent span coverage of the top-k chunks (needs the overlap table of the span stage)
    table = data_dir / "overlap_table.jsonl"
    if table.exists():
        cov = qcov.span_coverage(run, table, qids, qcov.COVERAGE_KS, qcov.corpus_chunk_chars(data_dir / "corpus.jsonl"))
        qcov.write_span_coverage_csv(model_out_dir / "span_coverage.csv", cov)
        summary["span_coverage"] = qcov.mean_of(cov)
    stored = sorted(model_out_dir.rglob("ChemQuest.json"))
    if stored:
        summary["cross_check"] = qmetrics.cross_check(run, qrels, stored[0], qids, max_k=depth)
        if not summary["cross_check"]["ok"]:
            print(f"[WARN] metrics cross-check against MTEB failed: {summary['cross_check']}")
    (model_out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

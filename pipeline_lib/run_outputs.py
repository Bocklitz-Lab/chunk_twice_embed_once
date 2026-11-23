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


# --------------------------------------------------------------------------------------
# MTEB-style aggregate scores and ChemQuest.json for runs that do not go through MTEB (BM25, hybrid, rerank)
# --------------------------------------------------------------------------------------
KS = [1, 3, 5, 10, 20, 100]


def mteb_style_scores(run, qrels, query_ids, ks=KS) -> Dict[str, float]:
    """Metrics with MTEB's own functions (pytrec_eval ndcg/map/recall/P and MTEB's MRR), rounded to 5 decimals.
    ``run`` is {qid: {doc: score}}, ``qrels`` {qid: {doc: relevance}}; a question without a label stays in (scores 0).

    Tied scores (frequent after a fusion or with fp16 logits) are broken like in every other run of the repo and like trec_eval:
    score descending, then chunk id descending. The ranking is turned into strictly decreasing scores first, because MTEB's own
    ``mrr`` sorts stably (ties in input order) while pytrec_eval breaks them by id: without this the MRR of a run with ties would
    differ from the per-query metrics written next to it."""
    from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator
    from mteb.evaluation.evaluators.utils import mrr

    full = {q: (qrels.get(q) or {"__none__": 0}) for q in query_ids}
    results = {q: {d: -float(i) for i, (d, _s) in enumerate(qmetrics.ranked(run[q]))} for q in query_ids}
    ndcg, _map, recall, prec, _ = RetrievalEvaluator.evaluate(full, results, list(ks))
    mr = mrr(full, results, list(ks))
    out: Dict[str, float] = {}
    for k in ks:
        out.update({f"ndcg_at_{k}": ndcg[f"NDCG@{k}"], f"map_at_{k}": _map[f"MAP@{k}"], f"recall_at_{k}": recall[f"Recall@{k}"],
                    f"precision_at_{k}": prec[f"P@{k}"], f"mrr_at_{k}": mr[f"MRR@{k}"]})
    return out


def write_chemquest_json(out_dir: Path, scores: Dict[str, Any], evaluation_time: Optional[float], *, method: str,
                         task_name: str = "ChemQuest", file_name: Optional[str] = None,
                         dataset_revision: str = "local", split: str = "test") -> Path:
    """``<task>.json`` in the layout MTEB writes (scores.<split>[0] with ``main_score`` = nDCG@10, top-level ``evaluation_time``).
    ``scores`` is one dict, or {split: dict} for a task with several evaluation splits."""
    per_split = dict(scores) if all(isinstance(v, dict) for v in scores.values()) and scores else {split: scores}
    per_split = {sp: dict(sc) for sp, sc in per_split.items()}
    for sc in per_split.values():
        sc.update({"main_score": sc["ndcg_at_10"], "hf_subset": "default", "languages": ["eng-Latn"]})
    path = Path(out_dir) / (file_name or f"{task_name}.json")
    path.write_text(json.dumps({
        "dataset_revision": dataset_revision, "task_name": task_name, "mteb_version": f"{method} (metrics by mteb RetrievalEvaluator)",
        "evaluation_time": evaluation_time, "kg_co2_emissions": None,
        "scores": {sp: [sc] for sp, sc in per_split.items()}}, indent=2), encoding="utf-8")
    return path


def read_eval_time(model_out_dir: Path) -> Optional[float]:
    """``evaluation_time`` of a finished run folder (results/<model>/): the run manifest's ``mteb_evaluation_time_s`` (dense,
    BM25, hybrid, rerank all write it), else the ``evaluation_time`` of the ChemQuest.json below the folder. None if neither exists."""
    model_out_dir = Path(model_out_dir)
    man = model_out_dir / "run_manifest.json"
    if man.is_file():
        t = (json.loads(man.read_text(encoding="utf-8")).get("timing_s") or {}).get("mteb_evaluation_time_s")
        if isinstance(t, (int, float)):
            return float(t)
    for p in sorted(model_out_dir.rglob("ChemQuest.json")):
        t = json.loads(p.read_text(encoding="utf-8")).get("evaluation_time")
        if isinstance(t, (int, float)):
            return float(t)
    return None

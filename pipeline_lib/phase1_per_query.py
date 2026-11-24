"""Per-query results of a Phase 1 run (query-level intervals and shortlist stability of ``make report``).

MTEB (``save_predictions=True``) writes ``<Task>_default_predictions.json`` into the model's output folder: the scores of every document for every query of
the LAST evaluation pass of the task. For ChemHotpotQARetrieval the passes are train, dev, test, so the file holds the ``test`` queries. Nothing in the MTEB
scores changes: the file is only dumped after the retrieval.

``write_task`` turns it into, next to the predictions,

  run_top100_<Task>.trec.gz     the ranking of every query of the configured split (top 100, gzipped TREC)
  per_query_<Task>.csv          nDCG@10, Recall@10, ... and Geom@10 per query (tools/metrics.py, pytrec_eval)
  per_query_<Task>.json         task, split, number of queries, the aggregate and the cross-check

and CHECKS that the mean per-query nDCG@10 and Recall@10 equal the scores MTEB stored for the configured split (``params.phase1.eval_split``, tolerance
``TOLERANCE``). If the predictions belong to another split (the configured split is not the last pass) it raises ``Phase1SplitError``: the file is never
read as the wrong split.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from pipeline_lib import phase1_split
from tools import metrics as qmetrics

TOLERANCE = 2e-5          # MTEB rounds its scores to 5 decimals


def _mteb_scores(model_out_dir: Path, task_name: str) -> Dict[str, Any]:
    hits = sorted(model_out_dir.glob(f"**/{task_name}.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} files {task_name}.json below {model_out_dir} (need exactly one)")
    return json.loads(hits[0].read_text(encoding="utf-8"))["scores"]


def write_task(model_out_dir: Any, task: Any, eval_split: Mapping[str, str], depth: int = 100) -> Dict[str, Any]:
    """Per-query files of one finished task (an MTEB task object with the loaded ``queries`` / ``relevant_docs``)."""
    model_out_dir = Path(model_out_dir)
    name = task.metadata.name
    split = phase1_split.split_of(name, eval_split)
    pred = model_out_dir / f"{name}_default_predictions.json"
    if not pred.is_file():
        raise FileNotFoundError(f"{pred} is missing (MTEB did not save predictions; it skips tasks whose results exist)")
    stored = phase1_split.pick_split(_mteb_scores(model_out_dir, name), name, eval_split, where=f"{name}.json")
    qrels = {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in task.relevant_docs[split].items()}
    queries = [str(q) for q in task.queries[split] if str(q) in qrels]
    run = qmetrics.load_run(pred)
    pq = qmetrics.per_query_metrics(run, qrels, qmetrics.DEFAULT_KS, queries)
    agg = qmetrics.aggregate(pq)
    check = {k: {"per_query_mean": agg[k], "mteb": float(stored[k]), "abs_diff": abs(agg[k] - float(stored[k]))} for k in ("ndcg_at_10", "recall_at_10")}
    if any(v["abs_diff"] > TOLERANCE for v in check.values()):
        raise phase1_split.Phase1SplitError(
            f"{name}: the saved predictions do not reproduce the {split!r} scores of {name}.json ({check}). MTEB saves the predictions of the LAST split it "
            "evaluated; the configured split is another one, so per-query results cannot be taken from this file.")
    run_cfg = {q: run[q] for q in queries}
    qmetrics.write_trec_run(run_cfg, model_out_dir / f"run_top{depth}_{name}.trec.gz", tag=name, depth=depth)
    qmetrics.write_per_query_csv(model_out_dir / f"per_query_{name}.csv", pq)
    summary = {"task": name, "split": split, "n_queries": len(queries), "aggregate": agg, "mteb_check": check, "predictions_file": pred.name}
    (model_out_dir / f"per_query_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

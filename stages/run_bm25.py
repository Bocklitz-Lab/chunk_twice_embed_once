"""Stage 6 for the BM25 baseline: the same inputs and the same outputs as ``stages/run_eval.py``.

Reads the generated stage 6 config (the task's data_dir / qrels_file, output_root, top_k, seed, optional ``bm25`` block),
builds a BM25 index over the prepared corpus, ranks every query and writes into
``<output_root>/bm25_<rank_bm25 version>/``: ChemQuest_default_predictions.json (top k), ChemQuest.json (metrics computed by
MTEB's own evaluator, with ``evaluation_time`` = index + search seconds), run_top100.trec.gz, per_query_metrics*.csv,
span_coverage.csv, metrics_summary.json, truncation-free by construction, run_manifest.json; then summary.json/csv last.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from pipeline_lib import run_outputs  # noqa: E402
from pipeline_lib.bm25_retriever import BM25Retriever  # noqa: E402
from pipeline_lib.run_manifest import Stopwatch, build_manifest, package_versions, set_seeds, write_manifest  # noqa: E402
from tools import metrics as qmetrics  # noqa: E402

MODEL_NAME = "bm25"
KS = [1, 3, 5, 10, 20, 100]


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def mteb_style_scores(run, qrels, query_ids) -> Dict[str, float]:
    """Metrics with MTEB's own functions (pytrec_eval ndcg/map/recall/P and MTEB's MRR), rounded to 5 decimals."""
    from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator
    from mteb.evaluation.evaluators.utils import mrr

    full = {q: (qrels.get(q) or {"__none__": 0}) for q in query_ids}      # a question without a label stays (scores 0)
    results = {q: dict(run[q]) for q in query_ids}
    ndcg, _map, recall, prec, _ = RetrievalEvaluator.evaluate(full, results, KS)
    mr = mrr(full, results, KS)
    out: Dict[str, float] = {}
    for k in KS:
        out.update({f"ndcg_at_{k}": ndcg[f"NDCG@{k}"], f"map_at_{k}": _map[f"MAP@{k}"], f"recall_at_{k}": recall[f"Recall@{k}"],
                    f"precision_at_{k}": prec[f"P@{k}"], f"mrr_at_{k}": mr[f"MRR@{k}"]})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BM25 baseline on a prepared task (same outputs as stages/run_eval.py).")
    ap.add_argument("--config", required=True, help="generated stage 6 config")
    args = ap.parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    spec = next((t for t in raw["tasks"] if isinstance(t, dict) and (t.get("kwargs") or {}).get("data_dir")), None)
    if spec is None:
        raise SystemExit("ERROR: BM25 needs a local task (tasks[].kwargs.data_dir)")
    data_dir = Path(spec["kwargs"]["data_dir"]).expanduser()
    qrels_name = spec["kwargs"].get("qrels_file") or "qrels.jsonl"
    depth = int(raw.get("top_k") or 100)
    bm = raw.get("bm25") or {}
    seeds = set_seeds(raw["seed"]) if raw.get("seed") is not None else None

    corpus = {str(r["id"]): r for r in _read_jsonl(data_dir / "corpus.jsonl")}
    queries = {str(r["id"]): str(r.get("text") or "") for r in _read_jsonl(data_dir / "queries.jsonl")}
    sw = Stopwatch()
    retr = BM25Retriever(k1=float(bm.get("k1", 1.5)), b=float(bm.get("b", 0.75)), lowercase=bool(bm.get("lowercase", True)))
    with sw.section("bm25_total_s"):
        retr.index(corpus)
        run = retr.search(queries, top_k=depth)
    eval_time = retr.timing["bm25_index_s"] + retr.timing["bm25_search_s"]

    rev = package_versions(["rank-bm25"]).get("rank-bm25") or "unknown"
    out_root = Path(raw["output_root"])
    out = out_root / f"{MODEL_NAME}_{rev}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ChemQuest_default_predictions.json").write_text(json.dumps(run), encoding="utf-8")
    qrels = qmetrics.load_qrels(data_dir / qrels_name)
    scores = mteb_style_scores(run, qrels, list(queries))
    scores.update({"main_score": scores["ndcg_at_10"], "hf_subset": "default", "languages": ["eng-Latn"]})
    (out / "ChemQuest.json").write_text(json.dumps({
        "dataset_revision": "local", "task_name": "ChemQuest", "mteb_version": "bm25 (metrics by mteb RetrievalEvaluator)",
        "evaluation_time": eval_time, "kg_co2_emissions": None, "scores": {"test": [scores]}}, indent=2), encoding="utf-8")
    per_query = run_outputs.write_per_query_outputs(out, data_dir, qrels_name, depth=depth, run_name=MODEL_NAME)
    files = [data_dir / n for n in ("corpus.jsonl", "queries.jsonl", qrels_name)]
    write_manifest(out / "run_manifest.json", build_manifest(
        repo_root=_REPO_ROOT, config_path=args.config, resolved_config=raw,
        model={"name": MODEL_NAME, "revision": rev, "k1": retr.k1, "b": retr.b, "lowercase": retr.lowercase,
               "text": "chunk text only (no title)", "tokens": "regex \\w+ on lower-cased text, no stemming, no stop words"},
        use_prompts=False, prompts=None, batch_size=None, seeds=seeds, data_files=files,
        timing={**sw.seconds, **retr.timing, "mteb_evaluation_time_s": eval_time},
        extra={"top_k": depth, "per_query_outputs": per_query}))
    row = {"model": MODEL_NAME, "revision": rev, "normalize": "", "batch_size": "", "task": "ChemQuest", "main_score": scores["main_score"]}
    (out_root / "summary.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    (out_root / "summary.csv").write_text("model,revision,normalize,batch_size,task,main_score\n"
                                          f"{MODEL_NAME},{rev},,,ChemQuest,{scores['main_score']}\n", encoding="utf-8")
    print(f"BM25: nDCG@10={scores['ndcg_at_10']} Recall@10={scores['recall_at_10']} evaluation_time={eval_time:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

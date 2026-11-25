"""Stage 6 for the BM25 baseline: the same inputs and the same outputs as ``stages/run_eval.py``.

Reads the generated stage 6 config (the task's data_dir / qrels_file, output_root, top_k, seed, optional ``bm25`` block),
builds a BM25 index over the prepared corpus, ranks every query and writes into
``<output_root>/bm25_<rank_bm25 version>/``: ChemQuest_default_predictions.json (top k), ChemQuest.json (metrics computed by
MTEB's own evaluator, with ``evaluation_time`` = index + search + metrics + prediction dump, the same stages MTEB's
evaluation_time covers for a dense model, model loading excluded), run_top100.trec.gz, per_query_metrics*.csv,
span_coverage.csv, metrics_summary.json, truncation-free by construction, run_manifest.json; then summary.json/csv last.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
KS = run_outputs.KS
mteb_style_scores = run_outputs.mteb_style_scores     # kept here for the callers and tests that import it from this stage


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BM25 baseline on a prepared task (same outputs as stages/run_eval.py).")
    ap.add_argument("--config", required=True, help="generated stage 6 config")
    args = ap.parse_args(argv)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    spec = next((t for t in raw["tasks"] if isinstance(t, dict) and (t.get("kwargs") or {}).get("data_dir")), None)
    if spec is None:
        raise SystemExit("ERROR: BM25 needs a local task (tasks[].kwargs.data_dir)")
    data_dir = Path(spec["kwargs"]["data_dir"]).expanduser()
    qrels_name = spec["kwargs"].get("qrels_file")
    if not qrels_name:
        raise SystemExit("ERROR: tasks[].kwargs.qrels_file is not set (labeling.qrels_file): there is no default label file")
    depth = int(raw.get("top_k") or 100)
    bm = raw.get("bm25") or {}
    seeds = set_seeds(raw["seed"]) if raw.get("seed") is not None else None

    corpus = {str(r["id"]): r for r in _read_jsonl(data_dir / "corpus.jsonl")}
    queries = {str(r["id"]): str(r.get("text") or "") for r in _read_jsonl(data_dir / "queries.jsonl")}
    sw = Stopwatch()
    retr = BM25Retriever(k1=float(bm.get("k1", 1.5)), b=float(bm.get("b", 0.75)), lowercase=bool(bm.get("lowercase", True)))
    out_root = Path(raw["output_root"])
    rev = package_versions(["rank-bm25"]).get("rank-bm25") or "unknown"
    out = out_root / f"{MODEL_NAME}_{rev}"
    out.mkdir(parents=True, exist_ok=True)
    qrels = qmetrics.load_qrels(data_dir / qrels_name)
    # evaluation_time = what MTEB's evaluation_time covers for a dense model: retrieval (here index + search), the metrics and
    # the dump of the ranking. Reading the corpus / queries files is not counted (MTEB loads the dataset before it starts the clock).
    t_eval = time.perf_counter()
    with sw.section("bm25_total_s"):
        retr.index(corpus)
        run = retr.search(queries, top_k=depth)
    with sw.section("metrics_s"):
        scores = mteb_style_scores(run, qrels, list(queries))
    with sw.section("dump_s"):
        (out / "ChemQuest_default_predictions.json").write_text(json.dumps(run), encoding="utf-8")
    eval_time = time.perf_counter() - t_eval
    run_outputs.write_chemquest_json(out, scores, eval_time, method="bm25")
    per_query = run_outputs.write_per_query_outputs(out, data_dir, qrels_name, depth=depth, run_name=MODEL_NAME)
    files = [data_dir / n for n in ("corpus.jsonl", "queries.jsonl", qrels_name)]
    write_manifest(out / "run_manifest.json", build_manifest(
        repo_root=_REPO_ROOT, config_path=args.config, resolved_config=raw,
        model={"name": MODEL_NAME, "revision": rev, "k1": retr.k1, "b": retr.b, "lowercase": retr.lowercase,
               "text": "chunk text only (no title)", "evaluation_time": "index + search + metrics + dump", "tokens": "regex \\w+ on lower-cased text, no stemming, no stop words"},
        use_prompts=False, prompts=None, batch_size=None, seeds=seeds, data_files=files,
        timing={**sw.seconds, **retr.timing, "mteb_evaluation_time_s": eval_time},     # index + search + metrics + dump
        extra={"top_k": depth, "per_query_outputs": per_query}))
    row = {"model": MODEL_NAME, "revision": rev, "normalize": "", "batch_size": "", "task": "ChemQuest", "main_score": scores["ndcg_at_10"]}
    (out_root / "summary.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    (out_root / "summary.csv").write_text("model,revision,normalize,batch_size,task,main_score\n"
                                          f"{MODEL_NAME},{rev},,,ChemQuest,{scores['ndcg_at_10']}\n", encoding="utf-8")
    print(f"BM25: nDCG@10={scores['ndcg_at_10']} Recall@10={scores['recall_at_10']} evaluation_time={eval_time:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

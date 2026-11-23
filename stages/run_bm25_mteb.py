"""BM25 baseline on the external MTEB retrieval tasks of Phase 1 (ChemNQRetrieval, ChemHotpotQARetrieval).

The task is loaded like MTEB loads it for a dense model (``mteb.get_tasks`` + ``load_data``); there is no chunking: the corpus is
the task's own passages. The text that is indexed is what MTEB hands to a dense model (the corpus string, or
``title + " " + text`` for a corpus of dicts; ``BM25Retriever(title_in_text=True)``). Metrics come from MTEB's own functions
(``run_outputs.mteb_style_scores``). The retriever, tokenizer and k1 / b are those of the ChemQuests BM25 (``stage6.bm25``).

Written in the layout of a Phase 1 model run, so ``stages/screen_top_k_models.py`` and the Geom@10 code read it like a dense model:

  <output root>/bm25_<rank_bm25 version>/bm25/<version>/<Task>.json        scores.<split>[0] (nDCG/MAP/Recall/P/MRR @ 1,3,5,10,20,100),
                                                                          evaluation_time = index + search + metrics
  <output root>/bm25_<rank_bm25 version>/bm25/<version>/model_meta.json   name "bm25", revision <version>
  <output root>/bm25_<rank_bm25 version>/<Task>_default_predictions.json  top-k ranking (like MTEB's save_predictions)
  <output root>/bm25_<rank_bm25 version>/run_manifest_<Task>.json         config, git, packages, hardware, timing

``evaluation_time`` is index + search + metrics, as MTEB's evaluation_time of the Phase 1 dense runs (they saved no predictions,
so the dump is not counted here either). Finished tasks are skipped (``--force`` re-runs). The default output root is
``<artifacts>/embedding_models_screening/baselines`` and NOT the folder stage 2 screens: BM25 must not take part in the
selection of the Phase 2 shortlist (stage 2 also ignores ``bm25*`` unless ``include_baselines`` is set).

  python -m stages.run_bm25_mteb                                   # tasks and paths from configs/main_config.yaml (params.external)
  python -m stages.run_bm25_mteb --tasks ChemNQRetrieval --output-root /tmp/bl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from pipeline_lib import run_outputs  # noqa: E402
from pipeline_lib.bm25_retriever import BM25Retriever  # noqa: E402
from pipeline_lib.run_manifest import Stopwatch, build_manifest, package_versions, set_seeds, write_manifest  # noqa: E402

MODEL_NAME = "bm25"


def load_mteb_task(name: str):
    """The loaded MTEB task object (corpus / queries / relevant_docs per split)."""
    import mteb

    tasks = mteb.get_tasks(tasks=[name])
    if len(tasks) != 1:
        raise ValueError(f"mteb.get_tasks({name!r}) returned {len(tasks)} tasks")
    task = tasks[0]
    task.load_data()
    return task


def split_data(task, split: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """(corpus {id: {"title", "text"}}, queries {id: text}, qrels {qid: {doc: relevance}}) of one split of a monolingual task.
    Only queries that have relevance judgements are evaluated (MTEB does the same)."""
    if getattr(task, "is_multilingual", False):
        raise NotImplementedError("multilingual tasks are not supported (the external tasks are monolingual)")
    corpus = {}
    for did, doc in task.corpus[split].items():
        corpus[str(did)] = {"title": doc.get("title", ""), "text": doc.get("text", "")} if isinstance(doc, dict) else {"title": "", "text": str(doc)}
    qrels = {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in task.relevant_docs[split].items()}
    queries = {str(q): (t if isinstance(t, str) else str(t)) for q, t in task.queries[split].items() if str(q) in qrels}
    qrels = {q: r for q, r in qrels.items() if q in queries}
    return corpus, queries, qrels


def run_task(name: str, leaf: Path, *, k1: float = 1.5, b: float = 0.75, lowercase: bool = True, top_k: int = 100,
             force: bool = False, seed: Optional[int] = None, config_path: Optional[str] = None,
             loader: Optional[Callable[[str], Any]] = None, rev: Optional[str] = None) -> Dict[str, Any]:
    """BM25 on one external task; writes ``<leaf>/<name>.json`` and the files next to it. Returns a status dict."""
    rev = rev or package_versions(["rank-bm25"]).get("rank-bm25") or "unknown"
    result_json = leaf / f"{name}.json"
    if result_json.is_file() and result_json.stat().st_size > 0 and not force:
        print(f"skip (finished): {result_json}")
        return {"task": name, "status": "skipped", "path": str(result_json)}
    seeds = set_seeds(seed) if seed is not None else None
    task = (loader or load_mteb_task)(name)
    sw = Stopwatch()
    per_split: Dict[str, Dict[str, float]] = {}
    preds: Dict[str, Dict[str, Dict[str, float]]] = {}
    retr = BM25Retriever(k1=k1, b=b, lowercase=lowercase, title_in_text=True)
    n_docs = n_queries = 0
    eval_time = 0.0
    for split in task.metadata.eval_splits:
        corpus, queries, qrels = split_data(task, split)
        n_docs, n_queries = n_docs + len(corpus), n_queries + len(queries)
        t0 = time.perf_counter()                                  # index + search + metrics; the data is loaded before the clock starts
        with sw.section(f"bm25_total_{split}_s"):
            retr.index(corpus)
            run = retr.search(queries, top_k=top_k)
        with sw.section(f"metrics_{split}_s"):
            per_split[split] = run_outputs.mteb_style_scores(run, qrels, list(queries))
        eval_time += time.perf_counter() - t0
        preds[split] = run
    leaf.mkdir(parents=True, exist_ok=True)
    parent = leaf.parents[1]                                      # <root>/bm25_<rev>/   (leaf = parent/bm25/<rev>)
    for split, run in preds.items():                              # MTEB: <task>_default_predictions.json (one split: no suffix)
        fname = f"{name}_default_predictions.json" if len(preds) == 1 else f"{name}_{split}_default_predictions.json"
        (parent / fname).write_text(json.dumps(run), encoding="utf-8")
    meta = {"name": MODEL_NAME, "revision": rev, "release_date": None, "languages": ["eng-Latn"], "n_parameters": None,
            "memory_usage_mb": None, "max_tokens": None, "embed_dim": None, "license": None, "open_weights": True,
            "public_training_code": None, "public_training_data": None, "framework": ["rank_bm25"], "reference": "https://pypi.org/project/rank-bm25/",
            "similarity_fn_name": None, "use_instructions": None, "training_datasets": None, "adapted_from": None, "superseded_by": None,
            "is_cross_encoder": None, "modalities": ["text"], "loader": None}
    (leaf / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ds_rev = (getattr(task.metadata, "dataset", None) or {}).get("revision", "unknown")
    run_outputs.write_chemquest_json(leaf, per_split, eval_time, method="bm25", task_name=name, dataset_revision=ds_rev)
    write_manifest(parent / f"run_manifest_{name}.json", build_manifest(
        repo_root=_REPO_ROOT, config_path=config_path, resolved_config=None,
        model={"name": MODEL_NAME, "revision": rev, "k1": k1, "b": b, "lowercase": lowercase,
               "text": "corpus string as MTEB passes it to a dense model (title + ' ' + text for dict entries)",
               "tokens": "regex \\w+ on lower-cased text, no stemming, no stop words",
               "evaluation_time": "index + search + metrics (as the Phase 1 dense runs: no dump)"},
        use_prompts=False, prompts=None, batch_size=None, seeds=seeds, data_files=None,
        timing={**sw.seconds, **retr.timing, "mteb_evaluation_time_s": eval_time},
        extra={"task": name, "splits": list(per_split), "n_docs": n_docs, "n_queries": n_queries, "top_k": top_k,
               "dataset": getattr(task.metadata, "dataset", None)}))
    main_split = next(iter(per_split))
    print(f"BM25 {name}: nDCG@10={per_split[main_split]['ndcg_at_10']} Recall@10={per_split[main_split]['recall_at_10']} "
          f"evaluation_time={eval_time:.1f}s ({n_docs} docs, {n_queries} queries) -> {result_json}")
    return {"task": name, "status": "done", "path": str(result_json), "evaluation_time": eval_time, "ndcg_at_10": per_split[main_split]["ndcg_at_10"]}


def _append_summary(root: Path, rev: str, results: List[Dict[str, Any]]) -> None:
    """summary.json / summary.csv in the layout of stage 1 (model, revision, normalize, batch_size, task, main_score)."""
    path = root / "summary.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for r in results:
        if r["status"] != "done":
            continue
        rows = [x for x in rows if not (x.get("model") == MODEL_NAME and x.get("task") == r["task"])]
        rows.append({"model": MODEL_NAME, "revision": rev, "normalize": "", "batch_size": "", "task": r["task"], "main_score": r["ndcg_at_10"]})
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    headers = ["model", "revision", "normalize", "batch_size", "task", "main_score"]
    (root / "summary.csv").write_text(",".join(headers) + "\n" + "".join(",".join(str(x.get(h, "")) for h in headers) + "\n" for x in rows), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-config", default="configs/main_config.yaml")
    ap.add_argument("--artifacts-root", help="default: params.paths.artifacts_root")
    ap.add_argument("--tasks", nargs="+", help="default: params.external.tasks")
    ap.add_argument("--output-root", help="default: <artifacts root>/<params.external.output_subdir>")
    ap.add_argument("--k1", type=float)
    ap.add_argument("--b", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    doc = yaml.safe_load(Path(args.main_config).read_text(encoding="utf-8"))
    params, ext = doc["params"], (doc["params"].get("external") or {})
    bm = (doc.get("stage6") or {}).get("bm25") or {}
    root_art = args.artifacts_root or params["paths"]["artifacts_root"]
    tasks = args.tasks or ext.get("tasks") or ["ChemNQRetrieval", "ChemHotpotQARetrieval"]
    out_root = Path(args.output_root or Path(root_art) / ext.get("output_subdir", "embedding_models_screening/baselines"))
    rev = package_versions(["rank-bm25"]).get("rank-bm25") or "unknown"
    leaf = out_root / f"{MODEL_NAME}_{rev}" / MODEL_NAME / rev
    out_root.mkdir(parents=True, exist_ok=True)
    results = [run_task(t, leaf, k1=args.k1 if args.k1 is not None else float(bm.get("k1", 1.5)),
                        b=args.b if args.b is not None else float(bm.get("b", 0.75)), lowercase=bool(bm.get("lowercase", True)),
                        top_k=args.top_k or int(ext.get("top_k", 100)), force=args.force, seed=params.get("seed"),
                        config_path=args.main_config, rev=rev) for t in tasks]
    _append_summary(out_root, rev, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

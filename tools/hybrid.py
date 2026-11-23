"""Hybrid retrieval by reciprocal rank fusion (RRF) of saved rankings. No embedding, no model: it reads two (or more)
rankings that already exist and writes a new run with the same outputs as a normal run.

  score(d) = sum over the input rankings of 1 / (k + rank(d)),   rank = 1 for the best chunk, k = 60 (Cormack et al. 2009)

A chunk that is not in the top ``depth`` of a ranking gets no contribution from it. Ties are broken like every other run
(score descending, chunk id descending). Both inputs must come from the same prepared configuration (same key: same
chunks, queries and labels) and must rank the same queries.

Run folder written  <tests root>/hybrid_rrf<k>_<dense dir>_<sparse dir>_<chunker>_c<L>_o<O>/ with run_info.json,
links to chunks/ and mteb_task/, results/summary.json and results/<name>/{ChemQuest.json, ChemQuest_default_predictions.json,
run_top100.trec.gz, per_query_metrics*.csv, span_coverage.csv, metrics_summary.json, run_manifest.json}.

evaluation_time (ChemQuest.json, run_manifest timing_s.mteb_evaluation_time_s) = dense evaluation_time + BM25 evaluation_time +
fusion_s, where fusion_s = reading the two rankings + fusion + metrics + dump of the fused ranking. The three parts are stored
separately in run_manifest.json (timing_s.dense_evaluation_time_s, bm25_evaluation_time_s, fusion_s). The hybrid needs both
inputs, so their cost is part of its cost; if an input run has no time the sum is null (and a warning is printed).

  python -m tools.hybrid --tests-root artifacts/pilot/tests --dense intfloat/e5-large-v2 BAAI/bge-m3 --sparse bm25
  python -m tools.hybrid --tests-root ... --dense intfloat/e5-large-v2 --sparse bm25 --k 60 --depth 100 --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import run_outputs  # noqa: E402
from pipeline_lib.run_manifest import derived_run_environment, git_info, package_versions, set_seeds, sha256_file  # noqa: E402
from tools import metrics as qmetrics  # noqa: E402

Run = Dict[str, Dict[str, float]]
DEFAULT_K = 60


def fuse(runs: Sequence[Run], k: int = DEFAULT_K, depth: Optional[int] = 100) -> Run:
    """Reciprocal rank fusion of rankings ({qid: {doc: score}}, higher is better). The rank is the position after
    ``qmetrics.ranked`` (score descending, id descending); only the first ``depth`` ranks count (None: all)."""
    if not runs:
        raise ValueError("nothing to fuse")
    if k < 0:
        raise ValueError("k must be >= 0")
    qids = set(runs[0])
    for r in runs[1:]:
        if set(r) != qids:
            raise ValueError("the rankings are for different queries")
    out: Run = {}
    for q in qids:
        acc: Dict[str, float] = {}
        for r in runs:
            for rank, (doc, _s) in enumerate(qmetrics.ranked(r[q], depth), start=1):
                acc[doc] = acc.get(doc, 0.0) + 1.0 / (k + rank)
        out[q] = acc
    return out


# ------------------------------------------------------------------------------------------------
# run folders
# ------------------------------------------------------------------------------------------------
def _model_out(run_dir: Path) -> Path:
    outs = [d for d in (run_dir / "results").glob("*") if d.is_dir() and (d / "run_top100.trec.gz").exists()]
    if len(outs) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one results/<model>/ folder with run_top100.trec.gz, found {len(outs)}")
    return outs[0]


def _load_ranking(out: Path) -> Run:
    preds = sorted(out.glob("*_predictions.json"))
    return qmetrics.load_run(preds[0] if len(preds) == 1 else out / "run_top100.trec.gz")


def discover(tests_root: Path) -> Dict[tuple, Dict[str, Path]]:
    """{(chunker, size, overlap): {model name: run folder}} from the run_info.json files."""
    cells: Dict[tuple, Dict[str, Path]] = {}
    if not Path(tests_root).is_dir():
        return cells
    for d in sorted(Path(tests_root).iterdir()):
        info = d / "run_info.json"
        if d.is_dir() and info.is_file() and (d / "results").is_dir():
            i = json.loads(info.read_text(encoding="utf-8"))
            if i.get("hybrid") or i.get("rerank"):          # derived runs are never an input of a fusion
                continue
            cells.setdefault((i["chunker"], int(i["size"]), int(i["overlap"])), {})[i["model"]] = d
    return cells


def hybrid_name(k: int) -> str:
    return f"hybrid_rrf{k}"


def _short(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def _sum_times(*parts: Optional[float]) -> Optional[float]:
    return None if any(p is None for p in parts) else float(sum(parts))


def run_hybrid(dense_dir: Path, sparse_dir: Path, tests_root: Path, *, k: int = DEFAULT_K, depth: int = 100,
               force: bool = False, run_name: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Fuse the ranking of ``dense_dir`` with that of ``sparse_dir`` (both run folders) into a new run folder."""
    di, si = (json.loads((d / "run_info.json").read_text(encoding="utf-8")) for d in (dense_dir, sparse_dir))
    if di["prepared_key"] != si["prepared_key"]:
        raise ValueError(f"the two runs are not on the same prepared configuration ({di['prepared_key'][:8]} vs {si['prepared_key'][:8]})")
    name = f"{hybrid_name(k)}_{_short(di['model'])}_{_short(si['model'])}"
    run_dir = Path(tests_root) / f"{name}_{di['chunker']}_c{di['size']}_o{di['overlap']}"
    if (run_dir / "results" / "summary.json").exists() and not force:
        return {"run_dir": str(run_dir), "status": "skipped"}
    seeds = set_seeds(seed) if seed is not None else None
    t0 = time.perf_counter()
    d_out, s_out = _model_out(dense_dir), _model_out(sparse_dir)
    runs = [_load_ranking(d_out), _load_ranking(s_out)]
    fused = fuse(runs, k=k, depth=depth)
    prepared = Path(di["prepared_dir"])

    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("chunks", "mteb_task"):
        link = run_dir / sub
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to((prepared / sub).resolve(), target_is_directory=True)
    out = run_dir / "results" / (run_name or name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ChemQuest_default_predictions.json").write_text(json.dumps(fused), encoding="utf-8")
    qrels_name = _primary_qrels(d_out, prepared)
    scores = run_outputs.mteb_style_scores(fused, qmetrics.load_qrels(prepared / "mteb_task" / qrels_name),
                                           qmetrics.load_query_ids(prepared / "mteb_task" / "queries.jsonl"))
    fusion_s = time.perf_counter() - t0                       # read + fuse + metrics + dump (what a dense evaluation_time covers)
    dense_s, sparse_s = run_outputs.read_eval_time(d_out), run_outputs.read_eval_time(s_out)
    total_s = _sum_times(dense_s, sparse_s, fusion_s)
    if total_s is None:
        print(f"[WARN] {run_dir.name}: an input run has no evaluation_time (dense {dense_s}, sparse {sparse_s}); the hybrid time is null")
    run_outputs.write_chemquest_json(out, scores, total_s, method="rrf")
    summary = run_outputs.write_per_query_outputs(out, prepared / "mteb_task", qrels_name, depth=depth, run_name=name)
    elapsed = time.perf_counter() - t0
    inputs = [{"role": role, "model": i["model"], "run_dir": str(d), "ranking": str(o / "run_top100.trec.gz"),
               "ranking_sha256": sha256_file(o / "run_top100.trec.gz")}
              for role, i, d, o in (("dense", di, dense_dir, d_out), ("sparse", si, sparse_dir, s_out))]
    manifest = {"method": "reciprocal_rank_fusion", "k": k, "depth": depth, "inputs": inputs,
                "tie_break": "score descending, chunk id descending", "no_embedding": True,
                "prepared_key": di["prepared_key"], "git": git_info(_REPO_ROOT), "packages": package_versions(["numpy", "pytrec_eval"]),
                "timing_s": {"dense_evaluation_time_s": dense_s, "bm25_evaluation_time_s": sparse_s, "fusion_s": fusion_s,
                            "evaluation_time_s": total_s, "mteb_evaluation_time_s": total_s, "fusion_and_metrics_s": elapsed},
                "evaluation_time": "dense evaluation_time + bm25 evaluation_time + fusion_s", "use_prompts": None,
                "data_sha256": _data_hashes(d_out), **derived_run_environment("cpu", seeds)}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    info = dict(di, model=f"{name}", hybrid={"method": "rrf", "k": k, "depth": depth, "dense": di["model"], "sparse": si["model"]})
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    main_score = summary["aggregate"].get("ndcg_at_10") if summary else None
    (run_dir / "results" / "summary.json").write_text(json.dumps([{"model": name, "revision": "rrf", "task": "ChemQuest",
                                                                  "main_score": main_score}], indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "status": "done", "ndcg_at_10": main_score, "evaluation_time": total_s}


def _data_hashes(model_out: Path) -> Dict[str, str]:
    m = model_out / "run_manifest.json"
    return json.loads(m.read_text(encoding="utf-8")).get("data_sha256", {}) if m.exists() else {}


def _primary_qrels(model_out: Path, prepared: Path) -> str:
    ms = json.loads((model_out / "metrics_summary.json").read_text(encoding="utf-8"))
    return ms["qrels_file"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests-root", required=True)
    ap.add_argument("--dense", nargs="+", required=True, help="model names of the dense runs (as in run_info.json)")
    ap.add_argument("--sparse", default="bm25", help="model name of the sparse run")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cells = discover(Path(args.tests_root))
    n = 0
    for cell, models in sorted(cells.items()):
        sp = next((d for m, d in models.items() if m == args.sparse), None)
        if sp is None:
            continue
        for dense in args.dense:
            if dense in models:
                r = run_hybrid(models[dense], sp, Path(args.tests_root), k=args.k, depth=args.depth, force=args.force)
                print(f"{r['status']:8s} {cell} {dense} + {args.sparse} -> {r['run_dir']}")
                n += r["status"] == "done"
    print(f"{n} hybrid runs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

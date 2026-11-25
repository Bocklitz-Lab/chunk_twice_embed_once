"""Cross-encoder reranking of the saved top-N of a dense run. Post-hoc, like ``tools/hybrid.py``: it reads a finished dense run
folder (its ranking, the prepared ``mteb_task/corpus.jsonl`` and ``queries.jsonl``), scores the (query, chunk) pairs of the top N
with a cross-encoder and writes a new run folder with the same outputs as a normal run. No new chunking, labels or embeddings.

  head  = the N best chunks of the dense ranking, re-ordered by the cross-encoder score (raw logit, higher = better)
  tail  = every chunk below rank N keeps its original rank and order (it always stays below the head)

Written scores: head = 1 + (logit - lowest head logit of the query) (> 0, the logit order), tail = -(original rank) (< 0, the
original order); ties are broken like every other run (score descending, chunk id descending). The raw logits are saved in
``rerank_logits.json.gz``. The ranking is cut at ``depth`` (100) like all runs, so N <= depth.

Run folder written  <tests root>/rerank_<model>_top<N>_<dense dir>_<chunker>_c<L>_o<O>/  with run_info.json (``rerank``:
model, revision, top_n, dense, depth), links to chunks/ and mteb_task/, results/summary.json and results/<name>/{ChemQuest.json,
ChemQuest_default_predictions.json, run_top100.trec.gz, per_query_metrics*.csv (span and lexical labels), span_coverage.csv,
metrics_summary.json, rerank_logits.json.gz, run_manifest.json}. Rerank runs are never inputs of ``tools/hybrid.py`` or of the
dense comparisons of the report (``run_info.json`` carries the key ``rerank``).

evaluation_time (ChemQuest.json, run_manifest timing_s.mteb_evaluation_time_s) = dense evaluation_time + rerank_s, where
rerank_s = reading corpus / queries + scoring the pairs + reordering + metrics + dump of the new ranking (the same stages that a
dense evaluation_time covers; the one-time model load is NOT counted and is stored as model_load_s). ``score_pairs_s`` (the
cross-encoder forward passes) is stored separately.

  python -m tools.rerank --tests-root artifacts/pilot_v2/tests --dense intfloat/e5-large-v2 --top-n 50
  python -m tools.rerank --tests-root ... --dense intfloat/e5-large-v2 --top-n 20 50 100 --chunker fixed_token --size 256 --overlap 0
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import run_outputs  # noqa: E402
from pipeline_lib.run_manifest import derived_run_environment, git_info, package_versions, set_seeds, sha256_file  # noqa: E402
from tools import hybrid  # noqa: E402
from tools import metrics as qmetrics  # noqa: E402

Run = Dict[str, Dict[str, float]]
Scorer = Callable[[List[Tuple[str, str]]], Sequence[float]]
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


# ------------------------------------------------------------------------------------------------
# the cross-encoder
# ------------------------------------------------------------------------------------------------
class CrossEncoderScorer:
    """``sentence_transformers.CrossEncoder`` as a ``scorer(pairs) -> logits``; loaded on first use (once per process)."""

    def __init__(self, model: str = DEFAULT_MODEL, revision: Optional[str] = None, *, batch_size: int = 32, max_length: int = 1024,
                 fp16: bool = True, device: Optional[str] = None):
        if not revision:
            raise ValueError("the reranker revision must be pinned (params.rerank.revision)")
        self.model_name, self.revision, self.batch_size, self.max_length, self.fp16, self.device = model, revision, batch_size, max_length, fp16, device
        self._ce = None
        self.model_load_s: Optional[float] = None
        self._len_cache: Dict[str, int] = {}

    def load(self):
        if self._ce is None:
            import torch
            from sentence_transformers import CrossEncoder

            t0 = time.perf_counter()
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            args = {"torch_dtype": torch.float16} if (self.fp16 and dev.startswith("cuda")) else {}
            self._ce = CrossEncoder(self.model_name, revision=self.revision, max_length=self.max_length, device=dev, automodel_args=args)
            self.device = dev
            self.model_load_s = time.perf_counter() - t0
        return self._ce

    def __call__(self, pairs: List[Tuple[str, str]]) -> List[float]:
        import torch

        ce = self.load()
        # raw logits (Identity): a sigmoid would saturate to ties for confident pairs
        out = ce.predict([list(p) for p in pairs], batch_size=self.batch_size, show_progress_bar=False, activation_fct=torch.nn.Identity())
        return [float(x) for x in out]

    def token_len(self, text: str) -> int:
        """Tokens of ``text`` without special tokens (cached); used to count pairs longer than ``max_length``."""
        if text not in self._len_cache:
            self._len_cache[text] = len(self.load().tokenizer(text, add_special_tokens=False)["input_ids"])
        return self._len_cache[text]

    def describe(self) -> Dict[str, Any]:
        return {"model": self.model_name, "revision": self.revision, "batch_size": self.batch_size, "max_length": self.max_length,
                "fp16": self.fp16, "device": self.device, "activation": "identity (raw logit)"}


# ------------------------------------------------------------------------------------------------
# reranking one ranking
# ------------------------------------------------------------------------------------------------
def rerank_run(run: Run, query_text: Dict[str, str], doc_text: Dict[str, str], scorer: Scorer, top_n: int,
               depth: int = 100) -> Tuple[Run, Dict[str, Dict[str, float]], int]:
    """(new ranking, {qid: {chunk: logit}} of the head, number of scored pairs). Head = the first ``top_n`` of the ranking cut at
    ``depth``, re-ordered by the scorer; tail keeps its order below the head (see the module docstring for the written scores)."""
    if top_n < 1 or top_n > depth:
        raise ValueError(f"top_n must be in [1, depth={depth}], got {top_n}")
    heads: Dict[str, List[str]] = {}
    tails: Dict[str, List[Tuple[int, str]]] = {}
    pairs: List[Tuple[str, str]] = []
    for q in sorted(run):
        ranked = [d for d, _ in qmetrics.ranked(run[q], depth)]
        heads[q] = ranked[:top_n]
        tails[q] = [(r, d) for r, d in enumerate(ranked[top_n:], start=top_n + 1)]
        pairs += [(query_text[q], doc_text[d]) for d in heads[q]]
    logits = list(scorer(pairs))
    if len(logits) != len(pairs):
        raise RuntimeError(f"the scorer returned {len(logits)} scores for {len(pairs)} pairs")
    out: Run = {}
    raw: Dict[str, Dict[str, float]] = {}
    i = 0
    for q in sorted(run):
        sc = {d: float(logits[i + j]) for j, d in enumerate(heads[q])}
        i += len(heads[q])
        raw[q] = sc
        low = min(sc.values()) if sc else 0.0
        new = {d: 1.0 + (s - low) for d, s in sc.items()}
        new.update({d: -float(r) for r, d in tails[q]})
        out[q] = new
    return out, raw, len(pairs)


# ------------------------------------------------------------------------------------------------
# run folders
# ------------------------------------------------------------------------------------------------
def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_texts(task_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """({chunk id: text}, {query id: text}); the chunk text is what MTEB embeds: (title + " " + text).strip()."""
    docs = {str(r["id"]): ((r["title"] + " " + r["text"]) if r.get("title") else r["text"]).strip() for r in _read_jsonl(task_dir / "corpus.jsonl")}
    queries = {str(r["id"]): str(r.get("text") or "") for r in _read_jsonl(task_dir / "queries.jsonl")}
    return docs, queries


def rerank_name(model: str, top_n: int, dense_model: str) -> str:
    return f"rerank_{hybrid._short(model.split('/')[-1])}_top{top_n}_{hybrid._short(dense_model)}"


def run_rerank(dense_dir: Path, tests_root: Path, scorer: Scorer, *, model: str = DEFAULT_MODEL, revision: str = "", top_n: int = 50,
               depth: int = 100, force: bool = False, scorer_info: Optional[Dict[str, Any]] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Rerank the top ``top_n`` of ``dense_dir`` (a dense run folder) into a new run folder under ``tests_root``."""
    di = json.loads((dense_dir / "run_info.json").read_text(encoding="utf-8"))
    if di.get("hybrid") or di.get("rerank") or str(di["model"]).startswith("bm25"):
        raise ValueError(f"{dense_dir.name}: only a dense run can be reranked (got {di['model']})")
    name = rerank_name(model, top_n, di["model"])
    run_dir = Path(tests_root) / f"{name}_{di['chunker']}_c{di['size']}_o{di['overlap']}"
    if (run_dir / "results" / "summary.json").exists() and not force:
        return {"run_dir": str(run_dir), "status": "skipped"}
    d_out = hybrid._model_out(dense_dir)
    prepared = Path(di["prepared_dir"])
    task_dir = prepared / "mteb_task"

    seeds = set_seeds(seed) if seed is not None else None
    if hasattr(scorer, "load"):
        scorer.load()                                          # the one-time model load is outside the clock (stored as model_load_s)
    scorer_info = scorer_info or (scorer.describe() if hasattr(scorer, "describe") else None)       # after the load: the device is known
    t0 = time.perf_counter()
    run = hybrid._load_ranking(d_out)
    docs, queries = load_texts(task_dir)
    t_score = time.perf_counter()
    new_run, raw, n_pairs = rerank_run(run, queries, docs, scorer, top_n, depth)
    score_pairs_s = time.perf_counter() - t_score

    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("chunks", "mteb_task"):
        link = run_dir / sub
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to((prepared / sub).resolve(), target_is_directory=True)
    out = run_dir / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "ChemQuest_default_predictions.json").write_text(json.dumps(new_run), encoding="utf-8")
    qrels_name = hybrid._primary_qrels(d_out, prepared)
    scores = run_outputs.mteb_style_scores(new_run, qmetrics.load_qrels(task_dir / qrels_name), qmetrics.load_query_ids(task_dir / "queries.jsonl"))
    rerank_s = time.perf_counter() - t0                        # read + score + reorder + metrics + dump
    truncated = _count_truncated(scorer, queries, docs, run, top_n, depth)        # diagnostics, not part of rerank_s
    with open(out / "rerank_logits.json.gz", "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
        gz.write(json.dumps(raw).encode("utf-8"))
    dense_s = run_outputs.read_eval_time(d_out)
    total_s = None if dense_s is None else float(dense_s + rerank_s)
    if total_s is None:
        print(f"[WARN] {run_dir.name}: the dense run has no evaluation_time; the rerank time is null")
    run_outputs.write_chemquest_json(out, scores, total_s, method="rerank")
    summary = run_outputs.write_per_query_outputs(out, task_dir, qrels_name, depth=depth, run_name=name)
    elapsed = time.perf_counter() - t0
    manifest = {"method": "cross_encoder_rerank", "top_n": top_n, "depth": depth, "reranker": scorer_info or {"model": model, "revision": revision},
                "inputs": [{"role": "dense", "model": di["model"], "run_dir": str(dense_dir), "ranking": str(d_out / "run_top100.trec.gz"),
                            "ranking_sha256": sha256_file(d_out / "run_top100.trec.gz")}],
                "head_scores": "1 + (logit - lowest head logit of the query); logits in rerank_logits.json.gz",
                "tail_scores": "-(original rank): chunks below rank top_n keep their order below the head",
                "tie_break": "score descending, chunk id descending", "n_pairs": n_pairs, "pairs_truncated": truncated,
                "prepared_key": di["prepared_key"], "git": git_info(_REPO_ROOT), "packages": package_versions(["numpy", "pytrec_eval", "sentence-transformers", "torch", "transformers"]),
                "timing_s": {"dense_evaluation_time_s": dense_s, "rerank_s": rerank_s, "score_pairs_s": score_pairs_s,
                             "pairs_per_s": (n_pairs / score_pairs_s) if score_pairs_s > 0 else None, "model_load_s": getattr(scorer, "model_load_s", None),
                             "evaluation_time_s": total_s, "mteb_evaluation_time_s": total_s, "rerank_and_outputs_s": elapsed},
                "evaluation_time": "dense evaluation_time + rerank_s (read + score + reorder + metrics + dump; model load not counted)",
                "use_prompts": None, "data_sha256": hybrid._data_hashes(d_out),
                **derived_run_environment(str((scorer_info or {}).get("device") or "unknown"), seeds)}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    info = dict(di, model=name, rerank={"model": model, "revision": revision, "top_n": top_n, "dense": di["model"], "depth": depth})
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    main_score = summary["aggregate"].get("ndcg_at_10") if summary else None
    (run_dir / "results" / "summary.json").write_text(json.dumps([{"model": name, "revision": revision, "task": "ChemQuest",
                                                                  "main_score": main_score}], indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "status": "done", "ndcg_at_10": main_score, "evaluation_time": total_s, "rerank_s": rerank_s}


def _count_truncated(scorer: Any, queries: Dict[str, str], docs: Dict[str, str], run: Run, top_n: int, depth: int) -> Optional[Dict[str, Any]]:
    """Pairs whose (query + chunk) token count (+ 4 special tokens) exceeds the scorer's ``max_length``; None for a scorer without one."""
    tl, max_len = getattr(scorer, "token_len", None), getattr(scorer, "max_length", None)
    if tl is None or not max_len:
        return None
    n = over = 0
    for q, r in run.items():
        for d, _ in qmetrics.ranked(r, depth)[:top_n]:
            n += 1
            over += (tl(queries[q]) + tl(docs[d]) + 4) > max_len
    return {"n_pairs": n, "n_truncated": over, "pct_truncated": 100.0 * over / n if n else 0.0, "max_length": max_len}


# ------------------------------------------------------------------------------------------------
# the configured step (params.rerank): used by stages/rerun.py and by the command line
# ------------------------------------------------------------------------------------------------
def _model_matches(spec: str, model: str) -> bool:
    return model == spec or model.endswith("/" + spec)


def _field_matches(spec: Any, value: Any) -> bool:
    if spec is None or spec == "all":
        return True
    return value in spec if isinstance(spec, (list, tuple, set)) else value == spec


def plan_jobs(cells: Dict[tuple, Dict[str, Path]], cfg: Dict[str, Any], *, models: Optional[Iterable[str]] = None,
              wanted_cells: Optional[Iterable[tuple]] = None) -> Tuple[List[Tuple[Path, int, dict]], List[str]]:
    """[(dense run folder, top_n, cell info)] and a list of warnings. Primary: every dense run (restricted to ``cfg['models']`` if
    given, and to ``models`` / ``wanted_cells`` of the invocation) at ``cfg['top_n']``. Ablation: each entry of ``cfg['ablation']``
    adds ``top_n`` (a number or a list) for the dense runs it matches (model, chunker, size, overlap; a missing field matches all)."""
    primary = int(cfg.get("top_n", 50))
    allowed = set(models) if models is not None else None
    wanted = set(wanted_cells) if wanted_cells is not None else None
    cfg_models = cfg.get("models") or None
    jobs: Dict[Tuple[str, int], Tuple[Path, int, dict]] = {}
    warns: List[str] = []

    def eligible(cell, model):
        return (not str(model).startswith("bm25") and (allowed is None or model in allowed) and (wanted is None or cell in wanted))

    for cell, runs in sorted(cells.items()):
        for model, d in sorted(runs.items()):
            if eligible(cell, model) and (cfg_models is None or any(_model_matches(m, model) for m in cfg_models)):
                jobs[(str(d), primary)] = (d, primary, {"chunker": cell[0], "size": cell[1], "overlap": cell[2], "model": model})
    for k, entry in enumerate(cfg.get("ablation") or []):
        tops = entry.get("top_n", [])
        tops = [int(t) for t in (tops if isinstance(tops, (list, tuple)) else [tops])]
        hit = 0
        for cell, runs in sorted(cells.items()):
            if not (_field_matches(entry.get("chunker"), cell[0]) and _field_matches(entry.get("size"), cell[1]) and _field_matches(entry.get("overlap"), cell[2])):
                continue
            for model, d in sorted(runs.items()):
                if eligible(cell, model) and (entry.get("model") is None or _model_matches(str(entry["model"]), model)):
                    hit += 1
                    for t in tops:
                        jobs[(str(d), t)] = (d, t, {"chunker": cell[0], "size": cell[1], "overlap": cell[2], "model": model})
        if not hit:
            warns.append(f"rerank.ablation[{k}] {entry} matches no dense run (is its size / chunker part of this run?)")
    return sorted(jobs.values(), key=lambda j: (j[2]["chunker"], j[2]["size"], j[2]["overlap"], j[2]["model"], j[1])), warns


def check_ablation(cfg: Dict[str, Any], chunkers: Iterable[str], cells: Iterable[Tuple[int, int]], models: Optional[Iterable[str]] = None) -> List[str]:
    """Problems of ``cfg['ablation']`` against the grid of an invocation (``chunkers`` x ``cells`` = (size, overlap) pairs) and its
    dense ``models``: an entry whose chunker / size / overlap is in no configuration of the grid, or whose model is not evaluated.
    Empty list = every entry can run. stages/rerun.py aborts on a problem before any run starts."""
    wanted = [(ch, s, o) for ch in chunkers for s, o in cells]
    names = list(models) if models is not None else None
    sizes = sorted({c[1] for c in wanted})
    problems = []
    for k, e in enumerate(cfg.get("ablation") or []):
        if not any(_field_matches(e.get("chunker"), c[0]) and _field_matches(e.get("size"), c[1]) and _field_matches(e.get("overlap"), c[2]) for c in wanted):
            problems.append(f"params.rerank.ablation[{k}] {e} matches no configuration of this grid (chunkers {sorted({c[0] for c in wanted})}, sizes {sizes}, "
                            f"overlaps {sorted({c[2] for c in wanted})})")
        if names is not None and e.get("model") is not None and not any(_model_matches(str(e["model"]), m) for m in names):
            problems.append(f"params.rerank.ablation[{k}] model {e['model']!r} is not among the evaluated models {names}")
    for m in cfg.get("models") or []:
        if names is not None and not any(_model_matches(str(m), n) for n in names):
            problems.append(f"params.rerank.models entry {m!r} is not among the evaluated models {names} (is it on the Phase 2 list of stage 2?)")
    return problems


def run_configured(tests_root: Path, cfg: Dict[str, Any], *, models: Optional[Iterable[str]] = None, wanted_cells: Optional[Iterable[tuple]] = None,
                   depth: int = 100, force: bool = False, scorer: Optional[Scorer] = None, dry_run: bool = False,
                   seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """The rerank step: plan the jobs of ``params.rerank`` over the dense runs found in ``tests_root`` and run them (finished runs are
    skipped, so the step is resumable). One cross-encoder is loaded for all jobs."""
    tests_root = Path(tests_root)
    model, revision = cfg.get("model", DEFAULT_MODEL), cfg.get("revision")
    jobs, warns = plan_jobs(hybrid.discover(tests_root), cfg, models=models, wanted_cells=wanted_cells)
    for w in warns:
        print(f"[WARN] {w}")
    if scorer is None and not dry_run:
        scorer = CrossEncoderScorer(model, revision, batch_size=int(cfg.get("batch_size", 32)), max_length=int(cfg.get("max_length", 1024)),
                                    fp16=bool(cfg.get("fp16", True)))
    out = []
    for d, top_n, cell in jobs:
        if dry_run:
            print(f"[dry-run] rerank {cell['model']} {cell['chunker']} c{cell['size']} o{cell['overlap']} top{top_n}")
            out.append({"run_dir": str(d), "status": "dry-run", "top_n": top_n})
            continue
        r = run_rerank(d, tests_root, scorer, model=model, revision=revision, top_n=top_n, depth=depth, force=force, seed=seed)
        print(f"rerank {cell['model']} {cell['chunker']} c{cell['size']} o{cell['overlap']} top{top_n}: {r['status']}"
              + (f" ({r['rerank_s']:.1f}s)" if "rerank_s" in r else ""))
        out.append({**r, "top_n": top_n})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests-root", required=True)
    ap.add_argument("--dense", nargs="*", help="model names of the dense runs (default: all dense runs found)")
    ap.add_argument("--top-n", nargs="+", type=int, default=[50])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", help="pinned commit of the reranker (default: params.rerank.revision of the main config)")
    ap.add_argument("--main-config", default="configs/main_config.yaml")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--chunker", nargs="*")
    ap.add_argument("--size", nargs="*", type=int)
    ap.add_argument("--overlap", nargs="*", type=int)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    import yaml

    base = (yaml.safe_load(Path(args.main_config).read_text(encoding="utf-8"))["params"].get("rerank") or {})
    cfg = {"model": args.model, "revision": args.revision or base.get("revision"), "top_n": args.top_n[0],
           "batch_size": args.batch_size or base.get("batch_size", 32), "max_length": args.max_length or base.get("max_length", 1024),
           "fp16": not args.no_fp16 and base.get("fp16", True), "models": args.dense or None,
           "ablation": [{"top_n": args.top_n[1:]}] if len(args.top_n) > 1 else []}
    cells = hybrid.discover(Path(args.tests_root))
    wanted = [c for c in cells if _field_matches(args.chunker, c[0]) and _field_matches(args.size, c[1]) and _field_matches(args.overlap, c[2])]
    res = run_configured(Path(args.tests_root), cfg, wanted_cells=wanted, depth=args.depth, force=args.force)
    print(f"{sum(r['status'] == 'done' for r in res)} rerank runs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

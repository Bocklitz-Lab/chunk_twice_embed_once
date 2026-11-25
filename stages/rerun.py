"""Prepare / eval split of Phase 2 (reviewer revisions).

  prepare   chunk + label every configuration ONCE (stages/prepare.py): 28 (size, overlap) pairs x 5 strategies = 140
  eval      for every model (and BM25) and configuration: stage 6 on the prepared folder
  all       prepare, then eval (+ the hybrid runs: reciprocal rank fusion of each dense run with BM25, params.hybrid;
            + the cross-encoder rerank runs of the dense runs, params.rerank (tools/rerank.py);
            + the report: statistics, diagnostics, tables in results/v2-revision/, tools/make_report.py, params.report)
  hybrid    only the hybrid step on the finished dense and BM25 runs (resumable: finished runs are skipped)
  rerank    only the rerank step (primary depth for params.rerank.models + the ablation entries; resumable)
  --pilot   the pilot of params.pilot in main_config.yaml (2 models + BM25, fixed-token and hierarchical-section,
            sizes 128 and 512, overlap 0) in its own artifacts root; never touches artifacts/tests

A run folder  <artifacts>/tests/<model_dir>_<chunker>_c<L>_o<O>/  keeps the layout of the published runs
(config/, results/, durations) and links  chunks/  and  mteb_task/  to the prepared folder shared by all models.
Finished runs (results/summary.json) are skipped; old results are never deleted.

  python -m stages.rerun prepare --pilot
  python -m stages.rerun eval --pilot
  python -m stages.rerun all                                        # the full rerun (models = shortlist of stage 2, stage2.output_per_dataset_csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from pipeline_lib import grid, split_config  # noqa: E402
from stages import prepare as prep  # noqa: E402
from pipeline_lib import efficiency as eff  # noqa: E402

BM25 = "bm25"
FINAL_RESULT = "results/summary.json"
DURATION_HEADER = "started_at,ended_at,duration_s,status,stage,model,revision,chunker,size,overlap,outdir"


def model_dir(name: str) -> str:
    d = name.replace("/", "_")
    return d.replace("-", "_")


def pinned_revisions(main_config: str) -> Dict[str, str]:
    """{model name: commit hash} of the 42 Phase 1 models (stage1.models of the main config)."""
    doc = yaml.safe_load(Path(main_config).read_text(encoding="utf-8"))
    return {m["name"]: m["revision"] for m in doc["stage1"]["models"]}


def stage1_batch_sizes(main_config: str) -> Dict[str, int]:
    """{model name: batch size} of the Phase 1 models that set ``batch_size`` in ``stage1.models`` (the others are not in the table)."""
    doc = yaml.safe_load(Path(main_config).read_text(encoding="utf-8"))
    return {m["name"]: int(m["batch_size"]) for m in doc["stage1"]["models"] if m.get("batch_size") is not None}


def batch_size_for(name: str, table: Dict[str, int], default: int, forced: Optional[int] = None) -> Tuple[int, str]:
    """(batch size, source) of a Phase 2 run: ``forced`` (``--param model.batch_size=...``) for every model, else the model's own Phase 1
    batch size, else ``params.model.batch_size``. BM25 has no batch size."""
    if forced is not None:
        return forced, "--param model.batch_size"
    if name in table:
        return table[name], "stage1.models"
    return default, "params.model.batch_size"


def forced_batch_size(params: List[str]) -> Optional[int]:
    for p in params:
        if p.startswith("model.batch_size="):
            return int(p.split("=", 1)[1])
    return None


def batch_size_changes(main_config: str, models: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Models whose Phase 2 batch size differs from ``params.model.batch_size`` (what every model used before the per-model setting)."""
    doc = yaml.safe_load(Path(main_config).read_text(encoding="utf-8"))
    default, table = int(doc["params"]["model"]["batch_size"]), stage1_batch_sizes(main_config)
    names = models if models is not None else [m["name"] for m in doc["stage1"]["models"]]
    return [{"model": n, "before": default, "now": batch_size_for(n, table, default)[0]} for n in names
            if batch_size_for(n, table, default)[0] != default]


def resolve_revision(name: str, rev: Optional[str], table: Dict[str, str]) -> str:
    """A moving or missing revision (main, empty) is replaced by the commit pinned in the config; unknown models with a
    moving revision are refused. An explicit commit hash is kept as it is."""
    if name.startswith(BM25):
        return rev or "rank_bm25"
    if rev and rev not in ("main", "master"):
        return rev
    if name not in table or table[name] in ("main", "master"):
        raise ValueError(f"{name}: no pinned revision (give name@<commit> or pin it in stage1.models of the main config)")
    return table[name]


def parse_model(spec: str) -> Tuple[str, str]:
    name, _, rev = spec.partition("@")
    return name, rev or "main"


RESTORE_HINT = ("The Phase 2 model list is the shortlist written by stage 2 (make stage1 stage2: Phase 1 with prompts). To use the PUBLISHED "
                "shortlist instead, restore the old Phase 1 results yourself (this repository does not copy files), e.g.\n"
                "    cp -a temp_files/artifacts_old/embedding_models_screening artifacts/\n"
                "and pass it: make eval MODELS_CSV=artifacts/embedding_models_screening/passed_per_dataset.csv   (or --models-csv ...).")


def default_models_csv(main_config: str, params: Optional[List[str]] = None) -> str:
    """The shortlist file of the main config: stage 2 ``output_per_dataset_csv`` (derived from params.paths.phase1_results_subdir)."""
    return str(split_config.resolve_section(main_config, "stage2", params or [])["output_per_dataset_csv"])


def check_models_csv(path: str) -> None:
    """Stop with a clear message if the shortlist file is missing or empty (before any chunking or evaluation starts)."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise SystemExit(f"ERROR: the model list MODELS_CSV does not exist or is empty: {path}\n{RESTORE_HINT}")


def models_from_csv(path: str) -> List[Tuple[str, str]]:
    seen, out = set(), []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["model_name"], r["revision"])
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def ensure_data(main_config: str, artifacts_root: str, params: List[str]) -> Path:
    """Download the pinned ChemQuests release (stage 3) if full_text.jsonl / qa.jsonl are not there yet."""
    with tempfile.TemporaryDirectory() as tmp:
        argv = ["--input", main_config, "--output-dir", tmp, "--sections", "stage3", "--param", f"paths.artifacts_root={artifacts_root}"]
        for p in params:
            argv += ["--param", p]
        if split_config.main(argv) != 0:
            raise RuntimeError("split_config failed for stage3")
        cfg = yaml.safe_load((Path(tmp) / "stage3.yaml").read_text(encoding="utf-8"))
        out = Path(cfg["output_dir"])
        if all((out / n).exists() for n in ("full_text.jsonl", "qa.jsonl")):
            if not (out / "question_metadata.jsonl").exists():
                print(f"note: {out}/question_metadata.jsonl is missing (the subfield / question-type analysis needs it); run stage 3 once to download it: "
                      "python -m stages.download_chemquests --config configs/stages/stage3.yaml")
            return out / "qa.jsonl"
        print(f"downloading the pinned ChemQuests release -> {out}")
        subprocess.run([sys.executable, "-m", "stages.download_chemquests", "--config", str(Path(tmp) / "stage3.yaml")],
                       cwd=_REPO_ROOT, check=True)
        return out / "qa.jsonl"


def _link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        if not link.is_symlink():
            raise RuntimeError(f"{link} exists and is not a link (an old run?); use another --artifacts-root")
        link.unlink()
    link.symlink_to(target, target_is_directory=True)


def efficiency_args(args: argparse.Namespace, root: str, eff_cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Flags for stages.run_eval from the rerun flags (--efficiency, --latency) and ``params.efficiency``; [] when both flags are off.

    Number of timed queries, warm-up, search device and the latency subset come from the config block ``params.efficiency``
    (``latency_queries``, ``latency_warmup``, ``latency_device``, ``latency_cells``, ``latency_models``). A flag given on the command line
    wins: any explicit ``--latency-chunkers/-sizes/-overlaps/-models/-cells`` replaces the configured subset."""
    if not (args.efficiency or args.latency):
        return []
    eff_cfg = eff_cfg or {}
    out = ["--efficiency", "--efficiency-out", str(Path(root) / "tests")]
    if args.latency:
        pick = lambda cli, key, default: cli if cli is not None else eff_cfg.get(key, default)      # noqa: E731
        out += ["--latency", "--latency-queries", str(pick(args.latency_queries, "latency_queries", 200)),
                "--latency-warmup", str(pick(args.latency_warmup, "latency_warmup", 10)),
                "--latency-device", str(pick(args.latency_device, "latency_device", "cpu"))]
        explicit = any([args.latency_chunkers, args.latency_sizes, args.latency_overlaps, args.latency_models, args.latency_cells])
        subset = {"--latency-chunkers": args.latency_chunkers, "--latency-sizes": args.latency_sizes,
                  "--latency-overlaps": args.latency_overlaps, "--latency-models": args.latency_models,
                  "--latency-cells": args.latency_cells} if explicit else \
                 {"--latency-cells": [f"{c['chunker']}:{c['size']}:{c['overlap']}" for c in eff_cfg.get("latency_cells") or []],
                  "--latency-models": eff_cfg.get("latency_models") or []}
        for flag, vals in subset.items():
            if vals:
                out += [flag, *map(str, vals)]
    return out


def run_eval_one(model: str, revision: str, chunker: str, size: int, overlap: int, prepared: Path, *, main_config: str,
                 artifacts_root: str, params: List[str], force: bool, dry_run: bool,
                 eff_args: Optional[List[str]] = None, batch_sizes: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    d = model_dir(model)
    run_dir = Path(artifacts_root) / "tests" / f"{d}_{chunker}_c{size}_o{overlap}"
    if not force and (run_dir / FINAL_RESULT).is_file() and (run_dir / FINAL_RESULT).stat().st_size > 0:
        print(f"skip (finished): {run_dir}")
        return {"run_dir": str(run_dir), "status": "skipped"}
    if (run_dir / "results").exists():
        print(f"earlier run incomplete or forced; results are written to a fresh folder: {run_dir}")
        stamp = time.strftime("%Y%m%dT%H%M%S")
        (run_dir / "results").rename(run_dir / f"results_old_{stamp}")      # moved, never deleted
    if dry_run:
        print(f"[dry-run] {model} {chunker} c{size} o{overlap} -> {run_dir}")
        return {"run_dir": str(run_dir), "status": "dry-run"}
    run_dir.mkdir(parents=True, exist_ok=True)
    _link(run_dir / "chunks", prepared / "chunks")
    _link(run_dir / "mteb_task", prepared / "mteb_task")
    cfg_dir = run_dir / "config"
    argv = ["--input", main_config, "--output-dir", str(cfg_dir), "--sections", "stage6",
            "--param", f"model.name={model}", "--param", f"model.revision={revision}", "--param", f"model.dir={d}",
            "--param", f"chunk.chunker={chunker}", "--param", f"chunk.size={size}", "--param", f"chunk.overlap={overlap}",
            "--param", f"paths.artifacts_root={artifacts_root}"]
    default_bs = int(yaml.safe_load(Path(main_config).read_text(encoding="utf-8"))["params"]["model"]["batch_size"])
    bs, bs_source = batch_size_for(model, batch_sizes or {}, default_bs, forced_batch_size(params))
    if not model.startswith(BM25):
        argv += ["--param", f"model.batch_size={bs}"]                  # the model's own (Phase 1) batch size; later --param entries still win
    for p in params:
        argv += ["--param", p]
    if split_config.main(argv) != 0:
        raise RuntimeError("split_config failed for stage6")
    seed = yaml.safe_load((cfg_dir / "stage6.yaml").read_text(encoding="utf-8")).get("seed")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    (run_dir / "run_info.json").write_text(json.dumps({
        "model": model, "revision": revision, "chunker": chunker, "size": size, "overlap": overlap,
        "batch_size": None if model.startswith(BM25) else bs, "batch_size_source": None if model.startswith(BM25) else bs_source,
        "prepared_dir": str(prepared.resolve()), "prepared_key": manifest["key"],
        "prepared_files_sha256": manifest["files_sha256"], "prepare_timing_s": manifest["timing_s"]}, indent=2), encoding="utf-8")
    script = "stages.run_bm25" if model.startswith(BM25) else "stages.run_eval"   # bm25, bm25-k1-2, ...
    env = dict(os.environ, PYTHONHASHSEED=str(seed if seed is not None else 0))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    print(f"eval {model} {chunker} c{size} o{overlap}")
    with open(run_dir / "stage6.log", "w", encoding="utf-8") as lf:
        extra = ["--eff-chunker", chunker, "--eff-size", str(size), "--eff-overlap", str(overlap), *eff_args] if eff_args and script == "stages.run_eval" else []
        r = subprocess.run([sys.executable, "-m", script, "--config", str(cfg_dir / "stage6.yaml"), *extra], cwd=_REPO_ROOT,
                           stdout=lf, stderr=subprocess.STDOUT, env=env)
    dur = int(round(time.time() - t0))
    (run_dir / "duration_stage6_s.txt").write_text(f"{dur}\n")
    dpath = Path(artifacts_root) / "tests" / "durations.csv"
    if not dpath.exists():
        dpath.write_text(DURATION_HEADER + "\n")
    with open(dpath, "a", encoding="utf-8") as f:
        f.write(",".join(map(str, [started, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), dur, r.returncode, "stage6",
                                   model, revision, chunker, size, overlap, run_dir])) + "\n")
    if r.returncode != 0:
        tail = (run_dir / "stage6.log").read_text(errors="replace")[-3000:]
        raise RuntimeError(f"stage 6 failed for {run_dir} (exit {r.returncode})\n{tail}")
    return {"run_dir": str(run_dir), "status": "done", "seconds": dur}


def run_hybrids(root: str, models: List[Tuple[str, str]], chunkers: List[str], cells: List[Tuple[int, int]], *, k: int, depth: int,
                force: bool = False, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Reciprocal rank fusion of every dense run with the BM25 run of the same configuration (tools/hybrid.py): no new embedding."""
    from tools import hybrid
    tests = Path(root) / "tests"
    found = hybrid.discover(tests)
    sparse = next((n for n, _ in models if n.startswith(BM25)), None)
    out = []
    if sparse is None:
        return out
    for ch in chunkers:
        for s, o in grid.cells_for(ch, cells):
            runs = found.get((ch, s, o), {})
            if sparse not in runs:
                continue
            for name, _rev in models:
                if name != sparse and not name.startswith(BM25) and name in runs:
                    r = hybrid.run_hybrid(runs[name], runs[sparse], tests, k=k, depth=depth, force=force, seed=seed)
                    print(f"hybrid rrf{k} {name} + {sparse} {ch} c{s} o{o}: {r['status']}")
                    out.append(r)
    return out


def run_reranks(root: str, models: List[Tuple[str, str]], cells: List[Tuple[int, int]], chunkers: List[str], rr: Dict[str, Any], *,
                force: bool = False, dry_run: bool = False, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Cross-encoder reranking of the finished dense runs (tools/rerank.py, params.rerank): primary depth for ``rr['models']`` (all dense
    runs of this invocation if the list is empty) plus the ablation entries. Finished rerank runs are skipped."""
    from tools import rerank
    wanted = [(ch, s, o) for ch in chunkers for s, o in grid.cells_for(ch, cells)]
    return rerank.run_configured(Path(root) / "tests", rr, models=[n for n, _ in models if not n.startswith(BM25)], wanted_cells=wanted,
                                 depth=int(rr.get("depth", 100)), force=force, dry_run=dry_run, seed=seed)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "eval", "all", "hybrid", "rerank"])
    ap.add_argument("--pilot", action="store_true", help="use params.pilot of the main config (own artifacts root)")
    ap.add_argument("--main-config", default="configs/main_config.yaml")
    ap.add_argument("--artifacts-root")
    ap.add_argument("--models", nargs="*", help="name@revision ... (bm25 for the baseline)")
    ap.add_argument("--models-csv", help="shortlist of stage 2 (default: stage2.output_per_dataset_csv of the main config)")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--no-report", action="store_true", help="skip the report step (statistics, diagnostics, tracked tables)")
    ap.add_argument("--report-dir", help="tracked folder for the summary tables (default params.report.out_dir)")
    ap.add_argument("--no-hybrid", action="store_true", help="skip the reciprocal-rank-fusion runs (dense + BM25)")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder rerank runs (params.rerank)")
    ap.add_argument("--chunkers", nargs="*")
    ap.add_argument("--sizes", nargs="*", type=int)
    ap.add_argument("--overlaps", nargs="*", type=int)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    g = ap.add_argument_group("efficiency measures (pipeline_lib/efficiency.py): never change a score or a stage timing")
    g.add_argument("--efficiency", action="store_true", help="eval: also record index / model / encoding efficiency -> <root>/tests/efficiency_phase2.csv")
    g.add_argument("--latency", action="store_true", help="eval: also measure per-query latency (batch size 1, exact cosine top-10)")
    g.add_argument("--latency-queries", type=int, help="queries timed per configuration, evenly spaced, 0 = all (default: params.efficiency.latency_queries)")
    g.add_argument("--latency-warmup", type=int, help="untimed warm-up queries (default: params.efficiency.latency_warmup)")
    g.add_argument("--latency-device", choices=["cpu", "cuda"], help="device of the timed search, cpu = as in MTEB (default: params.efficiency.latency_device)")
    g.add_argument("--latency-chunkers", nargs="*", default=[], help="latency only for these chunkers (subset of the grid)")
    g.add_argument("--latency-sizes", nargs="*", type=int, default=[], help="latency only for these sizes")
    g.add_argument("--latency-overlaps", nargs="*", type=int, default=[], help="latency only for these overlaps")
    g.add_argument("--latency-models", nargs="*", default=[], help="latency only for models whose name contains one of these")
    g.add_argument("--latency-cells", nargs="*", default=[], metavar="CHUNKER:SIZE:OVERLAP", help="latency only for exactly these configurations (default: params.efficiency.latency_cells)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--qa-sha256", help=argparse.SUPPRESS)   # tests only: the pinned release is the default
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.main_config).read_text(encoding="utf-8"))["params"]
    pilot = cfg.get("pilot") or {}
    root = args.artifacts_root or (pilot["artifacts_root"] if args.pilot else cfg["paths"]["artifacts_root"])
    chunkers = args.chunkers or (pilot["chunkers"] if args.pilot else grid.ALL_CHUNKERS)      # + the control strategies, each at its own cells (grid.CONTROL_CELLS)
    sizes = args.sizes or (pilot["sizes"] if args.pilot else grid.SIZES)
    overlaps = args.overlaps if args.overlaps is not None else (pilot["overlaps"] if args.pilot else grid.OVERLAPS)
    legacy = cfg.get("chunking_tokenizer") == "model" or any(p.startswith("chunking_tokenizer=model") for p in args.param)
    if args.models:
        models = [parse_model(m) for m in args.models]
    elif args.command == "prepare" and not legacy and not args.pilot:
        models = []                                          # the prepared folders do not depend on any model
    elif args.pilot:
        models = [parse_model(m) for m in pilot["models"]]
    else:
        models_csv = args.models_csv or default_models_csv(args.main_config, args.param)
        check_models_csv(models_csv)
        models = models_from_csv(models_csv) + ([] if args.no_bm25 else [(BM25, "rank_bm25")])
    table = pinned_revisions(args.main_config)
    bs_table = stage1_batch_sizes(args.main_config)
    models = [(n, resolve_revision(n, r, table)) for n, r in models]
    cells = grid.grid(sizes, overlaps)
    if args.pilot:
        cells = [(s, o) for s, o in sorted({(s, o) for s in sizes for o in overlaps})]   # the pilot asks for exact cells
    print(f"artifacts root: {root} | models: {[m[0] for m in models]} | chunkers: {chunkers} | cells: {cells}")
    rr_cfg = cfg.get("rerank") or {}
    if args.command in ("eval", "all", "rerank") and rr_cfg.get("enabled", False) and not args.no_rerank:
        from tools import rerank as _rerank
        bad = _rerank.check_ablation(rr_cfg, chunkers, cells, [n for n, _ in models if not n.startswith(BM25)])
        if bad and args.pilot:
            print("[WARN] pilot: " + "; ".join(bad))                  # the pilot has its own small grid
        elif bad:
            raise SystemExit("ERROR: the rerank ablation does not fit this grid (fix params.rerank.ablation, or pass --no-rerank):\n  " + "\n  ".join(bad))
    qa_path: Optional[Path] = None
    if not args.dry_run and args.command in ("prepare", "eval", "all"):
        qa_path = ensure_data(args.main_config, root, args.param)
    base_params = [f"paths.artifacts_root={root}"] + args.param

    prepared: Dict[Tuple[str, int, int, Optional[str]], Path] = {}
    if args.command in ("prepare", "all"):
        for ch in chunkers:
            for s, o in grid.cells_for(ch, cells):
                if legacy:                                   # legacy: chunks depend on the model's tokenizer
                    for name, rev in models:
                        if not name.startswith(BM25) and not args.dry_run:
                            prepared[(ch, s, o, name)] = prep.prepare_config(ch, s, o, main_config=args.main_config, artifacts_root=root,
                                                                           params=args.param + [f"model.name={name}", f"model.dir={model_dir(name)}"], force=args.force, qa_sha256=args.qa_sha256)
                elif not args.dry_run:
                    prepared[(ch, s, o, None)] = prep.prepare_config(ch, s, o, main_config=args.main_config, artifacts_root=root, params=args.param, force=args.force, qa_sha256=args.qa_sha256)
                else:
                    print(f"[dry-run] prepare {ch} c{s} o{o}")
    if args.command in ("eval", "all"):
        for name, rev in models:
            for ch in chunkers:
                for s, o in grid.cells_for(ch, cells):
                    extra = [f"model.name={name}", f"model.dir={model_dir(name)}"] if legacy else []
                    key = (ch, s, o, name if legacy and not name.startswith(BM25) else None)
                    folder = prepared.get(key) if not args.dry_run else None
                    if folder is None and not args.dry_run:
                        folder = prep.prepare_config(ch, s, o, main_config=args.main_config, artifacts_root=root,
                                                     params=args.param + (extra if legacy and not name.startswith(BM25) else []), qa_sha256=args.qa_sha256)
                    run_eval_one(name, rev, ch, s, o, folder, main_config=args.main_config, artifacts_root=root,
                                 params=args.param, force=args.force, dry_run=args.dry_run, eff_args=efficiency_args(args, root, cfg.get("efficiency")), batch_sizes=bs_table)
    hy = cfg.get("hybrid") or {}
    if args.command in ("eval", "all", "hybrid") and hy.get("enabled", False) and not args.no_hybrid and not args.dry_run:
        run_hybrids(root, models, chunkers, cells, k=int(hy.get("k", 60)), depth=int(hy.get("depth", 100)), force=args.force, seed=cfg.get("seed"))
    rr = cfg.get("rerank") or {}
    if args.command in ("eval", "all", "rerank") and rr.get("enabled", False) and not args.no_rerank:
        run_reranks(root, models, cells, chunkers, rr, force=args.force, dry_run=args.dry_run, seed=cfg.get("seed"))
    rep = cfg.get("report") or {}
    if args.command in ("eval", "all") and rep.get("enabled", False) and not args.no_report and not args.dry_run:
        from tools import make_report
        out_dir = args.report_dir or ((pilot.get("report_dir") if args.pilot else None) or rep.get("out_dir", "results/v2-revision"))
        ext = cfg.get("external") or {}
        ext_base = Path(args.artifacts_root or cfg["paths"]["artifacts_root"])           # Phase 1 results live in the main artifacts root, also for a pilot
        ext_roots = [ext_base / sub for sub in [*(ext.get("phase1_subdirs") or []), ext.get("output_subdir", "embedding_models_screening/baselines")]]
        m = make_report.build_report(root, out_dir, qa_path, resamples=int(rep.get("resamples", make_report.stats.DEFAULT_RESAMPLES)), seed=int(cfg.get("seed", 20261121)),
                                     overlap=bool(rep.get("overlap", True)), track_max_mb=float(rep.get("track_max_mb", 5)),
                                     rerank_top_n=int(rr.get("top_n", 50)), external_roots=ext_roots, main_config=args.main_config)
        if m["steps"].get("check_identical_exit_code"):
            print("WARNING: check_identical found a mismatch between models (see the report)")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

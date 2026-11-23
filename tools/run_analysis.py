"""``make analysis``: every analysis tool that runs on the outputs of the full rerun, one after the other, with the paths and settings of the main config.

  token_budget            span coverage and hit rate at a fixed budget of retrieved reference tokens          (Reviewer 2 item 7)
  subfield_breakdown      results by chemistry subfield and by rule-based question type                         (items 23, 24)
  variance_decomposition  eta^2 of model, strategy, size, overlap (Type II) with paper-level intervals          (item 20)
  winner_stability        how often each model / configuration is first across paper-level resamples            (item 19)
  phase_transfer          Phase 1 vs Phase 2 rank correlation over models, pairs, groups, scatter               (item 16)
  examples                named success / failure cases with span and chunks                                    (item 25)
  fragmentation           fragmented evidence at the small size vs gain at the large size, per question          (item 26)

(The cover-level sensitivity, item 4, and the label statistics, item 3, are steps of ``make report``; the manual validation, item 6, is
``tools/validation_sample.py`` and ``tools/kappa.py``.)

Defaults, all from ``configs/main_config.yaml``: roots = the main artifacts root and, if it exists, ``params.transfer.artifacts_root``; qa = ``<main root>/chemquests_hf_db/qa.jsonl``;
out-dir = ``params.analysis.out_dir``; resamples = ``params.report.resamples``; Phase 1 folder and shortlist = the files written by stage 1 and stage 2
(``params.paths.phase1_results_subdir``); runs overview = ``<root>/report/runs_overview.csv`` of each root (written by ``make report``, run it first).
A tool that fails does not stop the others; the exit code is 1 if any failed. ``phase_transfer`` is skipped (with a message) if its Phase 1 inputs do not exist.

  python -m tools.run_analysis                                 # everything, paths from the config
  python -m tools.run_analysis --only token_budget examples    # some tools
  python -m tools.run_analysis --skip phase_transfer --resamples 1000
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params, split_config  # noqa: E402

TOOLS = ("token_budget", "subfield_breakdown", "variance_decomposition", "winner_stability", "phase_transfer", "examples", "fragmentation")


def defaults(main_config: str) -> Dict[str, Any]:
    p = config_params.load_params(main_config)
    main_root = str(p["paths"]["artifacts_root"])
    t_root = str((p.get("transfer") or {}).get("artifacts_root") or "")
    stage2 = split_config.resolve_section(main_config, "stage2")
    stage1 = split_config.resolve_section(main_config, "stage1")
    roots = [main_root] + ([t_root] if t_root and (Path(t_root) / "tests").is_dir() else [])
    return {"roots": roots, "qa": f"{main_root}/chemquests_hf_db/qa.jsonl", "out_dir": (p.get("analysis") or {}).get("out_dir", "results/v2-revision/analysis"),
            "phase1_root": stage1["output_root"], "shortlist": stage2["output_per_dataset_csv"],
            "overviews": [f"{r}/report/runs_overview.csv" for r in roots]}


def commands(args: argparse.Namespace, d: Dict[str, Any]) -> Dict[str, List[str]]:
    """The argument list of every tool for these options and defaults."""
    roots, qa, out = args.roots or d["roots"], args.qa or d["qa"], args.out_dir or d["out_dir"]
    common = ["--roots", *roots, "--qa", qa, "--out-dir", out, "--main-config", args.main_config, "--label-set", args.label_set]
    if args.resamples:
        common += ["--resamples", str(args.resamples)]
    pt = ["--phase1-root", args.phase1_root or d["phase1_root"], "--shortlist", args.shortlist or d["shortlist"],
          "--runs-overview", *(args.runs_overview or d["overviews"]), "--out-dir", out, "--main-config", args.main_config, "--label-set", args.label_set]
    if args.resamples:
        pt += ["--resamples", str(args.resamples)]
    return {"token_budget": common, "subfield_breakdown": common, "variance_decomposition": common + ["--balanced"], "winner_stability": common,
            "phase_transfer": pt, "examples": common, "fragmentation": common}


def phase_transfer_inputs_exist(cmd: List[str]) -> Optional[str]:
    """None if the Phase 1 folder, the shortlist and every overview file exist, else the reason to skip."""
    get = lambda flag: cmd[cmd.index(flag) + 1]        # noqa: E731
    if not Path(get("--phase1-root")).is_dir():
        return f"Phase 1 folder {get('--phase1-root')} does not exist"
    if not Path(get("--shortlist")).is_file():
        return f"shortlist {get('--shortlist')} does not exist"
    i = cmd.index("--runs-overview") + 1
    missing = [x for x in cmd[i:] if not x.startswith("--") and not Path(x).is_file()]
    return f"runs overview missing: {missing}" if missing else None


def run(args: argparse.Namespace, runner: Optional[Callable[[str, List[str]], int]] = None) -> Dict[str, str]:
    """Run the chosen tools; returns {tool: 'ok' | 'skipped: ...' | 'FAILED: ...'}."""
    d = defaults(args.main_config)
    cmds = commands(args, d)
    chosen = [t for t in (args.only or TOOLS) if t not in (args.skip or [])]
    status: Dict[str, str] = {}
    for tool in chosen:
        cmd = cmds[tool]
        if tool == "phase_transfer":
            why = phase_transfer_inputs_exist(cmd)
            if why:
                status[tool] = f"skipped: {why}"
                print(f"[analysis] {tool}: skipped ({why})")
                continue
        print(f"[analysis] {tool} ...", flush=True)
        t0 = time.time()
        try:
            rc = runner(tool, cmd) if runner else importlib.import_module(f"tools.{tool}").main(cmd)
            status[tool] = "ok" if not rc else f"FAILED: exit code {rc}"
        except SystemExit as e:
            status[tool] = "ok" if not e.code else f"FAILED: {e.code}"
        except Exception as e:                                          # noqa: BLE001 - one failing tool must not stop the others
            traceback.print_exc()
            status[tool] = f"FAILED: {type(e).__name__}: {e}"
        print(f"[analysis] {tool}: {status[tool]} ({time.time() - t0:.0f} s)")
    return status


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-config", default=str(config_params.DEFAULT_MAIN_CONFIG))
    ap.add_argument("--roots", nargs="+")
    ap.add_argument("--qa")
    ap.add_argument("--out-dir")
    ap.add_argument("--phase1-root")
    ap.add_argument("--shortlist")
    ap.add_argument("--runs-overview", nargs="+")
    ap.add_argument("--resamples", type=int)
    ap.add_argument("--label-set", choices=["span", "lexical"], default="span")
    ap.add_argument("--only", nargs="+", choices=TOOLS)
    ap.add_argument("--skip", nargs="+", choices=TOOLS)
    args = ap.parse_args(argv)
    status = run(args)
    print("[analysis] summary: " + "; ".join(f"{k}: {v.split(':')[0]}" for k, v in status.items()))
    return 1 if any(v.startswith("FAILED") for v in status.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

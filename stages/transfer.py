"""Transfer runs (``make transfer``): models that Phase 1 did not shortlist, on a few ChemQuests configurations, in the new pipeline.

Settings: ``params.transfer`` of ``configs/main_config.yaml`` (models with pinned revisions, configurations, own artifacts root). For every
configuration this calls ``stages.rerun eval`` once with all models of the block (``--models name@revision ...``), without BM25, hybrid and
rerank (so the rerank-ablation check of the main run is not triggered), with the efficiency flags given here, and with the chunk cache and the
ChemQuests inputs of the MAIN artifacts root: a transfer run reads byte-identical chunks and labels as the main run (build them first with
``make prepare``). The report is written once, after the last configuration, to ``params.transfer.report_dir``.

Checked before anything starts (``validate``): every model has a 40-hex pinned revision (equal to the ``stage1.models`` pin for Phase 1 models), has
an entry in ``params.model_prompts``, appears once; every configuration is valid in ``pipeline_lib/grid.py``, has a known chunker and appears once;
the transfer root differs from the main root; every model of a pair is in the transfer model list. ``check_roles`` compares the roles with the stage 2 file.

  python -m stages.transfer --efficiency --latency          # what `make transfer` runs
  python -m stages.transfer --dry-run                       # print the commands and the planned runs, run nothing
  python -m stages.transfer --no-extras                     # skip role=extra models and extra configurations
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from pipeline_lib import grid  # noqa: E402

HEX40 = re.compile(r"^[0-9a-f]{40}$")
ROLES = ("non_shortlisted", "shortlisted_reference", "extra")


class TransferConfigError(ValueError):
    """``params.transfer`` is inconsistent with the rest of the main config."""


def load(main_config: str) -> Dict[str, Any]:
    doc = yaml.safe_load(Path(main_config).read_text(encoding="utf-8"))
    if "transfer" not in (doc.get("params") or {}):
        raise TransferConfigError("params.transfer is missing in the main config")
    return doc


def validate(doc: Dict[str, Any]) -> List[str]:
    """All problems of ``params.transfer`` (empty list = fine)."""
    p, tr = doc["params"], doc["params"]["transfer"]
    stage1 = {m["name"]: m["revision"] for m in doc["stage1"]["models"]}
    prompts = p.get("model_prompts") or {}
    problems: List[str] = []
    names = [m.get("name") for m in tr.get("models") or []]
    if not names:
        problems.append("params.transfer.models is empty")
    for n in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"model {n} is listed twice")
    for m in tr.get("models") or []:
        n, rev, role = m.get("name"), str(m.get("revision", "")), m.get("role")
        if role not in ROLES:
            problems.append(f"{n}: role must be one of {ROLES}, got {role!r}")
        if not HEX40.match(rev):
            problems.append(f"{n}: the revision must be a pinned 40-hex commit, got {rev!r}")
        if n in stage1 and stage1[n] != rev:
            problems.append(f"{n}: revision {rev[:8]} differs from the Phase 1 pin {stage1[n][:8]} (stage1.models)")
        if role in ("non_shortlisted", "shortlisted_reference") and n not in stage1:
            problems.append(f"{n}: role {role} but the model is not in stage1.models")
        if n not in prompts:
            problems.append(f"{n}: no entry in params.model_prompts (give an entry with empty strings for 'no prompt')")
    have = set(names)
    for pr in tr.get("pairs") or []:
        for n in [pr.get("base"), *(pr.get("tuned") or [])]:
            if n not in have:
                problems.append(f"pair {pr.get('id')}: model {n} is not in params.transfer.models (its runs would be missing)")
    ids = [c.get("id") for c in tr.get("configurations") or []]
    if not ids:
        problems.append("params.transfer.configurations is empty")
    for i in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"configuration id {i} is listed twice")
    seen = set()
    for c in tr.get("configurations") or []:
        key = (c.get("chunker"), c.get("size"), c.get("overlap"))
        if key[0] not in grid.CHUNKERS:
            problems.append(f"{c.get('id')}: unknown chunker {key[0]!r}")
        elif not (isinstance(key[1], int) and isinstance(key[2], int) and grid.is_valid(key[1], key[2])):
            problems.append(f"{c.get('id')}: size {key[1]} / overlap {key[2]} is not a valid configuration of the grid (pipeline_lib/grid.py)")
        if key in seen:
            problems.append(f"{c.get('id')}: {key} is listed twice")
        seen.add(key)
    main_root, t_root = str(p["paths"]["artifacts_root"]).rstrip("/"), str(tr.get("artifacts_root", "")).rstrip("/")
    if not t_root or t_root == main_root:
        problems.append(f"params.transfer.artifacts_root must be its own folder, different from {main_root!r}")
    return problems


def check_roles(doc: Dict[str, Any], shortlisted: set) -> List[str]:
    """Problems of the roles against the Phase 2 model list (stage 2 output): a ``non_shortlisted`` model must NOT be on it, a ``shortlisted_reference`` must be."""
    out = []
    for m in doc["params"]["transfer"].get("models") or []:
        if m["role"] == "non_shortlisted" and m["name"] in shortlisted:
            out.append(f"{m['name']}: role non_shortlisted, but the model is on the Phase 2 list (stage 2)")
        if m["role"] == "shortlisted_reference" and m["name"] not in shortlisted:
            out.append(f"{m['name']}: role shortlisted_reference, but the model is not on the Phase 2 list (stage 2)")
    return out


def shortlist_names(doc: Dict[str, Any], main_config: str) -> Optional[set]:
    """Model names of the stage 2 file (``None`` if stage 2 has not run)."""
    import csv
    from pipeline_lib import split_config
    path = Path(split_config.resolve_section(main_config, "stage2")["output_per_dataset_csv"])
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return {r["model_name"] for r in csv.DictReader(f)}


def selected(doc: Dict[str, Any], extras: bool = True, references: Optional[bool] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Models and configurations to run (``run_extras`` / ``run_shortlisted_references`` of the config, overridden by the arguments)."""
    tr = doc["params"]["transfer"]
    use_extras = extras and bool(tr.get("run_extras", True))
    use_refs = bool(tr.get("run_shortlisted_references", True)) if references is None else references
    models = [m for m in tr["models"] if (m["role"] != "extra" or use_extras) and (m["role"] != "shortlisted_reference" or use_refs)]
    configs = [c for c in tr["configurations"] if use_extras or not c.get("extra")]
    return {"models": models, "configurations": configs}


def build_calls(doc: Dict[str, Any], main_config: str, *, efficiency: bool, latency: bool, extras: bool = True,
                references: Optional[bool] = None, extra_args: Optional[List[str]] = None) -> List[List[str]]:
    """The ``stages.rerun`` argument lists, one per configuration (the report only in the last one)."""
    p, tr = doc["params"], doc["params"]["transfer"]
    sel = selected(doc, extras, references)
    main_root = str(p["paths"]["artifacts_root"]).rstrip("/")
    calls = []
    for i, c in enumerate(sel["configurations"]):
        argv = ["eval", "--main-config", main_config, "--artifacts-root", str(tr["artifacts_root"]),
                "--models", *[f"{m['name']}@{m['revision']}" for m in sel["models"]],
                "--chunkers", c["chunker"], "--sizes", str(c["size"]), "--overlaps", str(c["overlap"]),
                "--no-bm25", "--no-hybrid", "--no-rerank",
                "--param", f"chunk_cache.dir={main_root}/prepared", "--param", f"paths.chemquests_dir={main_root}/chemquests_hf_db"]
        if efficiency or latency:
            argv.append("--efficiency")
        if latency:
            argv.append("--latency")
        if i < len(sel["configurations"]) - 1:
            argv.append("--no-report")
        else:
            argv += ["--report-dir", str(tr["report_dir"])]
        argv += list(extra_args or [])
        calls.append(argv)
    return calls


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-config", default="configs/main_config.yaml")
    ap.add_argument("--efficiency", action="store_true", help="record index / model / encoding efficiency (as `make eval`)")
    ap.add_argument("--latency", action="store_true", help="also per-query latency for the configurations of params.efficiency")
    ap.add_argument("--no-extras", action="store_true", help="skip role=extra models and configurations marked extra")
    ap.add_argument("--no-references", action="store_true", help="skip the shortlisted reference models (they are in the main run)")
    ap.add_argument("--dry-run", action="store_true", help="validate and print, run nothing")
    ap.add_argument("--force", action="store_true")
    args, rest = ap.parse_known_args(argv)
    doc = load(args.main_config)
    problems = validate(doc)
    if problems:
        raise SystemExit("ERROR: params.transfer is not valid:\n  " + "\n  ".join(problems))
    names = shortlist_names(doc, args.main_config)
    if names is None:
        print("WARNING: the stage 2 file does not exist yet: the roles of params.transfer are not checked against the Phase 2 list")
    else:
        bad = check_roles(doc, names)
        if bad:
            raise SystemExit("ERROR: params.transfer roles disagree with the Phase 2 list:\n  " + "\n  ".join(bad))
    calls = build_calls(doc, args.main_config, efficiency=args.efficiency, latency=args.latency, extras=not args.no_extras,
                        references=False if args.no_references else None, extra_args=rest + (["--force"] if args.force else []))
    sel = selected(doc, not args.no_extras, False if args.no_references else None)
    print(f"transfer: {len(sel['models'])} models x {len(sel['configurations'])} configurations = {len(sel['models']) * len(sel['configurations'])} runs "
          f"-> {doc['params']['transfer']['artifacts_root']}")
    from stages import rerun
    rc = 0
    for call in calls:
        print("stages.rerun " + " ".join(call[:1] + call[call.index("--chunkers"):call.index("--overlaps") + 2]))
        if args.dry_run:
            continue
        rc = rerun.main(call) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

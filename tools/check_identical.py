"""Check that every model evaluated a configuration on byte-identical chunks and labels.

For each configuration (chunker, size, overlap) found under ``<tests root>/<model>_<chunker>_c<L>_o<O>/`` the files the
models read are hashed (sha256) THROUGH the run folders: ``chunks/chunks.jsonl``, ``mteb_task/corpus.jsonl``,
``queries.jsonl`` and every ``qrels_*.jsonl``. They must be

  1. identical across all models of the configuration,
  2. equal to the hashes in the prepared folder's manifest.json (nothing changed after preparation), and
  3. equal to the hashes the evaluation process recorded when it loaded them (results/**/run_manifest.json).

Exit code 1 if anything differs. ``--out`` writes a CSV (one row per configuration and file).

  python -m tools.check_identical --tests-root artifacts/pilot/tests --chunker fixed_token --size 128 --overlap 0
  python -m tools.check_identical --tests-root artifacts/tests          # every configuration found
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib.grid import ALL_CHUNKERS as CHUNKERS  # noqa: E402     # incl. the control strategies (fixed_token_nomatter)
from pipeline_lib.run_manifest import sha256_file  # noqa: E402

_RUN = re.compile(r"^(?P<model>.+)_(?P<chunker>" + "|".join(sorted(CHUNKERS, key=len, reverse=True)) + r")_c(?P<size>\d+)_o(?P<overlap>\d+)$")


def _files(run: Path) -> Dict[str, Path]:
    out = {}
    for rel in ("chunks/chunks.jsonl", "mteb_task/corpus.jsonl", "mteb_task/queries.jsonl"):
        if (run / rel).is_file():
            out[rel] = run / rel
    for q in sorted((run / "mteb_task").glob("qrels_*.jsonl")):
        out[f"mteb_task/{q.name}"] = q
    return out


def _eval_hashes(run: Path) -> Dict[str, str]:
    """Hashes the evaluation process recorded for corpus / queries / qrels (file name -> sha256)."""
    out: Dict[str, str] = {}
    for m in sorted((run / "results").rglob("run_manifest.json")):
        for p, h in (json.loads(m.read_text()).get("data_sha256") or {}).items():
            out[Path(p).name] = h
    return out


def check_configuration(runs: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Rows {file, ok, models, distinct_hashes, problem} for one configuration; ``runs`` = {model: run folder}."""
    per_model: Dict[str, Dict[str, str]] = {m: {rel: sha256_file(p) for rel, p in _files(r).items()} for m, r in runs.items()}
    rels = sorted({rel for h in per_model.values() for rel in h})
    manifests = {}
    for m, r in runs.items():
        info = r / "run_info.json"
        manifests[m] = json.loads(info.read_text()).get("prepared_files_sha256", {}) if info.exists() else {}
    rows = []
    for rel in rels:
        problems = []
        hashes = {m: h.get(rel) for m, h in per_model.items()}
        if None in hashes.values():
            problems.append("missing for " + ", ".join(m for m, h in hashes.items() if h is None))
        distinct = {h for h in hashes.values() if h}
        if len(distinct) > 1:
            problems.append("differs between models")
        for m, man in manifests.items():
            if man.get(rel) and hashes.get(m) and man[rel] != hashes[m]:
                problems.append(f"{m}: differs from the prepared manifest")
        for m, r in runs.items():
            ev = _eval_hashes(r).get(Path(rel).name)
            if rel != "chunks/chunks.jsonl" and ev and hashes.get(m) and ev != hashes[m]:
                problems.append(f"{m}: differs from the hash recorded when the model was evaluated")
        rows.append({"file": rel, "ok": not problems, "models": len(runs), "distinct_hashes": len(distinct),
                     "sha256": next(iter(distinct)) if len(distinct) == 1 else "", "problem": "; ".join(problems)})
    return rows


def find_configurations(tests_root: Path) -> Dict[tuple, Dict[str, Path]]:
    cfgs: Dict[tuple, Dict[str, Path]] = defaultdict(dict)
    for d in sorted(tests_root.iterdir()):
        m = _RUN.match(d.name) if d.is_dir() else None
        if m and ((d / "mteb_task").exists() or (d / "chunks").exists()):
            cfgs[(m["chunker"], int(m["size"]), int(m["overlap"]))][m["model"]] = d
    return cfgs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests-root", required=True)
    ap.add_argument("--chunker")
    ap.add_argument("--size", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--out", help="CSV report")
    args = ap.parse_args(argv)
    cfgs = find_configurations(Path(args.tests_root))
    if args.chunker is not None:
        cfgs = {k: v for k, v in cfgs.items() if k == (args.chunker, args.size, args.overlap)}
    if not cfgs:
        print("no configuration found")
        return 1
    report, bad = [], 0
    for (ch, s, o), runs in sorted(cfgs.items()):
        rows = check_configuration(runs)
        n_ok = sum(r["ok"] for r in rows)
        status = "OK" if n_ok == len(rows) and len(runs) > 1 else ("SINGLE MODEL" if len(runs) == 1 and n_ok == len(rows) else "MISMATCH")
        print(f"{ch} c{s} o{o}: {len(runs)} models ({', '.join(sorted(runs))}) -> {status}")
        for r in rows:
            print(f"   {'ok ' if r['ok'] else 'BAD'} {r['file']:34s} {r['sha256'][:16] or '-':16s} {r['problem']}")
            report.append({"chunker": ch, "size": s, "overlap": o, **r})
        bad += status == "MISMATCH"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0]))
            w.writeheader()
            w.writerows(report)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

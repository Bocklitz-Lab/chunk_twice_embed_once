"""Prepare: chunk ONE configuration and build its labels, independent of any embedding model (stages 4 + 5).

For one (strategy, size, overlap) this writes ``<prepared root>/<strategy>_c<L>_o<O>__<key8>/``:

  chunks/chunks.jsonl, chunks/chunking_stats.json     stage 4 (the reference tokenizer counts the sizes)
  mteb_task/corpus.jsonl, queries.jsonl                  the MTEB task
  mteb_task/qrels_lexical.jsonl                          lexical labels (BM25 + Jaccard + fuzzy)
  mteb_task/qrels_span_c<c>.jsonl, overlap_table.jsonl, label_stats_c<c>.json     span labels
  coverage_diagnostics.csv, no_relevant_chunk.csv      (see tools/coverage_diagnostics.py)
  config/stage4.yaml, stage5.yaml, manifest.json         manifest last: a folder without it is incomplete

Every evaluated model reuses the folder (see ``stages/rerun.py``); the key (pipeline_lib/chunk_cache.py) changes with
anything that changes the content, so a stale folder is never reused. Both qrels files are always written.

  python -m stages.prepare --chunker fixed_token --size 128 --overlap 0           # prints the folder
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_lib import chunk_cache, split_config  # noqa: E402
from pipeline_lib.run_manifest import git_info, package_versions, sha256_file  # noqa: E402
from pipeline_lib.span_labels import qrels_filename  # noqa: E402
from tools import coverage_diagnostics  # noqa: E402

LEXICAL_QRELS = "qrels_lexical.jsonl"


def _dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=split_config.EscapingDumper,
                  default_flow_style=False, width=4096)


def resolve_stage_configs(main_config: str, chunker: str, size: int, overlap: int, artifacts_root: str,
                          params: List[str]) -> Dict[str, Dict[str, Any]]:
    """Stage 4 / 5 configs exactly as split_config resolves them for this configuration."""
    with tempfile.TemporaryDirectory() as tmp:
        argv = ["--input", main_config, "--output-dir", tmp, "--sections", "stage4", "stage5",
                "--param", f"chunk.chunker={chunker}", "--param", f"chunk.size={size}", "--param", f"chunk.overlap={overlap}",
                "--param", f"paths.artifacts_root={artifacts_root}"]
        for p in params:
            argv += ["--param", p]
        rc = split_config.main(argv)
        if rc != 0:
            raise RuntimeError(f"split_config failed (exit {rc})")
        return {s: yaml.safe_load((Path(tmp) / f"{s}.yaml").read_text(encoding="utf-8")) for s in ("stage4", "stage5")}


def _run(cmd: List[str], log: Path) -> float:
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, cwd=_REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        tail = Path(log).read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n--- {log} (tail) ---\n{tail}")
    return dt


def prepare_config(chunker: str, size: int, overlap: int, *, main_config: str = "configs/main_config.yaml",
                   artifacts_root: str = "artifacts", params: Optional[List[str]] = None, qa_sha256: Optional[str] = None,
                   force: bool = False, verbose: bool = True) -> Path:
    """Build (or find) the prepared folder of one configuration and return its path."""
    params = list(params or [])
    cfgs = resolve_stage_configs(main_config, chunker, size, overlap, artifacts_root, params)
    s4, s5 = cfgs["stage4"], cfgs["stage5"]
    if qa_sha256:
        s5["qa_sha256"] = qa_sha256   # tests only; the pinned release is the default
    root_cfg = yaml.safe_load(Path(main_config).read_text(encoding="utf-8"))["params"]
    cache_dir = str(root_cfg["chunk_cache"]["dir"]).replace("${params.paths.artifacts_root}", artifacts_root)
    for p in params:                                   # allow --param chunk_cache.dir=...
        k, v = split_config.parse_param(p)
        if k == "chunk_cache.dir":
            cache_dir = v
    ck = chunk_cache.compute_key(s4, s5, s4["input_path"], s5["qa_jsonl_file"], _REPO_ROOT)
    folder = Path(cache_dir) / chunk_cache.folder_name(chunker, size, overlap, ck["key"])
    if not force and chunk_cache.is_complete(folder, ck["key"]):
        if verbose:
            print(f"prepared (cached): {folder}")
        return folder

    if verbose:
        print(f"preparing {chunker} c{size} o{overlap} -> {folder}")
    (folder / "chunks").mkdir(parents=True, exist_ok=True)
    (folder / "mteb_task").mkdir(parents=True, exist_ok=True)
    (folder / "logs").mkdir(exist_ok=True)
    # a half-built folder of the same key is rebuilt from scratch: remove its manifest first
    if (folder / chunk_cache.MANIFEST).exists():
        (folder / chunk_cache.MANIFEST).unlink()

    s4["output_path"] = str(folder / "chunks" / "chunks.jsonl")
    s5.update({"chunks_jsonl_file": s4["output_path"], "output_folder": str(folder / "mteb_task")})
    _dump(folder / "config" / "stage4.yaml", s4)
    _dump(folder / "config" / "stage5.yaml", s5)
    py = sys.executable
    timing: Dict[str, float] = {}
    timing["chunking_s"] = _run([py, "-m", "stages.chunking", "--config", str(folder / "config" / "stage4.yaml")],
                                folder / "logs" / "stage4.log")
    task = folder / "mteb_task"
    timing["qrels_span_s"] = _run([py, "-m", "stages.make_span_qrels", "--config", str(folder / "config" / "stage5.yaml")],
                                  folder / "logs" / "stage5_span.log")
    shared = {n: sha256_file(task / n) for n in ("corpus.jsonl", "queries.jsonl")}
    timing["qrels_lexical_s"] = _run([py, "-m", "stages.make_mteb_task", "--config", str(folder / "config" / "stage5.yaml")],
                                     folder / "logs" / "stage5_lexical.log")
    for n, h in shared.items():   # both label scripts write the corpus and the queries: they must agree byte for byte
        if sha256_file(task / n) != h:
            raise RuntimeError(f"{n} differs between the span and the lexical stage 5 in {folder}")

    labeling = s5["labeling"]
    span_file = labeling.get("qrels_file") or qrels_filename("span", labeling["cover_level"])
    if not (task / span_file).exists():
        span_file = qrels_filename("span", labeling["cover_level"])
    sp = s4["strategy"]["params"]
    t0 = time.perf_counter()
    coverage_diagnostics.diagnose_config(task, s4["output_path"], sp.get("model_name"), sp.get("tokenizer_revision"),
                                         span_qrels=span_file, out_dir=folder)
    timing["diagnostics_s"] = time.perf_counter() - t0

    stats_p = folder / "chunks" / "chunking_stats.json"
    rel_files = ["chunks/chunks.jsonl", "chunks/chunking_stats.json"] + sorted(
        f"mteb_task/{p.name}" for p in task.iterdir() if p.is_file())
    manifest = {
        "status": "complete", "key": ck["key"], "key_inputs": ck["inputs"],
        "config_name": f"{chunker}_c{size}_o{overlap}", "chunker": chunker, "size": size, "overlap": overlap,
        "chunking_mode": (root_cfg.get("chunking_tokenizer") == "model" and "per_model_legacy") or "reference",
        "tokenizer": sp.get("model_name"), "tokenizer_revision": sp.get("tokenizer_revision"),
        "breakpoint_model": sp.get("breakpoint_model_name"), "breakpoint_model_revision": sp.get("breakpoint_model_revision"),
        "span_qrels_file": span_file, "lexical_qrels_file": LEXICAL_QRELS,
        "timing_s": timing, "chunking_stats": json.loads(stats_p.read_text()) if stats_p.exists() else None,
        "files_sha256": chunk_cache.files_sha256(folder, rel_files),
        "git": git_info(_REPO_ROOT), "packages": package_versions(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (folder / chunk_cache.MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"prepared: {folder}  ({', '.join(f'{k}={v:.1f}' for k, v in timing.items())})")
    return folder


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chunk one configuration and build its qrels (no embedding model).")
    ap.add_argument("--chunker", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--overlap", type=int, required=True)
    ap.add_argument("--main-config", default="configs/main_config.yaml")
    ap.add_argument("--artifacts-root", default="artifacts")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="override params.KEY (split_config)")
    ap.add_argument("--force", action="store_true", help="rebuild even if a complete folder exists")
    args = ap.parse_args(argv)
    folder = prepare_config(args.chunker, args.size, args.overlap, main_config=args.main_config,
                            artifacts_root=args.artifacts_root, params=args.param, force=args.force)
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

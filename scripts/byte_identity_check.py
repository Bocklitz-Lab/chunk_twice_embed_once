"""Byte-identity check of the pipeline outputs between two code versions on a SYNTHETIC corpus (CPU: all-MiniLM-L6-v2, BM25, hybrid).

  python scripts/byte_identity_check.py --base-ref split-and-figures --work /tmp/idcheck

Builds the synthetic corpus of tests/analysis_world.py, runs ``stages.rerun eval`` (fixed_token and recursive_token, sizes 128 / 256, overlap 0, MiniLM dense, BM25,
RRF hybrid, no rerank, no report) once with the code of ``--base-ref`` (a git worktree) and once with the working tree, and compares: corpus, queries, span qrels,
lexical qrels, the decompressed top-100 rankings, per-query metrics (both label sets) and the scores of ChemQuest.json. Files that exist in one tree only are listed
(expected: the base tree has ``qrels.jsonl``). Times are not compared. Exit code 1 on a difference in a compared file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MINILM = "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
COMPARE = ("qrels.jsonl", "corpus.jsonl", "queries.jsonl", "qrels_span_c090.jsonl", "qrels_lexical.jsonl", "per_query_metrics.csv", "per_query_metrics__qrels_lexical.csv",
           "chunks.jsonl", "overlap_table.jsonl")


def run_tree(code: Path, root: Path, data: Path, sha: str, env: dict) -> None:
    shutil.rmtree(root, ignore_errors=True)
    (root).mkdir(parents=True)
    shutil.copytree(data, root / "chemquests_hf_db")
    cmd = [str(REPO / "myenv/bin/python"), "-m", "stages.rerun", "eval", "--artifacts-root", str(root), "--models", MINILM, "bm25@rank_bm25",
           "--chunkers", "fixed_token", "recursive_token", "--sizes", "128", "256", "--overlaps", "0", "--no-rerank", "--no-report", "--qa-sha256", sha]
    r = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True)
    (root / "run.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"run failed in {code} (exit {r.returncode}); see {root / 'run.log'}")


def files(root: Path) -> dict:
    out = {}
    for p in sorted((root / "prepared").rglob("*")) + sorted((root / "tests").rglob("*")):
        if p.is_file() and not p.is_symlink() or (p.is_file() and p.is_symlink()):
            rel = p.relative_to(root)
            if p.name in COMPARE or p.name.startswith("run_top100") or p.name == "ChemQuest.json":
                out[str(rel)] = p
    return out


def content(p: Path) -> bytes:
    if p.name.endswith(".gz"):
        return gzip.open(p, "rb").read()
    if p.name == "ChemQuest.json":
        j = json.loads(p.read_text())
        j.pop("evaluation_time", None)
        j.pop("kg_co2_emissions", None)
        return json.dumps(j, sort_keys=True).encode()
    return p.read_bytes()


def norm(rel: str) -> str:
    """The prepared folder name carries the cache key, which may differ between code versions: compare by configuration."""
    parts = Path(rel).parts
    if parts[0] == "prepared":
        parts = (parts[0], parts[1].split("__")[0], *parts[2:])
    return "/".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ref", default="split-and-figures")
    ap.add_argument("--work", default="/tmp/idcheck")
    a = ap.parse_args()
    work = Path(a.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sys.path.insert(0, str(REPO / "tests"))
    sys.path.insert(0, str(REPO))
    from analysis_world import build_corpus
    data = build_corpus(work / "synthetic")["dir"]
    sha = hashlib.sha256((data / "qa.jsonl").read_bytes()).hexdigest()
    base = work / "base_code"
    subprocess.run(["git", "worktree", "add", "--detach", str(base), a.base_ref], cwd=REPO, check=True, capture_output=True)
    env = {**os.environ, "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ""}
    try:
        run_tree(base, work / "base_root", data, sha, env)
        run_tree(REPO, work / "new_root", data, sha, env)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(base)], cwd=REPO, capture_output=True)
    fa, fb = ({norm(k): v for k, v in files(work / r).items()} for r in ("base_root", "new_root"))
    diff, same = [], 0
    for k in sorted(set(fa) & set(fb)):
        if content(fa[k]) == content(fb[k]):
            same += 1
        else:
            diff.append(k)
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    print(f"compared {same + len(diff)} files: identical {same}, different {len(diff)}")
    for k in diff:
        print("  DIFFERENT:", k)
    print("only in the base tree:", only_a or "-")
    print("only in the new tree:", only_b or "-")
    return 1 if diff else 0


if __name__ == "__main__":
    raise SystemExit(main())

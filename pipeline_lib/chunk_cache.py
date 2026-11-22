"""Prepared-chunk cache: chunks + qrels built once per (strategy, size, overlap) and reused by every model.

A prepared folder is ``<root>/<chunker>_c<L>_o<O>__<key8>/``. ``key`` is a hash over everything that determines its
content: the resolved stage 4 and stage 5 configs (without paths), the hashes of ``full_text.jsonl`` and ``qa.jsonl``,
the source of the chunkers / labelling code, and the versions of the tokenizer and splitter libraries. A change in any
of them gives a new folder; an old folder is never overwritten or deleted by a different key.
``manifest.json`` is written last and holds the hash of every file, so a folder is complete only with its manifest.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pipeline_lib.run_manifest import package_versions, sha256_file

CODE_FILES = (
    "pipeline_lib/fixed_token_chunker.py", "pipeline_lib/recursive_token_chunker.py",
    "pipeline_lib/hierarchical_section_chunker.py", "pipeline_lib/semantic_chunker.py",
    "pipeline_lib/span_labels.py", "stages/chunking.py", "stages/make_span_qrels.py", "stages/make_mteb_task.py",
)
KEY_PACKAGES = ("transformers", "tokenizers", "langchain-text-splitters", "langchain-experimental", "sentence-transformers")
STAGE4_PATHS = ("input_path", "output_path")
STAGE5_PATHS = ("qa_jsonl_file", "full_text_jsonl_file", "chunks_jsonl_file", "output_folder")
MANIFEST = "manifest.json"


def strip_paths(stage4: Dict[str, Any], stage5: Dict[str, Any]):
    """Copies of the stage configs without anything that is only a location (paths, cache directories)."""
    s4, s5 = copy.deepcopy(stage4), copy.deepcopy(stage5)
    for k in STAGE4_PATHS:
        s4.pop(k, None)
    for k in STAGE5_PATHS:
        s5.pop(k, None)
    s4.get("strategy", {}).get("params", {}).pop("cache_dir", None)   # speeds up semantic chunking, never changes it
    return s4, s5


def code_hashes(repo_root: str | Path) -> Dict[str, str]:
    root = Path(repo_root)
    return {f: sha256_file(root / f) for f in CODE_FILES if (root / f).exists()}


def compute_key(stage4: Dict[str, Any], stage5: Dict[str, Any], full_text: str | Path, qa: str | Path,
                repo_root: str | Path) -> Dict[str, Any]:
    s4, s5 = strip_paths(stage4, stage5)
    inputs = {
        "stage4": s4, "stage5": s5,
        "full_text_sha256": sha256_file(full_text), "qa_sha256": sha256_file(qa),
        "code_sha256": code_hashes(repo_root),
        "packages": {k: v for k, v in package_versions(KEY_PACKAGES).items()},
    }
    blob = json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False)
    return {"key": hashlib.sha256(blob.encode("utf-8")).hexdigest(), "inputs": inputs}


def folder_name(chunker: str, size: int, overlap: int, key: str) -> str:
    return f"{chunker}_c{size}_o{overlap}__{key[:8]}"


def read_manifest(folder: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(folder) / MANIFEST
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_complete(folder: str | Path, key: Optional[str] = None) -> bool:
    m = read_manifest(folder)
    return bool(m and m.get("status") == "complete" and (key is None or m.get("key") == key))


def files_sha256(folder: str | Path, relpaths: Iterable[str]) -> Dict[str, str]:
    return {r: sha256_file(Path(folder) / r) for r in relpaths if (Path(folder) / r).is_file()}


def verify_files(folder: str | Path) -> Dict[str, str]:
    """{relative path: problem} for every file whose hash differs from the manifest (empty = intact)."""
    m = read_manifest(folder) or {}
    bad: Dict[str, str] = {}
    for rel, expected in (m.get("files_sha256") or {}).items():
        p = Path(folder) / rel
        if not p.is_file():
            bad[rel] = "missing"
        elif sha256_file(p) != expected:
            bad[rel] = "hash differs from manifest"
    return bad

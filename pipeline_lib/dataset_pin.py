"""Pinned ChemQuests release used for every published result of this study.

The span labels depend on ``start_index`` / ``end_index`` in ``qa.jsonl``; two
HF revisions differ in those fields (927 / 425 of 952 questions), so the
dataset must never be fetched from a moving ref such as ``main``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHEMQUESTS_REPO_ID = "Bocklitz-Lab/ChemQuests"
CHEMQUESTS_REVISION = "5ab7dc673653b800f368fe30851efbdb24385e0d"
CHEMQUESTS_QA_SHA256 = "e66ced49821787f7448b688a261ca0bb65c5cfe0f9961753316aec52c6b20e8d"


class DatasetIntegrityError(RuntimeError):
    """The qa.jsonl on disk is not the pinned release."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_qa_sha256(path: str | Path, expected: str = CHEMQUESTS_QA_SHA256) -> str:
    """Raise DatasetIntegrityError unless SHA256(path) == expected. Returns the digest."""
    path = Path(path)
    if not path.is_file():
        raise DatasetIntegrityError(f"qa.jsonl not found for integrity check: {path}")
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise DatasetIntegrityError(
            f"qa.jsonl SHA256 mismatch for {path}:\n  expected {expected}\n  actual   {actual}\n"
            f"The pinned ChemQuests release is {CHEMQUESTS_REPO_ID}@{CHEMQUESTS_REVISION}. "
            "Refusing to continue: span labels would be computed on different start/end indices."
        )
    return actual

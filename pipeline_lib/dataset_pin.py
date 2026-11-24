"""Pinned ChemQuests release used for every published result of this study.

The span labels depend on ``start_index`` / ``end_index`` in ``qa.jsonl``; two
HF revisions differ in those fields (927 / 425 of 952 questions), so the
dataset must never be fetched from a moving ref such as ``main``.

``qa/question_metadata.jsonl`` (question type and chemistry subfields per question) is not part of that release. It was added in a later
dataset commit whose tree is identical to the release in every other file (same git blob ids for qa.jsonl, metadata.jsonl and every
full_text/<id>.txt, checked on 2026-11-24), so it has its OWN pin: the benchmark data stays at ``CHEMQUESTS_REVISION``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHEMQUESTS_REPO_ID = "Bocklitz-Lab/ChemQuests"
CHEMQUESTS_REVISION = "5ab7dc673653b800f368fe30851efbdb24385e0d"
CHEMQUESTS_QA_SHA256 = "e66ced49821787f7448b688a261ca0bb65c5cfe0f9961753316aec52c6b20e8d"

# separate pin of the per-question labels (first and only dataset commit that holds the file at the time of writing)
QUESTION_METADATA_REPO_PATH = "qa/question_metadata.jsonl"      # path in the dataset repository
QUESTION_METADATA_FILE = "question_metadata.jsonl"               # name next to qa.jsonl in the prepared data
QUESTION_METADATA_REVISION = "790d2bc6829ccd7e1e480a6a4610149650b79e9f"
QUESTION_METADATA_SHA256 = "4e1f83fbe22323bfd358e95c3814cd39b508d76d67df2680410acc7fb31c5a70"


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


def verify_question_metadata_sha256(path: str | Path, expected: str = QUESTION_METADATA_SHA256) -> str:
    """Raise DatasetIntegrityError unless SHA256(path) == expected (the question metadata file). Returns the digest."""
    path = Path(path)
    if not path.is_file():
        raise DatasetIntegrityError(f"{QUESTION_METADATA_FILE} not found for integrity check: {path}")
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise DatasetIntegrityError(
            f"{QUESTION_METADATA_FILE} SHA256 mismatch for {path}:\n  expected {expected}\n  actual   {actual}\n"
            f"The pinned file is {CHEMQUESTS_REPO_ID}@{QUESTION_METADATA_REVISION}:{QUESTION_METADATA_REPO_PATH}."
        )
    return actual


def question_metadata_provenance(qa_jsonl: str | Path) -> dict:
    """What the manifests record about the question metadata next to ``qa_jsonl``: file, pinned revision, SHA-256 on disk and whether it is the pinned one.
    ``{"file": None}`` if the file is not there (nothing is guessed)."""
    path = Path(qa_jsonl).with_name(QUESTION_METADATA_FILE)
    if not path.is_file():
        return {"file": None}
    digest = sha256_file(path)
    return {"file": str(path), "repo": CHEMQUESTS_REPO_ID, "repo_path": QUESTION_METADATA_REPO_PATH, "revision": QUESTION_METADATA_REVISION,
            "sha256": digest, "matches_pin": digest == QUESTION_METADATA_SHA256}

"""B10: the ChemQuests release is pinned everywhere and qa.jsonl is hash-checked."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_lib.dataset_pin import (  # noqa: E402
    CHEMQUESTS_QA_SHA256,
    CHEMQUESTS_REVISION,
    DatasetIntegrityError,
    verify_qa_sha256,
)
from stages import download_chemquests  # noqa: E402

FULL_REV = "5ab7dc673653b800f368fe30851efbdb24385e0d"
FULL_SHA = "e66ced49821787f7448b688a261ca0bb65c5cfe0f9961753316aec52c6b20e8d"


def test_constants_are_the_full_hashes():
    assert CHEMQUESTS_REVISION == FULL_REV
    assert CHEMQUESTS_QA_SHA256 == FULL_SHA


def test_pinned_everywhere():
    from pipeline_lib.my_tasks import ChemQuest

    assert ChemQuest.metadata.dataset["revision"] == FULL_REV
    for rel, key in [
        ("configs/stages/stage3.yaml", None),
        ("configs/download_chemquests_config.yaml", None),
        ("configs/main_config.yaml", "stage3"),
    ]:
        cfg = yaml.safe_load((ROOT / rel).read_text())
        if key:
            cfg = cfg[key]
        assert cfg["revision"] == FULL_REV, rel
        assert cfg["qa_sha256"] == FULL_SHA, rel


def test_verify_accepts_match_and_rejects_mismatch(tmp_path):
    f = tmp_path / "qa.jsonl"
    f.write_bytes(b'{"a": 1}\n')
    good = hashlib.sha256(b'{"a": 1}\n').hexdigest()
    assert verify_qa_sha256(f, good) == good
    with pytest.raises(DatasetIntegrityError, match="mismatch"):
        verify_qa_sha256(f, FULL_SHA)
    with pytest.raises(DatasetIntegrityError):
        verify_qa_sha256(tmp_path / "missing.jsonl", good)


def _fake_hub(monkeypatch, tmp_path, payload: bytes):
    src = tmp_path / "hub_cache.bin"
    src.write_bytes(payload)
    calls = []

    def fake(repo_id, filename, repo_type, revision, token):
        calls.append((repo_id, filename, repo_type, revision))
        return str(src)

    monkeypatch.setattr(download_chemquests, "hf_hub_download", fake)
    return calls


def test_download_checks_sha_and_uses_revision(tmp_path, monkeypatch):
    payload = b'{"question": "q"}\n'
    sha = hashlib.sha256(payload).hexdigest()
    calls = _fake_hub(monkeypatch, tmp_path, payload)
    out = tmp_path / "out"
    download_chemquests.download_files(
        "x/y", ["qa.jsonl"], str(out), revision=FULL_REV, expected_qa_sha256=sha
    )
    assert (out / "qa.jsonl").read_bytes() == payload
    assert calls == [("x/y", "qa.jsonl", "dataset", FULL_REV)]


def test_download_stops_on_wrong_sha(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, tmp_path, b"not the pinned file\n")
    with pytest.raises(DatasetIntegrityError):
        download_chemquests.download_files(
            "x/y", ["qa.jsonl"], str(tmp_path / "out"), revision=FULL_REV
        )  # default expected SHA = pinned release


def test_existing_wrong_file_is_not_trusted(tmp_path, monkeypatch):
    calls = _fake_hub(monkeypatch, tmp_path, b"unused")
    out = tmp_path / "out"
    out.mkdir()
    (out / "qa.jsonl").write_bytes(b"stale file from another revision\n")
    with pytest.raises(DatasetIntegrityError):
        download_chemquests.download_files("x/y", ["qa.jsonl"], str(out), revision=FULL_REV)
    assert calls == []  # skipped download (overwrite=False), but still verified


def test_download_main_rejects_moving_ref(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"repo_id": "x/y", "output_dir": str(tmp_path / "o"), "revision": "main"}))
    r = subprocess.run(
        [sys.executable, "-m", "stages.download_chemquests", "--config", str(cfg)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "pinned commit" in r.stderr


def test_stage5_stops_on_wrong_qa_sha(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({"question": "q", "corpus_id": "1", "content": "c", "start_index": 0, "end_index": 1}) + "\n")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"source_doc_id": "1", "chunk_index": 0, "chunk_id": "1:0", "text": "c"}) + "\n")
    cfg = tmp_path / "s5.yaml"
    cfg.write_text(yaml.safe_dump({
        "qa_jsonl_file": str(qa), "chunks_jsonl_file": str(chunks), "output_folder": str(tmp_path / "o"),
    }))
    r = subprocess.run(
        [sys.executable, "-m", "stages.make_mteb_task", "--config", str(cfg)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "SHA256 mismatch" in r.stderr
    assert not (tmp_path / "o" / "queries.jsonl").exists()

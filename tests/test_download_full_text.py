"""Stage 3: the pinned release has no full_text.jsonl; it is built from full_text/<id>.txt + metadata.jsonl."""
import hashlib
import json
from pathlib import Path

import pytest

from pipeline_lib.dataset_pin import DatasetIntegrityError
from stages import download_chemquests as D


def _hub(tmp_path, docs):
    t = tmp_path / "txt"; t.mkdir()
    for k, v in docs.items():
        (t / f"{k}.txt").write_bytes(v.encode("utf-8"))
    m = tmp_path / "metadata.jsonl"
    m.write_text("".join(json.dumps({"corpus_id": k, "title": "x"}) + "\n" for k in docs))
    return t, m


def test_rules_and_layout(tmp_path):
    docs = {"10": "a/b é\r\nline2\n", "2": "left \right\rx", "1": "plain"}      # CRLF, a lone CR, a trailing newline, "/"
    t, m = _hub(tmp_path, docs)
    rep = D.build_full_text_jsonl(t, m, tmp_path / "ft.jsonl")
    raw = (tmp_path / "ft.jsonl").read_bytes().decode()
    assert [json.loads(l)["corpus_id"] for l in raw.splitlines()] == ["1", "10", "2"]            # sorted as strings
    assert json.loads(raw.splitlines()[1])["text"] == "a/b é\nline2"
    assert json.loads(raw.splitlines()[2])["text"] == "left \night\nx"
    assert raw.splitlines()[1] == '{"corpus_id":"10","text":"a\\/b \\u00e9\\nline2"}'            # compact, ASCII, "\/"
    assert rep["changed_by_normalisation"] == {"crlf_or_cr_to_lf": ["10", "2"], "stripped_whitespace": ["10"]}


def test_ids_of_metadata_and_text_files_must_agree(tmp_path):
    t, m = _hub(tmp_path, {"1": "a", "2": "b"})
    (t / "3.txt").write_text("c")
    with pytest.raises(DatasetIntegrityError, match="disagree"):
        D.build_full_text_jsonl(t, m, tmp_path / "ft.jsonl")


def _qa(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_verify_spans(tmp_path):
    ft = tmp_path / "ft.jsonl"
    ft.write_text(json.dumps({"corpus_id": "98", "text": "hello world"}) + "\n")
    _qa(tmp_path / "qa.jsonl", [{"corpus_id": "098", "start_index": 6, "end_index": 11, "content": "world"}])
    assert D.verify_spans(ft, tmp_path / "qa.jsonl") == 1                                          # '098' == '98'
    _qa(tmp_path / "qa.jsonl", [{"corpus_id": "98", "start_index": 6, "end_index": 11, "content": "worlds"}])
    with pytest.raises(DatasetIntegrityError, match="do not match"):
        D.verify_spans(ft, tmp_path / "qa.jsonl")


def test_download_builds_full_text_checks_spans_and_reports(tmp_path, monkeypatch):
    qa = json.dumps({"question": "q", "corpus_id": "1", "content": "lain", "start_index": 1, "end_index": 5}) + "\n"
    files = {"metadata.jsonl": json.dumps({"corpus_id": "1"}) + "\n", "qa.jsonl": qa, "full_text/1.txt": "plain\n"}
    store = tmp_path / "hub"; store.mkdir(); (store / "full_text").mkdir()
    for k, v in files.items():
        (store / k).write_text(v)
    calls = []

    def fake_dl(repo_id, filename, repo_type, revision, token):
        calls.append(filename)
        return str(store / filename)

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type, revision, token=None):
            return ["qa.jsonl", "full_text/1.txt", "README.md"]

    monkeypatch.setattr(D, "hf_hub_download", fake_dl)
    monkeypatch.setattr(D, "HfApi", FakeApi)
    out = tmp_path / "out"
    D.download_files("x/y", ["metadata.jsonl", "full_text.jsonl", "qa.jsonl"], str(out), revision="a" * 40,
                     expected_qa_sha256=hashlib.sha256(qa.encode()).hexdigest())
    assert "full_text.jsonl" not in calls and "full_text/1.txt" in calls                           # never asks the hub for it
    assert json.loads((out / "full_text.jsonl").read_text())["text"] == "plain"
    rep = json.loads((out / "full_text_build.json").read_text())
    assert rep["n_documents"] == 1 and rep["revision"] == "a" * 40 and len(rep["full_text_jsonl_sha256"]) == 64
    # a qa.jsonl whose spans do not fit the texts stops stage 3
    bad = qa.replace('"lain"', '"xxxx"')
    (store / "qa.jsonl").write_text(bad)
    with pytest.raises(DatasetIntegrityError, match="do not match"):
        D.download_files("x/y", ["metadata.jsonl", "full_text.jsonl", "qa.jsonl"], str(tmp_path / "out2"), revision="a" * 40,
                         expected_qa_sha256=hashlib.sha256(bad.encode()).hexdigest())

"""Truncation log: counts texts over the model's max sequence length with the model's own tokenizer; records only."""
import json
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("transformers")
from transformers import AutoTokenizer

from pipeline_lib import truncation as T

REV = "86b5e0934494bd15c9632b12f734a8a67f723594"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("bert-base-uncased", revision=REV, use_fast=True)


def _task(tmp_path, texts, queries=("what is it",)):
    d = tmp_path / "task"; d.mkdir(exist_ok=True)
    (d / "corpus.jsonl").write_text("".join(json.dumps({"id": f"{i}:0", "title": f"{i} - Chunk 0", "text": t}) + "\n" for i, t in enumerate(texts)))
    (d / "queries.jsonl").write_text("".join(json.dumps({"id": str(i), "text": q}) + "\n" for i, q in enumerate(queries)))
    return d


def test_counts_include_special_tokens_title_and_prompt(tmp_path, tok):
    d = _task(tmp_path, ["word " * 10, "word " * 20])
    # "N - Chunk 0 " is 5 tokens ("0", "-", "chunk", "0" -> 4) + 10 words + [CLS][SEP]
    base = T.truncation_report(d, tok, 100)
    assert base["passages"]["n_truncated"] == 0
    n1 = len(tok("0 - Chunk 0 " + "word " * 10)["input_ids"])
    assert base["passages"]["max_tokens"] == len(tok("1 - Chunk 0 " + "word " * 20)["input_ids"])
    rep = T.truncation_report(d, tok, n1)          # exactly at the limit: not truncated; the longer text is
    assert rep["passages"]["n_truncated"] == 1 and rep["passages"]["pct_truncated"] == 50.0
    assert rep["passages"]["mean_tokens_lost_when_truncated"] == base["passages"]["max_tokens"] - n1
    prompted = T.truncation_report(d, tok, n1, prompts={"query": "query: ", "passage": "passage: "})
    assert prompted["passages"]["n_truncated"] == 2 and prompted["prompts_included"] is True   # "passage: " adds tokens
    assert prompted["queries"]["max_tokens"] > base["queries"]["max_tokens"]


def test_run_eval_writes_the_report_without_changing_anything(tmp_path, tok, monkeypatch):
    pytest.importorskip("mteb"); pytest.importorskip("sentence_transformers")
    from stages import run_eval
    d = _task(tmp_path, ["word " * 30, "word " * 3])
    (d / "qrels_x.jsonl").write_text(json.dumps({"query_id": "0", "doc_id": "0:0", "relevance": 1}) + "\n")

    class FakeBase:
        tokenizer = tok
        max_seq_length = 20

    class FakeMTEB:
        def __init__(self, tasks): pass
        def run(self, model, **kw):
            (Path(kw["output_folder"]) / "ChemQuest_default_predictions.json").write_text(json.dumps({"0": {"0:0": 1.0}}))
            return []

    cfg = {"models": [{"name": "m/x", "revision": "r"}], "output_root": str(tmp_path / "res"),
           "tasks": [{"class": "x.Y", "kwargs": {"data_dir": str(d), "qrels_file": "qrels_x.jsonl"}}]}
    p = tmp_path / "s6.yaml"; p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(run_eval, "MTEB", FakeMTEB)
    monkeypatch.setattr(run_eval, "SentenceTransformer", lambda *a, **k: FakeBase())
    monkeypatch.setattr(run_eval, "resolve_tasks", lambda specs, mods: [object()])
    monkeypatch.setattr(sys, "argv", ["run_eval", "--config", str(p)])
    run_eval.main()
    rep = json.loads((tmp_path / "res" / "m__x_r" / "truncation_report.json").read_text())
    assert rep["max_seq_length"] == 20 and rep["passages"]["n_truncated"] == 1 and rep["passages"]["n_texts"] == 2
    m = json.loads((tmp_path / "res" / "m__x_r" / "run_manifest.json").read_text())
    assert m["truncation"]["passages"]["n_truncated"] == 1

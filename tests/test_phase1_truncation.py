"""B3: Phase 1 truncation (external tasks): the count, the stage-1 hook, the collector in make report, the offline tool."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pipeline_lib import model_info, truncation as T
from tools import phase1_truncation as P1


class WordTok:
    """One token per word, plus [CLS] and [SEP] (add_special_tokens)."""
    model_max_length = 16

    def __call__(self, texts, add_special_tokens=True, truncation=False):
        return {"input_ids": [[0] * (len(t.split()) + (2 if add_special_tokens else 0)) for t in texts]}


def fake_task(name="ChemNQRetrieval", splits=("test",), n_long=2):
    corpus = {f"d{i}": {"title": "T", "text": "w " * (20 if i < n_long else 3)} for i in range(5)}
    queries = {"q0": "what is it", "q1": "unjudged query", "q2": "x " * 30}
    qrels = {"q0": {"d0": 1}, "q2": {"d1": 1}}
    return SimpleNamespace(metadata=SimpleNamespace(name=name, eval_splits=list(splits)), is_multilingual=False, data_loaded=True,
                           corpus={s: corpus for s in splits}, queries={s: queries for s in splits}, relevant_docs={s: qrels for s in splits})


def test_mteb_passage_string_matches_what_mteb_encodes():
    assert T.mteb_passage_string({"title": "A", "text": "b"}) == "A b"
    assert T.mteb_passage_string({"title": "", "text": "b"}) == "b"                    # an empty title: MTEB's strip leaves the text
    assert T.mteb_passage_string({"text": "b "}) == "b"
    assert T.mteb_passage_string("plain ") == "plain"


def test_task_texts_only_judged_queries():
    passages, queries = T.task_texts(fake_task(), "test")
    assert len(passages) == 5 and passages[0].startswith("T w") and queries == ["what is it", "x " * 30]


def test_phase1_report_counts_with_title_special_tokens_and_prompt():
    rep = T.phase1_report(fake_task(), WordTok(), 16, None, "m/x", "rev")
    p = rep["splits"]["test"]["passages"]
    # long passage: "T" + 20 words + 2 special = 23 > 16 (2 of 5); short: 1 + 3 + 2 = 6
    assert (p["n_texts"], p["n_truncated"], p["max_tokens"]) == (5, 2, 23) and rep["reported_split"] == "test"
    q = rep["splits"]["test"]["queries"]
    assert q["n_texts"] == 2 and q["n_truncated"] == 1 and q["max_tokens"] == 32
    prompted = T.phase1_report(fake_task(), WordTok(), 16, {"query": "query: ", "passage": "passage: "}, "m/x", "rev")
    assert prompted["splits"]["test"]["passages"]["max_tokens"] == 23 + 1 and prompted["prompts_included"] is True


def test_splits_with_one_corpus_are_counted_once_and_test_is_reported():
    rep = T.phase1_report(fake_task("ChemHotpotQARetrieval", ("train", "dev", "test")), WordTok(), 16, None, "m/x", "rev")
    assert rep["reported_split"] == "test"
    assert {rep["splits"][s]["passages"]["n_truncated"] for s in rep["splits"]} == {2}
    rows = T.phase1_rows(rep)
    assert [r["split"] for r in rows] == ["train", "dev", "test"] and [r["reported"] for r in rows] == [False, False, True]
    assert set(rows[0]) == set(T.PHASE1_COLUMNS)


def test_write_reports_and_collect(tmp_path):
    out = tmp_path / "root" / "m__x_rev"
    out.mkdir(parents=True)
    paths = T.write_phase1_reports([fake_task(), fake_task("ChemHotpotQARetrieval", ("test",))], WordTok(), 16, None, "m/x", "rev", out)
    assert [p.name for p in paths] == ["truncation_report_ChemNQRetrieval.json", "truncation_report_ChemHotpotQARetrieval.json"]
    rows = T.collect_phase1([tmp_path / "root", tmp_path / "missing"])
    assert {(r["model"], r["task"], r["split"]) for r in rows} == {("m/x", "ChemNQRetrieval", "test"), ("m/x", "ChemHotpotQARetrieval", "test")}


def test_a_failing_report_never_stops_the_run(tmp_path, capsys):
    bad = fake_task()
    bad.corpus = {}
    T.write_phase1_reports([bad], WordTok(), 16, None, "m/x", "rev", tmp_path)
    assert (tmp_path / "truncation_report_ChemNQRetrieval.error.txt").exists() and "[WARN]" in capsys.readouterr().out
    assert not list(tmp_path.glob("truncation_report_*.json"))


def test_model_screening_writes_the_reports_next_to_the_results(tmp_path, monkeypatch):
    pytest.importorskip("mteb"); pytest.importorskip("sentence_transformers")
    from stages import model_screening as MS

    class FakeBase:
        tokenizer = WordTok()
        max_seq_length = 16
        _modules = {}

        def get_sentence_embedding_dimension(self):
            return 8

    class FakeMTEB:
        def __init__(self, tasks): pass
        def run(self, model, **kw): return []

    cfg = {"models": [{"name": "m/x", "revision": "r"}], "tasks": ["ChemNQRetrieval"], "output_root": str(tmp_path / "res"), "seed": 1,
           "use_prompts": True, "model_prompts": {"m/x": {"query": "query: ", "passage": "passage: ", "card": "t"}}}
    p = tmp_path / "s1.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(MS, "MTEB", FakeMTEB)
    monkeypatch.setattr(MS, "SentenceTransformer", lambda *a, **k: FakeBase())
    monkeypatch.setattr(MS, "resolve_tasks", lambda specs, mods: [fake_task()])
    monkeypatch.setattr(sys, "argv", ["model_screening", "--config", str(p)])
    MS.main()
    rep = json.loads((tmp_path / "res" / "m__x_r" / "truncation_report_ChemNQRetrieval.json").read_text())
    assert rep["prompts_included"] is True and rep["max_seq_length"] == 16 and rep["splits"]["test"]["passages"]["n_truncated"] == 2


def test_make_report_writes_table_truncation_phase1(tmp_path):
    from tools import make_report as R
    ext = tmp_path / "ext"
    (ext / "m__x_r").mkdir(parents=True)
    T.write_phase1_reports([fake_task()], WordTok(), 16, None, "m/x", "r", ext / "m__x_r")
    root = tmp_path / "art"
    (root / "tests").mkdir(parents=True)
    qa = root / "qa.jsonl"
    qa.write_text(json.dumps({"question": "q", "corpus_id": "0", "content": "q", "start_index": 0, "end_index": 1}) + "\n")
    out = tmp_path / "out"
    from test_make_report import _run                                                  # build_report needs at least one run
    _run(root / "tests", "bm25_fixed_token_c128_o0", "bm25", "fixed_token", 128, {"0": (0.5, 0.5)})
    m = R.build_report(root, out, qa, resamples=5, overlap=False, verbose=False, external_roots=[ext])
    rows = list(csv.DictReader(open(out / "table_truncation_phase1.csv")))
    assert rows[0]["model"] == "m/x" and float(rows[0]["pct_passages_truncated"]) == 40.0 and m["steps"]["phase1_truncation_rows"] == 1


# ---- the offline tool for the published runs --------------------------------------------------------------------
def _fake_cache(tmp_path):
    import pandas as pd
    root = tmp_path / "hub" / "datasets--BASF-AI--ChemNQRetrieval"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("abc")
    snap = root / "snapshots" / "abc"
    for sub in ("corpus", "queries", "data"):
        (snap / sub).mkdir(parents=True)
    pd.DataFrame({"_id": ["d0", "d1"], "title": ["T", ""], "text": ["w " * 20, "w w"]}).to_parquet(snap / "corpus" / "corpus-00000-of-00001.parquet")
    pd.DataFrame({"_id": ["q0", "q1"], "text": ["what", "other"]}).to_parquet(snap / "queries" / "queries-00000-of-00001.parquet")
    pd.DataFrame({"query-id": ["q0"], "corpus-id": ["d0"], "score": [1]}).to_parquet(snap / "data" / "test-00000-of-00001.parquet")
    return tmp_path / "hub"


def test_cached_parquet_task_loader(tmp_path):
    pytest.importorskip("pyarrow")
    task = P1.load_cached_task("ChemNQRetrieval", cache_dir=str(_fake_cache(tmp_path)))
    assert task.metadata.eval_splits == ["test"] and list(task.corpus["test"]) == ["d0", "d1"]
    passages, queries = T.task_texts(task, "test")
    assert passages[0].startswith("T w") and passages[1] == "w w" and queries == ["what"]
    with pytest.raises(FileNotFoundError):
        P1.load_cached_task("NotCached", cache_dir=str(tmp_path / "hub"))


def test_tool_skips_uncached_models_and_unloadable_tasks(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"hidden_size": 8, "max_position_embeddings": 16}))
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r, c=None: snap if n == "a/cached" else None)
    tasks = {"ChemNQRetrieval": fake_task()}

    def loader(name):
        if name not in tasks:
            raise FileNotFoundError("not cached")
        return tasks[name]

    models = [{"name": "a/cached", "revision": "r"}, {"name": "b/uncached", "revision": "r"}]
    res = P1.run(models, ["ChemNQRetrieval", "ChemHotpotQARetrieval"], model_prompts={"a/cached": {"query": "", "passage": ""}}, use_prompts=False,
                 loader=loader, tokenizer_loader=lambda s: WordTok())
    assert {r["model"] for r in res["rows"]} == {"a/cached"} and res["rows"][0]["max_seq_length"] == 16 and res["rows"][0]["n_passages_truncated"] == 2
    reasons = {(s["model"], s["task"]): s["reason"] for s in res["skipped"]}
    assert "not in the local cache" in reasons[("b/uncached", "")] and "not loadable" in reasons[("", "ChemHotpotQARetrieval")]


def test_tool_prompts_option(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"hidden_size": 8, "max_position_embeddings": 16}))
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r, c=None: snap)
    kw = dict(model_prompts={"a/m": {"query": "q: ", "passage": "passage: ", "card": "x"}}, loader=lambda n: fake_task(), tokenizer_loader=lambda s: WordTok())
    none = P1.run([{"name": "a/m", "revision": "r"}], ["ChemNQRetrieval"], use_prompts=False, **kw)["rows"][0]
    cfg = P1.run([{"name": "a/m", "revision": "r"}], ["ChemNQRetrieval"], use_prompts=True, **kw)["rows"][0]
    assert none["prompts_included"] is False and cfg["prompts_included"] is True and cfg["max_passage_tokens"] == none["max_passage_tokens"] + 1


def test_tool_main_writes_both_files(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"hidden_size": 8, "max_position_embeddings": 16}))
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r, c=None: snap if n == "a/m" else None)
    monkeypatch.setattr(P1, "_default_loader", lambda name: fake_task(name))
    monkeypatch.setattr(P1, "_default_tokenizer", lambda s: WordTok())
    cfg = {"params": {"use_prompts": False, "model_prompts": {}}, "stage1": {"models": [{"name": "a/m", "revision": "r"}, {"name": "z/n", "revision": "r"}],
                                                                           "tasks": ["ChemNQRetrieval"]}}
    c = tmp_path / "main.yaml"
    c.write_text(yaml.safe_dump(cfg))
    assert P1.main(["--main-config", str(c), "--out-dir", str(tmp_path / "out")]) == 0
    assert list(csv.DictReader(open(tmp_path / "out" / "table_truncation_phase1.csv")))[0]["model"] == "a/m"
    assert list(csv.DictReader(open(tmp_path / "out" / "table_truncation_phase1_skipped.csv")))[0]["model"] == "z/n"


def test_reported_split_is_the_configured_one_and_a_missing_split_raises():
    from pipeline_lib.phase1_split import Phase1SplitError
    task = fake_task("ChemHotpotQARetrieval", ("train", "dev", "test"))
    rep = T.phase1_report(task, WordTok(), 16, None, "m/x", "rev", eval_split={"ChemHotpotQARetrieval": "dev"})
    assert rep["reported_split"] == "dev" and [r["reported"] for r in T.phase1_rows(rep)] == [False, True, False]
    with pytest.raises(Phase1SplitError):
        T.phase1_report(fake_task("ChemHotpotQARetrieval", ("train", "dev")), WordTok(), 16, None, "m/x", "rev", eval_split={"ChemHotpotQARetrieval": "test"})

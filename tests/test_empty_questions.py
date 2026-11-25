"""B2: questions without a relevant chunk are kept and score 0 (never dropped)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("mteb")
pytest.importorskip("pytrec_eval")

from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator  # noqa: E402

from pipeline_lib import my_tasks  # noqa: E402
from pipeline_lib.my_tasks import ChemQuest  # noqa: E402


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _task_dir(tmp_path, qrels, queries=("q zero", "q one", "q two")):
    d = tmp_path / "mteb_task"
    d.mkdir()
    _write_jsonl(d / "corpus.jsonl", [{"id": f"c{i}", "title": "t", "text": f"chunk {i}"} for i in range(3)])
    _write_jsonl(d / "queries.jsonl", [{"id": str(i), "title": "t", "text": t} for i, t in enumerate(queries)])
    _write_jsonl(d / "qrels_lexical.jsonl", qrels)
    return d


def _load(d):
    t = ChemQuest(data_dir=str(d), qrels_file="qrels_lexical.jsonl")
    t.load_data()
    return t


def _perfect_run(queries, doc_for):
    # every query ranks its "own" chunk first, then the others
    return {q: {**{"c0": 1.0, "c1": 0.5, "c2": 0.25}, doc_for[q]: 3.0} for q in queries}


def test_relevance_zero_query_is_evaluated_and_scores_zero(tmp_path):
    d = _task_dir(tmp_path, [
        {"query_id": "0", "doc_id": "c0", "relevance": 1},
        {"query_id": "1", "doc_id": "c1", "relevance": 1},
        {"query_id": "2", "doc_id": "c2", "relevance": 0, "score": 3.0},  # score must be ignored
    ])
    t = _load(d)
    qrels = t.relevant_docs["test"]
    assert set(t.queries["test"]) == {"0", "1", "2"}
    assert qrels["2"] == {"c2": 0}

    run = _perfect_run(["0", "1", "2"], {"0": "c0", "1": "c1", "2": "c2"})
    ndcg, _map, recall, _p, _n = RetrievalEvaluator.evaluate(qrels, run, [1, 10])
    # q2 only has a relevance-0 row: it is evaluated, scores 0, and the mean is over all 3 queries
    assert ndcg["NDCG@10"] == pytest.approx(2 / 3, abs=1e-4)
    assert recall["Recall@10"] == pytest.approx(2 / 3, abs=1e-4)

    # per query: dropping q2 (old behaviour) would inflate the mean to 1.0
    only_pos = {k: v for k, v in qrels.items() if k != "2"}
    ndcg_old, *_ = RetrievalEvaluator.evaluate(only_pos, {k: run[k] for k in only_pos}, [10])
    assert ndcg_old["NDCG@10"] == pytest.approx(1.0)


def test_query_without_any_qrels_row_is_kept(tmp_path):
    d = _task_dir(tmp_path, [
        {"query_id": "0", "doc_id": "c0", "relevance": 1},
        {"query_id": "1", "doc_id": "c1", "relevance": 1},
    ])  # query 2 has no row at all
    t = _load(d)
    assert set(t.queries["test"]) == {"0", "1", "2"}
    assert set(t.relevant_docs["test"]["2"].values()) == {0}
    run = _perfect_run(["0", "1", "2"], {"0": "c0", "1": "c1", "2": "c2"})
    ndcg, *_ = RetrievalEvaluator.evaluate(t.relevant_docs["test"], run, [10])
    assert ndcg["NDCG@10"] == pytest.approx(2 / 3, abs=1e-4)


def test_empty_question_text_is_kept(tmp_path):
    d = _task_dir(tmp_path, [
        {"query_id": "0", "doc_id": "c0", "relevance": 1},
        {"query_id": "1", "doc_id": "c1", "relevance": 0},
        {"query_id": "2", "doc_id": "c2", "relevance": 0},
    ], queries=("a", "", "c"))
    t = _load(d)
    assert t.queries["test"]["1"] == ""


def test_logs_number_of_queries_without_positive(tmp_path, caplog):
    d = _task_dir(tmp_path, [
        {"query_id": "0", "doc_id": "c0", "relevance": 1},
        {"query_id": "1", "doc_id": "c1", "relevance": 0},
    ])
    with caplog.at_level("INFO", logger=my_tasks.log.name):
        _load(d)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "2 without a positive label" in msg  # q1 (relevance 0) and q2 (no row)


def test_labels_must_be_binary_and_come_from_relevance(tmp_path):
    d = _task_dir(tmp_path, [{"query_id": "0", "doc_id": "c0", "relevance": 2}])
    with pytest.raises(ValueError, match="0 or 1"):
        _load(d)
    d2 = tmp_path / "second"
    d2.mkdir()
    d = _task_dir(d2, [{"query_id": "0", "doc_id": "c0", "score": 1.0}])  # no `relevance`
    with pytest.raises(ValueError, match="relevance"):
        _load(d)


def test_all_zero_dataset_is_rejected(tmp_path):
    d = _task_dir(tmp_path, [{"query_id": "0", "doc_id": "c0", "relevance": 0}])
    with pytest.raises(ValueError, match="No query has a positive label"):
        _load(d)


# ---------------- HF branch (my_tasks.py:298 used to read `score`) ----------------
class _FakeSplit(list):
    column_names = ["query-id", "corpus-id", "relevance", "score"]


def test_hf_branch_reads_relevance_and_keeps_all_queries(monkeypatch):
    test_rows = _FakeSplit([
        {"query-id": "0", "corpus-id": "c0", "relevance": 1, "score": 5},
        {"query-id": "1", "corpus-id": "c1", "relevance": 0, "score": 9},  # score 9 must not matter
    ])

    def fake_load_dataset(path, name=None, revision=None, split=None):
        assert revision == "5ab7dc673653b800f368fe30851efbdb24385e0d"
        if name is None:
            return {"test": test_rows}
        if name == "corpus":
            return [{"_id": f"c{i}", "title": "t", "text": "x"} for i in range(3)]
        if name == "queries":
            return [{"_id": str(i), "text": "" if i == 2 else f"q{i}"} for i in range(3)]
        raise AssertionError(name)

    monkeypatch.setattr(my_tasks, "load_dataset", fake_load_dataset)
    t = ChemQuest()
    t.load_data()
    assert set(t.queries["test"]) == {"0", "1", "2"}
    assert t.relevant_docs["test"]["0"] == {"c0": 1}
    assert t.relevant_docs["test"]["1"] == {"c1": 0}
    assert set(t.relevant_docs["test"]["2"].values()) == {0}


# ---------------- stage 5 ----------------
def _run_stage5(tmp_path, qa_rows, chunk_rows):
    qa = tmp_path / "qa.jsonl"
    _write_jsonl(qa, qa_rows)
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(chunks, chunk_rows)
    out = tmp_path / "task"
    cfg = tmp_path / "s5.yaml"
    cfg.write_text(yaml.safe_dump({
        "qa_jsonl_file": str(qa), "chunks_jsonl_file": str(chunks), "output_folder": str(out),
        "qa_sha256": hashlib.sha256(qa.read_bytes()).hexdigest(),
        "search_scope": "single", "min_score_threshold": 0.95, "max_results_per_query": 5,
    }))
    r = subprocess.run([sys.executable, "-m", "stages.make_mteb_task", "--config", str(cfg)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rd = lambda n: [json.loads(l) for l in (out / n).read_text().splitlines() if l.strip()]
    return r.stdout, rd("queries.jsonl"), rd("qrels_lexical.jsonl"), out


def test_stage5_keeps_question_without_label(tmp_path):
    chunk_rows = [
        {"source_doc_id": "1", "chunk_index": 0, "chunk_id": "1:0", "text": "alpha beta gamma"},
        {"source_doc_id": "1", "chunk_index": 1, "chunk_id": "1:1", "text": "delta epsilon zeta"},
    ]
    qa_rows = [
        {"question": "what is alpha", "corpus_id": "1", "content": "alpha beta", "start_index": 0, "end_index": 10},
        {"question": "", "corpus_id": "1", "content": "", "start_index": -1, "end_index": -1},  # nothing to match
        {"question": "what is zeta", "corpus_id": "1", "content": "delta epsilon", "start_index": 17, "end_index": 30},
    ]
    stdout, queries, qrels, out = _run_stage5(tmp_path, qa_rows, chunk_rows)
    assert [q["id"] for q in queries] == ["0", "1", "2"]          # nothing skipped
    zero = [r for r in qrels if r["query_id"] == "1"]
    assert len(zero) == 1 and zero[0]["relevance"] == 0 and zero[0]["doc_id"] == "1:0"
    assert {r["query_id"] for r in qrels if r["relevance"] == 1} == {"0", "2"}
    assert "without a positive label (kept, relevance-0 row): 1" in stdout

    # and the loader + evaluator keep the empty question in the mean
    t = ChemQuest(data_dir=str(out), qrels_file="qrels_lexical.jsonl")
    t.load_data()
    assert set(t.queries["test"]) == {"0", "1", "2"}
    assert t.relevant_docs["test"]["1"] == {"1:0": 0}

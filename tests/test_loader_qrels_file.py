"""Loader: the qrels file named in the config is the one that is read; all queries are kept; labels are 0/1 from `relevance`."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("mteb")
pytest.importorskip("pytrec_eval")

from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator  # noqa: E402

from pipeline_lib.my_tasks import ChemQuest  # noqa: E402


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _dir(tmp_path, **files):
    d = tmp_path / "task"
    d.mkdir()
    _write(d / "corpus.jsonl", [{"id": f"c{i}", "title": "t", "text": f"chunk {i}"} for i in range(3)])
    _write(d / "queries.jsonl", [{"id": str(i), "title": "t", "text": f"q{i}"} for i in range(3)])
    for name, rows in files.items():
        _write(d / name.replace("__", "."), rows)
    return d


def _load(d, qrels_file):
    t = ChemQuest(data_dir=str(d), qrels_file=qrels_file)
    t.load_data()
    return t


SPAN = [{"query_id": "0", "doc_id": "c0", "relevance": 1},
        {"query_id": "1", "doc_id": "c1", "relevance": 1}, {"query_id": "1", "doc_id": "c2", "relevance": 1},
        {"query_id": "2", "doc_id": "c0", "relevance": 0}]
LEX = [{"query_id": "0", "doc_id": "c2", "relevance": 1, "score": 2.0},
       {"query_id": "1", "doc_id": "c2", "relevance": 1, "score": 1.0},
       {"query_id": "2", "doc_id": "c1", "relevance": 1, "score": 1.0}]


def test_the_named_file_is_read_and_others_are_ignored(tmp_path):
    d = _dir(tmp_path, qrels_span_c080__jsonl=SPAN, qrels_lexical__jsonl=LEX, qrels__jsonl=LEX)
    span = _load(d, "qrels_span_c080.jsonl").relevant_docs["test"]
    lex = _load(d, "qrels_lexical.jsonl").relevant_docs["test"]
    assert span == {"0": {"c0": 1}, "1": {"c1": 1, "c2": 1}, "2": {"c0": 0}}
    assert lex == {"0": {"c2": 1}, "1": {"c2": 1}, "2": {"c1": 1}}


def test_without_a_name_there_is_no_default_file(tmp_path):
    """No generic qrels.jsonl is read or written any more: the label file is always named (labeling.qrels_file)."""
    d = _dir(tmp_path, qrels__jsonl=LEX, qrels_span_c080__jsonl=SPAN)
    with pytest.raises(FileNotFoundError, match="there is no default file"):
        _load(d, None)


def test_missing_named_file_raises_and_does_not_fall_back(tmp_path):
    d = _dir(tmp_path, qrels__jsonl=LEX)
    with pytest.raises(FileNotFoundError, match="qrels_span_c080.jsonl"):
        _load(d, "qrels_span_c080.jsonl")


def test_query_with_only_a_relevance_0_row_is_kept_and_scores_0(tmp_path):
    d = _dir(tmp_path, qrels_span_c080__jsonl=SPAN)
    t = _load(d, "qrels_span_c080.jsonl")
    assert set(t.queries["test"]) == {"0", "1", "2"}
    run = {"0": {"c0": 3.0, "c1": 1.0, "c2": 0.5}, "1": {"c1": 3.0, "c2": 2.0, "c0": 1.0}, "2": {"c0": 3.0, "c1": 1.0, "c2": 0.5}}
    ndcg, _, recall, _, _ = RetrievalEvaluator.evaluate(t.relevant_docs["test"], run, [10])
    # queries 0 and 1 are perfect, query 2 (only a relevance-0 row) scores 0: the mean is over all 3
    assert ndcg["NDCG@10"] == pytest.approx(2 / 3, abs=1e-5)
    assert recall["Recall@10"] == pytest.approx(2 / 3, abs=1e-5)


@pytest.mark.parametrize("bad", [2, 0.5, -1, 3.0, "x"])
def test_non_binary_labels_raise(tmp_path, bad):
    rows = [dict(r) for r in SPAN]
    rows[0]["relevance"] = bad
    d = _dir(tmp_path, qrels_span_c080__jsonl=rows)
    with pytest.raises(ValueError):
        _load(d, "qrels_span_c080.jsonl")


def test_relevance_is_read_not_score(tmp_path):
    rows = [{"query_id": "0", "doc_id": "c0", "relevance": 1, "score": 7.0},
            {"query_id": "1", "doc_id": "c1", "relevance": 0, "score": 9.0},
            {"query_id": "2", "doc_id": "c2", "relevance": 1}]
    d = _dir(tmp_path, q__jsonl=rows)
    assert _load(d, "q.jsonl").relevant_docs["test"] == {"0": {"c0": 1}, "1": {"c1": 0}, "2": {"c2": 1}}
    (tmp_path / "x").mkdir()
    d2 = _dir(tmp_path / "x", q__jsonl=[{"query_id": "0", "doc_id": "c0", "score": 1.0}])
    with pytest.raises(ValueError, match="relevance"):
        _load(d2, "q.jsonl")


def test_a_labelled_chunk_missing_from_the_corpus_raises(tmp_path):
    rows = SPAN + [{"query_id": "0", "doc_id": "not_in_corpus", "relevance": 1}]
    d = _dir(tmp_path, q__jsonl=rows)
    with pytest.raises(ValueError, match="not in corpus"):
        _load(d, "q.jsonl")


def test_query_without_any_row_is_kept(tmp_path):
    d = _dir(tmp_path, q__jsonl=[{"query_id": "0", "doc_id": "c0", "relevance": 1}])
    t = _load(d, "q.jsonl")
    assert set(t.queries["test"]) == {"0", "1", "2"}
    assert all(not any(v for v in t.relevant_docs["test"][q].values()) for q in ("1", "2"))

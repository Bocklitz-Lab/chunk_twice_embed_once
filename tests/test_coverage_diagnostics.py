"""tools/coverage_diagnostics.py on the synthetic documents of test_make_span_qrels."""
import csv
import json
from pathlib import Path

import pytest

from tests.test_make_span_qrels import QUESTIONS, _build, _run
from tools import coverage_diagnostics as D
from tools import metrics as M


@pytest.fixture()
def task(tmp_path):
    d = _build(tmp_path)
    _run(d, "--cover-level", "0.9")
    out = d / "out"
    # lexical labels written by hand: q0 two chunks, q1 none, the others one chunk of their document
    rows = [{"query_id": "0", "doc_id": "0:1", "relevance": 1}, {"query_id": "0", "doc_id": "0:0", "relevance": 1},
            {"query_id": "1", "doc_id": "0:0", "relevance": 0}]
    for q in range(2, 8):
        rows.append({"query_id": str(q), "doc_id": "0:0" if QUESTIONS[q][0] == "0" else f"{QUESTIONS[q][0]}:0", "relevance": 1})
    (out / "qrels_lexical.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_config_diagnostics(task):
    out = task / "out"
    row = D.diagnose_config(out, task / "chunks.jsonl", span_qrels="qrels_span_c090.jsonl", out_dir=out / "diag")
    assert row["n_questions"] == 8 and row["n_chunks"] == 8 and row["n_docs"] == 3
    assert row["mean_chunk_chars"] == pytest.approx(
        sum(len(json.loads(l)["text"]) for l in open(task / "chunks.jsonl")) / 8)
    assert row["pct_span_inside_one_chunk"] == pytest.approx(50.0)        # q0, q4, q5, q6
    assert row["n_zero_relevant_span"] == 1                              # q3: no chunk overlaps the span
    assert row["n_zero_relevant_lexical"] == 1                           # q1 has only a relevance-0 row
    assert row["mean_relevant_lexical"] == pytest.approx(8 / 8)   # q0 has two, q1 none, q2..q7 one each
    span = M.load_qrels(out / "qrels_span_c090.jsonl")
    assert row["mean_relevant_span"] == pytest.approx(sum(sum(v > 0 for v in r.values()) for r in span.values()) / 8)
    csv_rows = list(csv.DictReader(open(out / "diag" / "no_relevant_chunk.csv")))
    assert {(r["label_set"], r["query_id"]) for r in csv_rows} == {("lexical", "1"), ("span", "3")}
    assert (out / "diag" / "coverage_diagnostics.csv").exists()


def test_span_coverage_at_k_uses_the_union_of_the_top_k_chunks(task):
    out = task / "out"
    run = {"0": {"0:1": 3.0, "0:0": 2.0}, "1": {"0:0": 3.0, "0:1": 2.0, "0:2": 1.0}, "2": {"1:0": 2.0, "1:1": 1.0},
           "3": {"1:0": 1.0}, "4": {"2:0": 2.0, "2:1": 1.0}, "5": {"2:0": 1.0, "2:1": 0.5}, "6": {"0:0": 1.0}, "7": {"0:1": 2.0, "0:2": 1.0}}
    cc = D.corpus_chunk_chars(out / "corpus.jsonl")
    cov = D.span_coverage(run, out / "overlap_table.jsonl", M.load_query_ids(out / "queries.jsonl"), (1, 5, 10), cc)
    assert cov["0"]["span_coverage_at_1"] == 1.0                           # span inside chunk 0:1
    assert cov["1"]["span_coverage_at_1"] == pytest.approx(20 / 60)       # chunk 0:0 holds 20 of 60 characters
    assert cov["1"]["span_coverage_at_5"] == 1.0                           # union with 0:1 covers it
    assert cov["2"]["span_coverage_at_5"] == pytest.approx(20 / 50)       # the 30 characters in the gap stay uncovered
    assert cov["3"]["span_coverage_at_10"] == 0.0                          # no chunk touches the span
    assert cov["4"]["span_coverage_at_1"] == 1.0
    assert cov["7"]["span_coverage_at_1"] == pytest.approx(0.5) and cov["7"]["span_coverage_at_5"] == 1.0
    assert cov["0"]["retrieved_chars_at_1"] == 100.0
    assert cov["0"]["span_char_precision_at_1"] == pytest.approx(40 / 100)
    # overlapping chunks are not double counted: 2:0 [0,120) and 2:1 [80,200) for the span [90,110)
    assert cov["4"]["span_coverage_at_5"] == 1.0
    m = D.mean_of(cov)
    assert 0 < m["span_coverage_at_1"] < m["span_coverage_at_5"] <= 1.0


def test_union_length():
    assert D._union_len([(0, 10), (5, 15), (20, 30)]) == 25
    assert D._union_len([]) == 0

"""tools/token_budget.py: the fixed retrieved-token budget (union of overlapping chunks, cut of the last chunk, hit at >= c)."""
import json
import re
from fractions import Fraction

import numpy as np
import pytest

from analysis_world import build_world
from tools import analysis_data as AD, token_budget as TB

TEXT = "aa bb cc dd ee ff gg hh"              # tokens (2 characters each) start at 0, 3, 6, 9, 12, 15, 18, 21


def ws_tokens(text):
    ms = [m for m in re.finditer(r"\S+", text)]
    return [m.start() for m in ms], [m.end() for m in ms]


DOCS = {"1": ws_tokens(TEXT), "2": ws_tokens(TEXT)}
POS = {"A": ("1", 0, 8), "B": ("1", 6, 15), "C": ("2", 0, 8), "D": ("1", 18, 23)}      # A = aa bb cc, B overlaps A at cc: dd ee, C is another document


def walk(ranking, span, budgets):
    return TB.walk_budgets(ranking, POS, DOCS, span, budgets)


def test_interval_helpers():
    assert TB.new_pieces([(0, 8)], 6, 15) == [(8, 15)]
    assert TB.new_pieces([(0, 4), (6, 8)], 0, 10) == [(4, 6), (8, 10)]
    assert TB.new_pieces([], 3, 5) == [(3, 5)] and TB.new_pieces([(0, 10)], 2, 5) == []
    iv = [(0, 4), (8, 10)]
    TB.add_interval(iv, 3, 9)
    assert iv == [(0, 10)]
    iv = [(0, 4)]
    TB.add_interval(iv, 6, 8)
    assert iv == [(0, 4), (6, 8)]


def test_overlap_is_counted_once_and_the_last_chunk_is_cut_at_a_token():
    span = ("1", 4, 14)                                      # 10 characters, chunk A holds 4 of them, B (new part [8,15)) the other 6
    r = walk(["A", "B"], span, [3, 4, 5])
    assert r[3]["covered"] == 4 and r[3]["tokens_used"] == 3 and r[3]["reached"]              # budget filled exactly by chunk A
    assert r[4]["covered"] == 4 + 3 and r[4]["n_chunks"] == 2                                  # one token (dd, ends at 11) of B's new text: [8, 11) -> 3 characters
    assert r[5]["covered"] == 10                                                               # B's new text is dd ee = 2 tokens (the overlap cc is not paid twice)
    assert r[5]["span_len"] == 10 and r[5]["tokens_used"] == 5


def test_other_documents_cost_budget_but_cover_nothing():
    r = walk(["C", "A"], ("1", 0, 8), [3, 6])
    assert r[3]["covered"] == 0 and r[6]["covered"] == 8


def test_a_budget_smaller_than_the_first_chunk_cuts_it():
    r = walk(["A"], ("1", 0, 8), [2])
    assert r[2]["covered"] == 5 and r[2]["tokens_used"] == 2                                   # aa bb: characters [0, 5)
    r = walk(["A"], ("1", 0, 8), [1])
    assert r[1]["covered"] == 2


def test_ranking_exhausted_before_the_budget():
    r = walk(["A"], ("1", 0, 8), [100])
    assert not r[100]["reached"] and r[100]["tokens_used"] == 3 and r[100]["covered"] == 8


def test_rank_order_matters_and_repeats_cost_nothing():
    span = ("1", 18, 23)
    assert walk(["A", "B", "D"], span, [5])[5]["covered"] == 0                                    # A (3 tokens) + B's new text (2) fill the budget before D
    assert walk(["A", "B", "D"], span, [6])[6]["covered"] == 2                                    # one token of D (gg: characters 18-20) still fits
    assert walk(["D", "A"], span, [3])[3]["covered"] == 5
    assert walk(["A", "A", "A", "D"], span, [6])[6]["covered"] == 5                            # the same chunk again is free


def test_hit_uses_exact_fractions():
    res = [{"covered": 9, "span_len": 10}, {"covered": 89, "span_len": 100}, {"covered": 90, "span_len": 100}, {"covered": 0, "span_len": 5}]
    assert list(TB.hit_array(res, Fraction(9, 10))) == [1.0, 0.0, 1.0, 0.0]
    assert list(TB.hit_array(res, Fraction(1))) == [0.0, 0.0, 0.0, 0.0]


def test_chunk_positions_and_hf_offsets(tmp_path):
    f = tmp_path / "chunks.jsonl"
    f.write_text(json.dumps({"chunk_id": "007:1", "source_doc_id": "007", "start_char": 5, "end_char": 9}) + "\n")
    assert TB.load_chunk_positions(f) == {"007:1": ("7", 5, 9)}                              # ids are normalised
    class Tok:
        def __call__(self, text, **kw):
            return {"offset_mapping": [(0, 0), (0, 2), (3, 5)]}
    assert TB.hf_token_offsets("aa bb", Tok()) == ([0, 3], [2, 5])                           # empty offsets (special tokens) are dropped


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("tb"))


def ws_tokenize(text):
    return ws_tokens(text)


def test_end_to_end_on_a_small_corpus(world, tmp_path):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense", "bm25"])
    rows = TB.analyse(runs, world["qa_path"], world["dir"] / "full_text.jsonl", budgets=[20, 100, 400], cover_levels=[0.9], resamples=100, seed=1,
                      tokenize=ws_tokenize, per_query_dir=tmp_path / "pq", verbose=False)
    assert len(rows) == len(runs) * 3
    by = {(r["model"], r["chunker"], r["size"], r["budget_tokens"]): r for r in rows}
    for r in rows:
        assert 0 <= r["mean_span_coverage"] <= 1 and r["span_coverage_ci_low"] <= r["mean_span_coverage"] + 1e-9 <= r["span_coverage_ci_high"] + 1e-9
        assert 0 <= r["hit_rate"] <= r["mean_span_coverage"] + 1e-9 + 1                    # a hit needs coverage >= 0.9
        assert r["cover_level"] == 0.9 and r["resamples"] == 100
    for m in ("org/m_a", "bm25"):                                                               # more budget never covers less
        cov = [by[(m, "fixed_token", 128, b)]["mean_span_coverage"] for b in (20, 100, 400)]
        assert cov[0] <= cov[1] <= cov[2]
    assert by[("org/m_a", "fixed_token", 512, 100)]["mean_span_coverage"] > by[("org/m_c", "fixed_token", 512, 100)]["mean_span_coverage"]   # a better ranking covers more
    assert by[("org/m_a", "fixed_token", 128, 20)]["share_budget_reached"] == 1.0
    assert len(list((tmp_path / "pq").glob("token_budget_per_query__*.csv"))) == len(runs)


def test_coverage_equals_a_direct_count_for_one_question(world):
    """Walk the ranking of one run by hand (whole chunks, no overlap in this corpus) and compare."""
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    meta = runs.meta[runs.index["org/m_a|fixed_token|128|0"]]
    pos = TB.load_chunk_positions(world["prepared"][("fixed_token", 128, 0)] / "chunks" / "chunks.jsonl")
    from tools import metrics as M
    run = M.load_run(__import__("pathlib").Path(meta["out_dir"]) / "run_top100.trec.gz")
    tok = {str(p): ws_tokens(__import__("analysis_world").paper_text(p)) for p in range(1, 13)}
    qa = world["qa"][4]
    span = (str(int(qa["corpus_id"])), qa["start_index"], qa["end_index"])
    ranking = [d for d, _ in M.ranked(run["4"])]
    used = cov = 0
    for cid in ranking:
        doc, a, b = pos[cid]
        n = sum(1 for s in tok[doc][0] if a <= s < b)
        if used + n > 60:
            break
        used += n
        cov += max(0, min(b, span[2]) - max(a, span[1])) if doc == span[0] else 0
    got = TB.walk_budgets(ranking, pos, tok, span, [60])[60]
    assert got["covered"] >= cov                                              # the cut chunk may add a part of the span
    assert got["covered"] <= span[2] - span[1]


def test_budgets_come_from_the_config():
    assert TB.budgets_from_config() == [512, 1024, 2048]

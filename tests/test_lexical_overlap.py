"""tools/lexical_overlap.py: question/evidence overlap and the BM25 advantage by overlap."""
import json
import math

import numpy as np
import pytest

from tools import lexical_overlap as L


def test_content_words_drop_stopwords_keep_numbers_and_types():
    assert L.content_words("What is the yield of the Pd/C catalyst at 80 C?") == {"yield", "pd", "c", "catalyst", "80"}
    assert L.content_words("") == set()


def test_overlap_share():
    s, n, hit = L.overlap_share("What was the yield of furfural at 150 C?", "Furfural yield reached 7.5% at 150 degrees")
    assert n == 4 and hit == 3 and s == 0.75      # furfural, yield, 150 and c; c is not in the span -> verify below
    assert L.overlap_share("What is the?", "anything")[0] != L.overlap_share("What is the?", "anything")[0]   # NaN: no content words
    s, n, hit = L.overlap_share("yield furfural", "the furfural yield")
    assert (s, n, hit) == (1.0, 2, 2)
    s, n, hit = L.overlap_share("yield furfural temperature", "furfural")
    assert (n, hit) == (3, 1) and s == pytest.approx(1 / 3)


def test_evidence_rank_is_the_best_relevant_and_101_when_missing():
    r = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert L.evidence_rank(r, ["c", "b"]) == 2 and L.evidence_rank(r, ["x"]) == 101


def test_bins_and_distribution():
    assert L.bin_of(0.0) == "[0,0.25)" and L.bin_of(0.25) == "[0.25,0.5)" and L.bin_of(1.0) == "[0.75,1]" and L.bin_of(float("nan")) is None
    d = L.distribution([0.0, 0.5, 1.0, 1.0, float("nan")])
    assert d["n"] == 5 and d["n_with_content_words"] == 4 and d["pct_zero_overlap"] == 25.0 and d["pct_full_overlap"] == 50.0
    assert d["pct_bin_0.75_1"] == 50.0


def test_advantage_grows_with_overlap_gives_positive_slope():
    rng = np.random.default_rng(0)
    n = 120
    table = [{"query_id": str(i), "corpus_id": str(i // 4), "overlap_share": i / (n - 1)} for i in range(n)]
    dense = {str(i): {"ndcg_at_10": 0.5, "recall_at_10": 0.5, "mrr_at_100": 0.5} for i in range(n)}
    bm = {str(i): {m: 0.5 + 0.4 * (i / (n - 1) - 0.5) for m in ("ndcg_at_10", "recall_at_10", "mrr_at_100")} for i in range(n)}
    rows = L.advantage_rows(table, bm, dense, seed=1)
    trend = next(r for r in rows if r["metric"] == "ndcg_at_10" and r["bin"] == "trend")
    assert trend["slope_per_unit_overlap"] == pytest.approx(0.4) and trend["spearman_rho"] == pytest.approx(1.0)
    assert trend["slope_ci_low"] <= 0.4 <= trend["slope_ci_high"]
    low = next(r for r in rows if r["metric"] == "ndcg_at_10" and r["bin"] == "[0,0.25)")
    high = next(r for r in rows if r["metric"] == "ndcg_at_10" and r["bin"] == "[0.75,1]")
    assert low["mean_advantage_bm25"] < 0 < high["mean_advantage_bm25"] and low["ci_low"] <= low["mean_advantage_bm25"] <= low["ci_high"]


def test_question_table_uses_row_index_as_query_id(tmp_path):
    p = tmp_path / "qa.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question": "yield furfural?", "corpus_id": "098", "content": "furfural yield"},
        {"question": "temperature?", "corpus_id": "1", "content": "something else"}]))
    t = L.question_table(p)
    assert [r["query_id"] for r in t] == ["0", "1"] and t[0]["overlap_share"] == 1.0 and t[1]["overlap_share"] == 0.0 and t[0]["corpus_id"] == "098"

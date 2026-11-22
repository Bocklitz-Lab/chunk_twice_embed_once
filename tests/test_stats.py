"""tools/stats.py: bootstrap CIs (query and paper level), paired tests, Holm, label agreement, Kendall's tau."""
import csv
import json

import numpy as np
import pytest

from tests.test_make_span_qrels import QUESTIONS, _build, _run
from tools import metrics as M
from tools import stats as S


def _pq(path, ndcg, recall):
    pq = {str(i): {"ndcg_at_10": float(n), "recall_at_10": float(r), "geom_at_10": float(np.sqrt(n * r)), "n_relevant": 1}
          for i, (n, r) in enumerate(zip(ndcg, recall))}
    M.write_per_query_csv(path, pq)
    return pq


def test_ci_contains_the_estimate_is_seeded_and_geom_is_geom_of_means():
    rng = np.random.default_rng(1)
    n, r = rng.random(300), (rng.random(300) > 0.3).astype(float)
    a = S.bootstrap_ci(n, r, B=1000, seed=7)
    b = S.bootstrap_ci(n, r, B=1000, seed=7)
    assert a == b and S.bootstrap_ci(n, r, B=1000, seed=8) != a
    for m in S.METRICS:
        assert a[m]["ci_low"] <= a[m]["estimate"] <= a[m]["ci_high"]
    assert a["geom_at_10"]["estimate"] == pytest.approx(np.sqrt(n.mean() * r.mean()))
    assert a["ndcg_at_10"]["estimate"] == pytest.approx(n.mean())
    # the interval shrinks with the number of resamples' noise floor: width ~ 2 * 1.96 * sd / sqrt(n)
    assert (a["ndcg_at_10"]["ci_high"] - a["ndcg_at_10"]["ci_low"]) == pytest.approx(2 * 1.96 * n.std() / np.sqrt(300), rel=0.2)


def test_paper_level_interval_is_wider_when_questions_of_a_paper_are_alike():
    rng = np.random.default_rng(3)
    paper_effect = np.repeat(rng.random(40), 8)               # 40 papers x 8 questions, all alike within a paper
    n = np.clip(paper_effect + rng.normal(0, 0.02, 320), 0, 1)
    groups = np.repeat(np.arange(40), 8).astype(str)
    q = S.bootstrap_ci(n, n, None, B=1000, seed=1)["ndcg_at_10"]
    p = S.bootstrap_ci(n, n, groups, B=1000, seed=1)["ndcg_at_10"]
    assert (p["ci_high"] - p["ci_low"]) > 1.8 * (q["ci_high"] - q["ci_low"])
    assert p["estimate"] == pytest.approx(q["estimate"])


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(2)
    base = rng.random(400)
    a = {"ndcg_at_10": np.clip(base + 0.05, 0, 1), "recall_at_10": np.clip(base + 0.05, 0, 1)}
    b = {"ndcg_at_10": base, "recall_at_10": base}
    res = S.paired_bootstrap(a, b, B=1000, seed=1)
    assert res["ndcg_at_10"]["diff"] > 0.03 and res["ndcg_at_10"]["ci_low"] > 0 and res["ndcg_at_10"]["p_bootstrap"] < 0.01
    assert S.wilcoxon_p(a["ndcg_at_10"], b["ndcg_at_10"]) < 0.001
    same = S.paired_bootstrap(b, b, B=1000, seed=1)
    assert same["ndcg_at_10"]["diff"] == 0 and same["ndcg_at_10"]["p_bootstrap"] == 1.0
    assert S.wilcoxon_p(b["ndcg_at_10"], b["ndcg_at_10"]) == 1.0


def test_holm_matches_the_textbook_example():
    # sorted p: .01, .02, .03, .04 with m=4 -> .04, .06, .06, .06 (monotone)
    assert S.holm([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert S.holm([0.5]) == [0.5] and S.holm([0.4, 0.9]) == pytest.approx([0.8, 0.9])


def test_cli_ci_and_compare_write_csv_with_holm(tmp_path):
    rng = np.random.default_rng(5)
    base = rng.random(60)
    for name, shift in (("A", 0.0), ("B", 0.05), ("C", 0.0)):
        _pq(tmp_path / f"{name}.csv", np.clip(base + shift, 0, 1), np.clip(base + shift, 0, 1))
    qa = tmp_path / "qa.jsonl"
    qa.write_text("".join(json.dumps({"corpus_id": i // 4}) + "\n" for i in range(60)))
    assert S.main(["ci", "--per-query", f"A={tmp_path/'A.csv'}", "--qa", str(qa), "--out", str(tmp_path / "ci.csv")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "ci.csv")))
    assert {r["level"] for r in rows} == {"query", "paper"} and len(rows) == 6
    assert S.main(["compare", "--runs", *(f"{n}={tmp_path/(n+'.csv')}" for n in "ABC"), "--baseline", "A", "--qa", str(qa),
                   "--out", str(tmp_path / "cmp.csv")]) == 0
    cmp_rows = list(csv.DictReader(open(tmp_path / "cmp.csv")))
    assert len(cmp_rows) == 2 * 2 * 3                          # {query, paper} x {B, C} x 3 metrics
    assert all(float(r["p_bootstrap_holm"]) >= float(r["p_bootstrap"]) for r in cmp_rows)
    ba = next(r for r in cmp_rows if r["B"] == "B" and r["metric"] == "ndcg_at_10" and r["level"] == "query")
    assert float(ba["diff"]) < 0 and float(ba["p_wilcoxon_per_query"]) < 0.05    # A - B: B is better


def test_label_agreement_on_the_synthetic_task(tmp_path):
    d = _build(tmp_path); _run(d, "--cover-level", "0.9")
    out = d / "out"
    # lexical: q0 {0:1 (best), 0:0}, q1 {0:0}, q2 {1:0}, q3 {1:0}, q4 {2:2}, q5 {2:1}, q6 {0:0}, q7 {}
    rows = [("0", "0:1", 3.0), ("0", "0:0", 2.0), ("1", "0:0", 2.0), ("2", "1:0", 2.0), ("3", "1:0", 2.0), ("4", "2:2", 2.0),
            ("5", "2:1", 2.0), ("6", "0:0", 2.0)]
    (out / "qrels_lexical.jsonl").write_text("".join(json.dumps({"query_id": q, "doc_id": c, "relevance": 1, "score": s}) + "\n" for q, c, s in rows))
    r = S.label_agreement(out)
    span = M.load_qrels(out / "qrels_span_c090.jsonl")
    assert r["n_questions"] == 8 and r["n_only_span_has_relevant"] >= 1       # q7 has span labels but no lexical label
    assert 0 <= r["pct_same_top_chunk"] <= 100 and 0 <= r["mean_jaccard"] <= 1
    assert r["mean_relevant_lexical"] == pytest.approx(8 / 8)
    # q0: span label is the chunk holding the span (0:1), lexical top (score 3) is 0:1 too -> counted as the same top chunk
    assert span["0"] == {"0:1": 1}


def test_kendall_tau_between_rankings(tmp_path):
    rows = [{"model": "m", "chunker": "c", "size": str(s), "overlap": "0", "span_geom_at_10": str(0.5 + s / 1000), "lexical_geom_at_10": str(0.6 + s / 900)}
            for s in range(100, 600, 100)]
    p = tmp_path / "runs.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out = S.ranking_agreement(p)
    assert out[0]["scope"] == "all_runs" and out[0]["tau"] == pytest.approx(1.0) and out[0]["n"] == 5
    rows[2]["lexical_geom_at_10"] = "0.1"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert S.ranking_agreement(p)[0]["tau"] < 1.0


def test_cli_without_out_prints_only(tmp_path, capsys):
    rng = np.random.default_rng(9)
    base = rng.random(40)
    _pq(tmp_path / "A.csv", base, base); _pq(tmp_path / "B.csv", np.clip(base + 0.1, 0, 1), base)
    assert S.main(["ci", "--per-query", str(tmp_path / "A.csv")]) == 0
    assert S.main(["compare", "--runs", f"A={tmp_path/'A.csv'}", f"B={tmp_path/'B.csv'}"]) == 0
    assert "geom_at_10" in capsys.readouterr().out

"""B7: 10,000 resamples from the config, one resampling per run, dense-vs-dense and rerank pairs, Holm families, vectorised overlap bootstrap."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("scipy")
from tools import lexical_overlap as LO, make_report as R, metrics as M, stats as S

ROOT = Path(__file__).resolve().parents[1]
REPORT = yaml.safe_load((ROOT / "configs" / "main_config.yaml").read_text())["params"]["report"]


def test_resamples_default_is_10000_and_read_from_the_config():
    assert REPORT["resamples"] == 10000 and S.DEFAULT_RESAMPLES == 10000 and LO.RESAMPLES == 10000
    assert not hasattr(LO, "B")                                        # the hard-coded constant is gone
    import inspect
    assert inspect.signature(R.build_report).parameters["resamples"].default == 10000


# ---- pairs and families -------------------------------------------------------------------------------------
CELL = ("fixed_token", 128, 0)


def _infos(**kw):
    return {f"{n}|fixed_token|128|0": v for n, v in kw.items()}


def test_dense_vs_dense_pairs_are_added_in_alphabetical_order():
    names = ["z/dense", "a/dense", "m/dense", "bm25"]
    pairs = R.comparison_pairs(names, {}, CELL)
    assert pairs[:3] == [("z/dense", "bm25"), ("a/dense", "bm25"), ("m/dense", "bm25")]
    assert pairs[3:] == [("a/dense", "m/dense"), ("a/dense", "z/dense"), ("m/dense", "z/dense")]
    assert len(R.comparison_pairs([f"d{i}" for i in range(13)] + ["bm25"], {}, CELL)) == 13 + 78                 # 13 models: 78 dense pairs


def test_rerank_is_compared_with_dense_bm25_and_its_hybrid():
    names = ["a/d", "b/d", "bm25", "hy_a", "hy_b", "rr_a"]
    infos = _infos(hy_a={"hybrid": {"dense": "a/d", "sparse": "bm25"}}, hy_b={"hybrid": {"dense": "b/d", "sparse": "bm25"}},
                   rr_a={"rerank": {"dense": "a/d", "top_n": 50}})
    pairs = R.comparison_pairs(names, infos, CELL, rerank_top_n=50)
    assert ("rr_a", "a/d") in pairs and ("rr_a", "bm25") in pairs and ("rr_a", "hy_a") in pairs and ("rr_a", "hy_b") not in pairs
    assert ("hy_a", "a/d") in pairs and ("hy_a", "bm25") in pairs and ("a/d", "b/d") in pairs
    assert len(pairs) == len(set(pairs)) == 2 + 4 + 3 + 1                                 # 2 dense-bm25, 4 hybrid, 3 rerank, 1 dense-dense
    # no BM25 and no hybrid: the rerank run is only compared with its dense run
    assert R.comparison_pairs(["a/d", "rr_a"], _infos(rr_a={"rerank": {"dense": "a/d", "top_n": 50}}), CELL) == [("rr_a", "a/d")]


def test_pair_families():
    infos = _infos(hy={"hybrid": {"dense": "a/d", "sparse": "bm25"}}, rr={"rerank": {"dense": "a/d", "top_n": 50}})
    assert R.pair_family(("a/d", "bm25"), infos, CELL) == "dense_vs_bm25"
    assert R.pair_family(("a/d", "b/d"), infos, CELL) == "dense_vs_dense"
    assert R.pair_family(("hy", "bm25"), infos, CELL) == "hybrid" and R.pair_family(("rr", "bm25"), infos, CELL) == "rerank"


def _run_files(tmp_path, scores):
    paths = {}
    for name, vals in scores.items():
        pq = {str(i): {"ndcg_at_10": n, "recall_at_10": r, "geom_at_10": (n * r) ** .5} for i, (n, r) in enumerate(vals)}
        p = tmp_path / f"{name.replace('/', '_')}.csv"
        M.write_per_query_csv(p, pq)
        paths[name] = str(p)
    return paths


def _scores(seed, n=40, shift=0.0):
    rng = np.random.default_rng(seed)
    return [(float(np.clip(rng.random() * .6 + shift, 0, 1)), float(np.clip(rng.random() * .6 + shift + .2, 0, 1))) for _ in range(n)]


def test_holm_is_applied_within_each_family(tmp_path):
    paths = _run_files(tmp_path, {"a": _scores(1), "b": _scores(2, shift=.1), "c": _scores(3), "bm25": _scores(4)})
    pairs = [("a", "bm25"), ("b", "bm25"), ("c", "bm25"), ("a", "b")]
    fams = {("a", "bm25"): "x", ("b", "bm25"): "x", ("c", "bm25"): "x", ("a", "b"): "y"}
    rows = S.compare_rows(paths, pairs, None, 200, 1, families=fams)
    sel = [r for r in rows if r["metric"] == "geom_at_10" and r["level"] == "query"]
    y = next(r for r in sel if r["family"] == "y")
    assert y["p_bootstrap_holm"] == y["p_bootstrap"]                               # a family of one: nothing to correct
    x = sorted((r for r in sel if r["family"] == "x"), key=lambda r: r["p_bootstrap"])
    assert x[0]["p_bootstrap_holm"] == pytest.approx(min(1.0, 3 * x[0]["p_bootstrap"]))
    one_family = S.compare_rows(paths, pairs, None, 200, 1)                         # default: one family per metric and level, as before
    assert {r["family"] for r in one_family} == {"all"}
    z = next(r for r in one_family if r["A"] == "a" and r["B"] == "b" and r["metric"] == "geom_at_10")
    assert z["p_bootstrap_holm"] >= y["p_bootstrap_holm"]


def test_build_report_writes_the_new_pairs_and_families(tmp_path):
    from test_make_report import _run
    root = tmp_path / "art"
    (root / "tests").mkdir(parents=True)
    qa = root / "qa.jsonl"
    qa.write_text("".join(json.dumps({"question": "q", "corpus_id": str(i // 2), "content": "w", "start_index": 0, "end_index": 1}) + "\n" for i in range(6)))
    base = {str(i): (0.5 + 0.05 * (i % 3), 0.7) for i in range(6)}
    for name, model, shift in (("d1", "org/d1", 0), ("d2", "org/d2", .1), ("d3", "org/d3", .2), ("bm", "bm25", 0)):
        _run(root / "tests", f"{name}_fixed_token_c128_o0", model, "fixed_token", 128, {q: (n + shift, r) for q, (n, r) in base.items()})
    out = tmp_path / "out"
    R.build_report(root, out, qa, resamples=30, overlap=False, verbose=False)
    import csv
    rows = list(csv.DictReader(open(out / "stats_compare.csv")))
    pairs = {(r["A"], r["B"], r["family"]) for r in rows}
    assert ("org/d1", "org/d2", "dense_vs_dense") in pairs and ("org/d1", "bm25", "dense_vs_bm25") in pairs and len({p[:2] for p in pairs}) == 6


# ---- the draws and the shared cache ------------------------------------------------------------------------
def _naive_resampled_means(ndcg, recall, groups, B, seed):
    """The bootstrap written as plainly as possible: draw units, concatenate their questions, take the means."""
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [np.array([i]) for i in range(len(ndcg))]
    else:
        members = [np.where(groups == g)[0] for g in np.unique(groups)]
    bn, br = [], []
    for _ in range(B):
        pick = rng.integers(0, len(members), len(members))
        q = np.concatenate([members[i] for i in pick])
        bn.append(ndcg[q].mean())
        br.append(recall[q].mean())
    return np.array(bn), np.array(br)


@pytest.mark.parametrize("clustered", [False, True])
def test_resampled_means_equal_the_plain_loop(clustered):
    rng = np.random.default_rng(5)
    n, r = rng.random(60), rng.random(60)
    groups = np.repeat(np.arange(15), 4).astype(str) if clustered else None
    got = S.resampled_means(n, r, groups, 50, 3)
    ref = _naive_resampled_means(n, r, groups, 50, 3)
    assert np.allclose(got[0], ref[0]) and np.allclose(got[1], ref[1])


def test_draws_are_memoised_and_shared_between_runs():
    assert S.resample_indices(30, 20, 1) is S.resample_indices(30, 20, 1)
    assert not np.array_equal(S.resample_indices(30, 20, 1), S.resample_indices(30, 20, 2))


def test_ci_and_compare_share_one_resampling_per_run_and_level(tmp_path):
    paths = _run_files(tmp_path, {"a": _scores(1), "b": _scores(2), "c": _scores(3)})
    qa = tmp_path / "qa.jsonl"
    qa.write_text("".join(json.dumps({"corpus_id": str(i // 4)}) + "\n" for i in range(40)))
    cache = {}
    ci = S.ci_rows(paths, str(qa), 100, 1, cache=cache)
    n_after_ci = len(cache)
    cmp_ = S.compare_rows(paths, [("a", "b"), ("a", "c"), ("b", "c")], str(qa), 100, 1, cache=cache)
    assert n_after_ci == 6 and len(cache) == 6                                      # 3 runs x 2 levels; the pairs added nothing
    assert ci == S.ci_rows(paths, str(qa), 100, 1) and cmp_ == S.compare_rows(paths, [("a", "b"), ("a", "c"), ("b", "c")], str(qa), 100, 1)


def test_paired_bootstrap_matches_its_building_blocks():
    rng = np.random.default_rng(2)
    a = {"ndcg_at_10": rng.random(50), "recall_at_10": rng.random(50)}
    b = {"ndcg_at_10": rng.random(50), "recall_at_10": rng.random(50)}
    direct = S.paired_bootstrap(a, b, None, 100, 4)
    an, ar = S.resampled_means(a["ndcg_at_10"], a["recall_at_10"], None, 100, 4)
    bn, br = S.resampled_means(b["ndcg_at_10"], b["recall_at_10"], None, 100, 4)
    assert direct == S.paired_from_means((a["ndcg_at_10"].mean(), a["recall_at_10"].mean()), (b["ndcg_at_10"].mean(), b["recall_at_10"].mean()), an, ar, bn, br, 100)


# ---- vectorised lexical-overlap bootstrap -------------------------------------------------------------------
def _old_cluster_boot(values, groups, stat, rng, B):
    uniq = np.unique(groups)
    members = [np.where(groups == g)[0] for g in uniq]
    res = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        res[b] = stat(np.concatenate([members[i] for i in pick]))
    return float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def test_vectorised_overlap_bootstrap_equals_the_old_loop():
    rng = np.random.default_rng(7)
    x, y = rng.random(80), rng.normal(size=80)
    groups = np.repeat(np.arange(20), 4).astype(str)
    old_mean = _old_cluster_boot(y, groups, lambda i: y[i].mean(), np.random.default_rng(9), 300)
    new_mean = LO._cluster_boot_mean(y, groups, np.random.default_rng(9), 300)
    assert np.allclose(old_mean, new_mean, atol=1e-12)
    old_slope = _old_cluster_boot(y, groups, lambda i: float(np.polyfit(x[i], y[i], 1)[0]), np.random.default_rng(9), 300)
    new_slope = LO._cluster_boot_slope(x, y, groups, np.random.default_rng(9), 300)
    assert np.allclose(old_slope, new_slope, atol=1e-9)


def test_advantage_rows_use_the_given_number_of_resamples_and_are_fast():
    import time
    table = [{"query_id": str(i), "corpus_id": str(i // 4), "overlap_share": (i % 10) / 10, "n_question_content_words": 5, "n_in_span": 2, "span_chars": 100} for i in range(80)]
    rng = np.random.default_rng(1)
    bm = {str(i): {k: float(rng.random()) for k in ("ndcg_at_10", "recall_at_10", "mrr_at_100")} for i in range(80)}
    de = {str(i): {k: float(rng.random()) for k in ("ndcg_at_10", "recall_at_10", "mrr_at_100")} for i in range(80)}
    t0 = time.time()
    rows = LO.advantage_rows(table, bm, de, resamples=10000)
    assert time.time() - t0 < 20                                                    # the old loop needed minutes for this setting
    a = LO.advantage_rows(table, bm, de, resamples=50)
    b = LO.advantage_rows(table, bm, de, resamples=50)
    assert a == b and len(rows) == len(a) and any(r["bin"] == "trend" and r["slope_ci_low"] < r["slope_ci_high"] for r in rows)

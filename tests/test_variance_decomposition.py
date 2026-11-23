"""tools/variance_decomposition.py: Type II sums of squares, eta^2 with paper-level intervals, balanced sub-grid, simple ranges."""
import itertools

import numpy as np
import pytest

from analysis_world import build_world
from tools import analysis_data as AD, variance_decomposition as VD


def grid_data(sizes=(1, 2, 3), overlaps=(0, 1), models=3, strategies=2, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    rows = [(m, s, z, o) for m in range(models) for s in range(strategies) for z in sizes for o in overlaps if (m, s, z, o) not in drop]
    codes = {"model": np.array([r[0] for r in rows]), "strategy": np.array([r[1] for r in rows]), "size": np.array([r[2] - 1 for r in rows]),
             "overlap": np.array([r[3] for r in rows])}
    y = 0.5 * codes["model"] + 0.1 * codes["strategy"] + 0.05 * codes["size"] + 0.02 * codes["overlap"] + 0.15 * (codes["model"] * codes["strategy"]) + 0.01 * rng.normal(size=len(rows))
    return codes, y


def rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum())


def reference_type2(codes, y):
    """Type II by explicit least squares on dummy matrices (an independent implementation)."""
    n = len(y)
    D = {f: VD.dummies(codes[f]) for f in VD.FACTORS}
    one = np.ones((n, 1))
    full = np.hstack([one] + [D[f] for f in VD.FACTORS])
    out = {f: rss(np.hstack([one] + [D[g] for g in VD.FACTORS if g != f]), y) - rss(full, y) for f in VD.FACTORS}
    inter = np.hstack([D["model"][:, [i]] * D["strategy"][:, [j]] for i in range(D["model"].shape[1]) for j in range(D["strategy"].shape[1])])
    out["model_x_strategy"] = rss(full, y) - rss(np.hstack([full, inter]), y)
    out["residual"] = rss(np.hstack([full, inter]), y)
    out["total"] = float(((y - y.mean()) ** 2).sum())
    return out


@pytest.mark.parametrize("drop", [(), ((0, 0, 3, 1), (1, 1, 1, 0), (2, 0, 2, 1))])
def test_type2_sums_of_squares_equal_an_explicit_least_squares_fit(drop):
    codes, y = grid_data(drop=drop)
    got = VD.type2(VD.design_bases(codes), y[None, :])
    ref = reference_type2(codes, y)
    for f in VD.FACTORS:
        assert got[f"ss_{f}"][0] == pytest.approx(ref[f], rel=1e-8, abs=1e-12), f
    assert got["ss_model_x_strategy"][0] == pytest.approx(ref["model_x_strategy"], rel=1e-8) and got["ss_residual"][0] == pytest.approx(ref["residual"], rel=1e-8)
    assert got["ss_total"][0] == pytest.approx(ref["total"])
    assert got["eta2_model"][0] == pytest.approx(ref["model"] / ref["total"])


def test_balanced_design_is_additive():
    codes, y = grid_data()
    r = VD.type2(VD.design_bases(codes), y[None, :])
    parts = sum(r[f"ss_{f}"][0] for f in (*VD.FACTORS, "model_x_strategy")) + r["ss_residual"][0]
    assert parts == pytest.approx(r["ss_total"][0], rel=1e-9)
    assert r["eta2_model"][0] > r["eta2_strategy"][0] > r["eta2_overlap"][0]                    # the effects were built in this order


def test_the_coding_of_the_factors_does_not_matter():
    codes, y = grid_data()
    shuffled = {k: (v.max() - v) for k, v in codes.items()}                                      # relabel the levels
    a, b = VD.type2(VD.design_bases(codes), y[None, :]), VD.type2(VD.design_bases(shuffled), y[None, :])
    for k in a:
        assert a[k][0] == pytest.approx(b[k][0], rel=1e-8, abs=1e-12)


def test_rows_of_resamples_are_independent_fits():
    codes, y = grid_data()
    Y = np.vstack([y, y[::-1], y + 1.0])
    r = VD.type2(VD.design_bases(codes), Y)
    for i in range(3):
        one = VD.type2(VD.design_bases(codes), Y[i][None, :])
        assert r["eta2_model"][i] == pytest.approx(one["eta2_model"][0])
    assert r["eta2_model"][2] == pytest.approx(r["eta2_model"][0])                              # a constant shift changes no sum of squares


def test_largest_balanced_subgrid():
    full = {(s, o) for s in (128, 192, 256, 320, 384, 448, 512) for o in (0, 16, 32, 48, 64) if o <= min(64, 0.3 * s) and s - o >= 96 and s + o <= 512}
    meta = [{"model": m, "chunker": c, "size": s, "overlap": o} for m in "ab" for c in "xy" for (s, o) in sorted(full)]
    keep, sizes, ovs = VD.balanced_rows(meta)
    # two rectangles have 20 cells (sizes 256-448 x overlaps 0-64 and sizes 192-448 x overlaps 0-48): the tie goes to more overlap levels
    assert sizes == [256, 320, 384, 448] and ovs == [0, 16, 32, 48, 64] and len(keep) == 2 * 2 * 4 * 5
    assert {(meta[i]["size"], meta[i]["overlap"]) for i in keep} == set(itertools.product(sizes, ovs))
    meta2 = [m for m in meta if not (m["model"] == "b" and m["size"] == 256 and m["overlap"] == 48)]          # a missing cell for one model is removed for all
    keep2, sizes2, ovs2 = VD.balanced_rows(meta2)
    assert (256, 48) not in {(meta2[i]["size"], meta2[i]["overlap"]) for i in keep2}


def test_simple_ranges():
    codes, _ = grid_data(sizes=(1,), overlaps=(0,))            # 3 models x 2 strategies, one run each
    g = np.array([[0.1, 0.5, 0.3, 0.6, 0.2, 0.9]])             # order: m0s0 m0s1 m1s0 m1s1 m2s0 m2s1
    r = VD.range_statistics(g, codes["model"], codes["strategy"])
    # strategy 0: models 0.1, 0.3, 0.2 -> range 0.2; strategy 1: 0.5, 0.6, 0.9 -> 0.4: mean 0.3.  models: |0.5-0.1|=0.4, |0.6-0.3|=0.3, |0.9-0.2|=0.7 -> 0.4667
    assert r["range_models"][0] == pytest.approx(0.3) and r["range_strategies"][0] == pytest.approx((0.4 + 0.3 + 0.7) / 3)
    assert r["range_models_minus_strategies"][0] == pytest.approx(0.3 - 1.4 / 3)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("vd"), models={"org/m_a": 2.2, "org/m_b": 1.2, "org/m_c": 0.2}, with_rerank=False)


def test_end_to_end_and_cli(world, tmp_path):
    runs = AD.load_runs([world["root"]], world["qa_path"])
    res = VD.analyse(runs, resamples=300, seed=2)
    by = {r["term"]: r for r in res["rows"]}
    assert set(by) == {"model", "strategy", "size", "overlap", "model_x_strategy", "residual", "model_minus_strategy"}
    for t in ("model", "strategy", "size", "model_x_strategy", "residual"):
        assert 0 <= by[t]["eta2"] <= 1 and by[t]["eta2_ci_low"] <= by[t]["eta2_ci_high"]
    assert by["model"]["eta2"] > by["strategy"]["eta2"]                                          # the models differ by far more than the strategies in this world
    assert by["model_minus_strategy"]["eta2"] == pytest.approx(by["model"]["eta2"] - by["strategy"]["eta2"])
    assert res["info"]["n_runs"] == 12 and res["info"]["sizes"] == [128, 512]
    rng = {r["statistic"]: r for r in res["ranges"]}
    assert rng["range_models"]["value"] > rng["range_strategies"]["value"] and rng["range_models"]["ci_low"] <= rng["range_models"]["ci_high"]
    out = tmp_path / "o"
    assert VD.main(["--roots", str(world["root"]), "--qa", str(world["qa_path"]), "--out-dir", str(out), "--resamples", "50", "--balanced"]) == 0
    for f in ("variance_decomposition.csv", "variance_decomposition_balanced.csv", "variance_ranges.csv", "variance_decomposition.json"):
        assert (out / f).exists(), f


def test_one_model_or_one_strategy_is_refused(world):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    keep = AD.RunSet(meta=[runs.meta[i] for i in runs.select(model="org/m_a")], qids=runs.qids, metrics={k: v[runs.select(model="org/m_a")] for k, v in runs.metrics.items()},
                     groups=runs.groups)
    with pytest.raises(ValueError, match="at least two models"):
        VD.analyse(keep, resamples=10, seed=1)

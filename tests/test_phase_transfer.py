"""tools/phase_transfer.py: Phase 1 -> Phase 2 rank correlation over models with a bootstrap over models, pairs, groups, beats, figure."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.stats import kendalltau, spearmanr

from tools import phase_transfer as PT

CFGS = [{"id": "C1", "chunker": "fixed_token", "size": 512, "overlap": 0}, {"id": "C2", "chunker": "recursive_token", "size": 512, "overlap": 0},
        {"id": "C6", "chunker": "fixed_token", "size": 128, "overlap": 0, "extra": True}]


def test_vectorised_rank_statistics_equal_scipy():
    rng = np.random.default_rng(1)
    X, Y = rng.integers(0, 6, size=(5, 12)).astype(float), rng.random((5, 12))      # X has ties
    sp, kt = PT.spearman_rows(X, Y), PT.kendall_rows(X, Y)
    for i in range(5):
        assert sp[i] == pytest.approx(spearmanr(X[i], Y[i]).statistic) and kt[i] == pytest.approx(kendalltau(X[i], Y[i], variant="b").statistic)


def test_correlation_perfect_reversed_and_degenerate():
    x = np.arange(10.0)
    up = PT.correlation(x, x * 2 + 1, 200, 1)
    assert up["spearman"] == pytest.approx(1) and up["kendall"] == pytest.approx(1) and up["spearman_ci_low"] == pytest.approx(1) and up["n_models"] == 10
    down = PT.correlation(x, -x, 200, 1)
    assert down["spearman"] == pytest.approx(-1) and down["kendall_ci_high"] == pytest.approx(-1)
    assert np.isnan(PT.correlation([1, 2, 3], [3, 2, 1], 50, 1)["spearman"])                  # fewer than four models
    assert np.isnan(PT.correlation([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], 50, 1)["spearman"])      # a constant variable
    noisy = PT.correlation(x, x + np.random.default_rng(0).normal(size=10) * 3, 500, 1)
    assert noisy["spearman_ci_low"] <= noisy["spearman"] <= noisy["spearman_ci_high"] + 1e-9 and noisy["spearman_p"] < 0.5
    assert PT.correlation(x, x[::-1] * 0 + np.arange(10) % 3, 100, 1) == PT.correlation(x, x[::-1] * 0 + np.arange(10) % 3, 100, 1)       # seeded


def _task_json(root, model, rev, task, nd, rc, split="test"):
    safe = model.replace("/", "__")
    d = Path(root) / f"{safe}_{rev}" / safe / rev
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task}.json").write_text(json.dumps({"scores": {split: [{"ndcg_at_10": nd, "recall_at_10": rc}]}}))


def test_load_phase1_needs_both_tasks_and_reads_the_configured_split_only(tmp_path):
    from pipeline_lib.phase1_split import Phase1SplitError
    rev = "a" * 40
    for t in PT.TASKS:
        _task_json(tmp_path, "org/both", rev, t, 0.5, 0.8)
    _task_json(tmp_path, "org/one", rev, PT.TASKS[0], 0.5, 0.8)
    p1 = PT.load_phase1(tmp_path, [("org/both", rev), ("org/one", rev), ("org/none", None)], eval_split={t: "test" for t in PT.TASKS}, n_queries={PT.TASKS[0]: {"test": 27}})
    assert set(p1) == {"org/both"}
    assert p1["org/both"][PT.TASKS[0]]["geom_at_10"] == pytest.approx((0.5 * 0.8) ** 0.5)
    assert p1["org/both"][PT.TASKS[0]]["split"] == "test" and p1["org/both"][PT.TASKS[0]]["n_queries"] == 27 and p1["org/both"][PT.TASKS[1]]["n_queries"] is None
    _task_json(tmp_path, "org/val", rev, PT.TASKS[0], 0.4, 0.4, "validation")
    _task_json(tmp_path, "org/val", rev, PT.TASKS[1], 0.4, 0.4, "train")
    with pytest.raises(Phase1SplitError, match="'test' is not in the result file"):               # no fallback to validation / train
        PT.load_phase1(tmp_path, [("org/val", None)], eval_split={t: "test" for t in PT.TASKS})
    assert PT.load_phase1(tmp_path, [("org/val", None)], eval_split={PT.TASKS[0]: "validation", PT.TASKS[1]: "train"})["org/val"][PT.TASKS[1]]["split"] == "train"


def test_phase_transfer_tables_carry_split_and_n_queries():
    p1 = {"a": {"ChemNQRetrieval": {"split": "test", "n_queries": 27}, "ChemHotpotQARetrieval": {"split": "test", "n_queries": 18}}}
    rows = PT.with_split_columns([{"x": 1}], p1)
    assert rows == [{"x": 1, "split": "ChemHotpotQARetrieval=test; ChemNQRetrieval=test", "n_queries": "ChemHotpotQARetrieval=18; ChemNQRetrieval=27"}]


def test_shortlist_files_of_both_formats(tmp_path):
    (tmp_path / "per.csv").write_text("model_id,model_name,revision\nx/a@abc,x/a,abc\ny/b@def,y/b,def\n")
    (tmp_path / "union.csv").write_text("model_id\nx/a@abc\nz/c@111\n")
    assert PT.read_shortlist(tmp_path / "per.csv") == {"x/a", "y/b"} and PT.read_shortlist(tmp_path / "union.csv") == {"x/a", "z/c"}


def test_read_phase2_first_file_wins_and_skips_baselines(tmp_path):
    head = "model,chunker,size,overlap,span_geom_at_10,span_ndcg_at_10,span_recall_at_10"
    (tmp_path / "main.csv").write_text(f"{head}\norg/a,fixed_token,512,0,0.7,0.6,0.8\nbm25,fixed_token,512,0,0.5,0.4,0.6\nhybrid_rrf60_x,fixed_token,512,0,0.9,0.9,0.9\n")
    (tmp_path / "transfer.csv").write_text(f"{head},kind\norg/a,fixed_token,512,0,0.1,0.1,0.1,dense\norg/b,fixed_token,512,0,0.3,0.2,0.4,dense\n")
    p2 = PT.read_phase2([tmp_path / "main.csv", tmp_path / "transfer.csv"])
    assert p2[("org/a", "fixed_token", 512, 0)]["geom_at_10"] == 0.7 and ("org/b", "fixed_token", 512, 0) in p2
    assert not any(k[0].startswith(("bm25", "hybrid")) for k in p2)


TRANSFER = {"configurations": CFGS, "models": [{"name": f"m{i}", "revision": "x", "role": "non_shortlisted" if i >= 6 else "shortlisted_reference", "group": "a" if i % 2 else "c"} for i in range(12)]
            + [{"name": "base", "revision": "x", "role": "extra", "group": "b"}],
            "pairs": [{"id": "P1", "base": "base", "tuned": ["m3"], "note": "n"}], "groups": {"a": "group a", "c": "group c", "b": "group b"}}


def synthetic(strength=0.9, seed=0):
    rng = np.random.default_rng(seed)
    p1, p2, short = {}, {}, set()
    for i in range(12):
        name = f"m{i}"
        score = 0.3 + 0.04 * i
        if i < 6:
            short.add(name)
        p1[name] = {t: {"ndcg_at_10": score, "recall_at_10": score + .1, "geom_at_10": (score * (score + .1)) ** .5, "split": "test"} for t in PT.TASKS}
        for c in CFGS:
            g = 0.5 + strength * (score - 0.3) + rng.normal() * 0.02
            p2[(name, c["chunker"], c["size"], c["overlap"])] = {"geom_at_10": g, "ndcg_at_10": g - .05, "recall_at_10": g + .05}
    for c in CFGS:
        p2[("base", c["chunker"], c["size"], c["overlap"])] = {"geom_at_10": 0.4, "ndcg_at_10": 0.35, "recall_at_10": 0.45}
    return p1, p2, short


def test_analyse_tables():
    p1, p2, short = synthetic()
    res = PT.analyse(p1, p2, short, TRANSFER, resamples=200, seed=1)
    models = {r["model"]: r for r in res["models"]}
    assert len(models) == 12 and models["m0"]["shortlisted"] and not models["m8"]["shortlisted"] and models["m2"]["group"] == "c" and not np.isnan(models["m0"]["phase2_geom_C6"])
    assert models["m5"]["phase2_geom_mean_main"] == pytest.approx(np.mean([models["m5"]["phase2_geom_C1"], models["m5"]["phase2_geom_C2"]]))        # C6 is not in the mean
    corr = {(r["population"], r["phase1"], r["phase2"]): r for r in res["correlations"]}
    allm = corr[("all models", "mean", "mean_main")]
    assert allm["n_models"] == 12 and allm["spearman"] > 0.9 and allm["spearman_ci_low"] < allm["spearman"] and allm["resamples"] == 200
    assert corr[("shortlisted only", "mean", "mean_main")]["n_models"] == 6 and corr[("excluded only", "mean", "C1")]["n_models"] == 6
    assert ("all models", PT.TASKS[0], "C6") in corr
    pair = {r["setting"]: r for r in res["pairs"]}
    assert pair["Phase 2 C1"]["delta_geom_at_10"] == pytest.approx(models["m3"]["phase2_geom_C1"] - 0.4) and np.isnan(pair["Phase 1 mean"]["base_geom_at_10"])
    assert "Phase 2 C6 (extra)" in pair and pair["Phase 2 mean main configurations"]["base_geom_at_10"] == pytest.approx(0.4)
    groups = {(r["group"], r["setting"]): r for r in res["groups"]}
    assert groups[("a", "Phase 2 mean main")]["n_models"] == 6 and "m1" in groups[("a", "Phase 2 mean main")]["members"] and ("b", "Phase 2 C1") in groups
    assert groups[("b", "Phase 2 C1")]["n_models"] == 0                                         # the extra model is not a group member
    beats = {r["setting"]: r for r in res["beats"]}
    assert beats["C1"]["best_shortlisted"] == "m5" and beats["C1"]["best_excluded"] == "m11" and beats["C1"]["gap_best_excluded_minus_best_shortlisted"] > 0
    assert beats["C1"]["n_pairs"] == 36 and beats["C1"]["n_pairs_excluded_beats_shortlisted"] == 36


def test_restricted_range_shows_in_the_shortlisted_only_correlation():
    """With a hard Phase 1 cut the shortlisted models lie in a narrow range and the noise dominates their rank correlation."""
    p1, p2, short = synthetic(strength=0.15, seed=3)
    res = PT.analyse(p1, p2, short, TRANSFER, resamples=300, seed=2)
    corr = {(r["population"], r["phase1"], r["phase2"]): r for r in res["correlations"]}
    all_ = corr[("all models", "mean", "mean_main")]
    assert all_["spearman_ci_high"] - all_["spearman_ci_low"] < 1.0 and all_["spearman"] > 0.5


def test_scatter_and_cli(tmp_path):
    p1, p2, short = synthetic()
    res = PT.analyse(p1, p2, short, TRANSFER, 100, 1)
    PT.scatter(res["models"], res["correlations"], tmp_path / "fig")
    assert (tmp_path / "fig.png").stat().st_size > 5000 and (tmp_path / "fig.pdf").exists()
    PT.scatter([], [], tmp_path / "empty")                                                  # nothing to plot is not an error
    assert (tmp_path / "empty.png").exists()


def test_a_pair_whose_base_is_not_a_phase1_model_keeps_the_phase1_columns_empty():
    """bert-base-uncased (base of the domain-pretraining pairs) has Phase 2 runs but no Phase 1 values: the pair table must still be written, Phase 1 settings NaN."""
    p1, p2, short = synthetic()
    tr = dict(TRANSFER, pairs=[{"id": "P5a", "base": "bert-base-uncased", "tuned": ["m3"], "note": "n"}])
    for c in CFGS:
        p2[("bert-base-uncased", c["chunker"], c["size"], c["overlap"])] = {"geom_at_10": 0.2, "ndcg_at_10": 0.15, "recall_at_10": 0.25}
    res = PT.analyse(p1, p2, short, tr, resamples=50, seed=1)
    rows = [r for r in res["pairs"] if r["pair"] == "P5a"]
    p2c1 = next(r for r in rows if r["setting"] == "Phase 2 C1")
    assert p2c1["base_geom_at_10"] == pytest.approx(0.2) and p2c1["delta_geom_at_10"] == pytest.approx(p2c1["tuned_geom_at_10"] - 0.2)
    ph1 = next(r for r in rows if r["setting"] == "Phase 1 mean")
    assert np.isnan(ph1["base_geom_at_10"]) and np.isnan(ph1["delta_geom_at_10"]) and not np.isnan(ph1["tuned_geom_at_10"])
    assert "bert-base-uncased" not in {m["model"] for m in res["models"]}                  # not part of the Phase 1 -> Phase 2 correlation

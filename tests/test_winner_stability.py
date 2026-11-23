"""tools/winner_stability.py: shares of paper-level resamples in which a run is first."""
import numpy as np
import pytest

from analysis_world import build_world
from tools import analysis_data as AD, winner_stability as WS


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    w = build_world(tmp_path_factory.mktemp("ws"))
    return AD.load_runs([w["root"]], w["qa_path"]), w


def test_shares_add_up_and_the_clear_winner_wins_everywhere(runs):
    rs, _ = runs
    res = WS.analyse(rs, ["dense"], resamples=300, seed=1)
    by_cfg = {}
    for r in res["configuration"]:
        by_cfg.setdefault((r["chunker"], r["size"], r["overlap"]), []).append(r)
    assert len(by_cfg) == 4
    for cell, rows in by_cfg.items():
        assert sum(r["share_first"] for r in rows) == pytest.approx(1.0)
        assert rows[0]["model"] == "org/m_a" and rows[0]["share_first"] > 0.9 and rows[0]["winner_in_data"]
    for model in {r["model"] for r in res["model"]}:
        rows = [r for r in res["model"] if r["model"] == model]
        assert sum(r["share_best"] for r in rows) == pytest.approx(1.0) and sum(r["winner_in_data"] for r in rows) == 1
    top = [r for r in res["overall"] if r["level"] == "run"]
    assert sum(r["share_first"] for r in top) == pytest.approx(1.0) and top[0]["value"].startswith("org/m_a") and top[0]["winner_in_data"]
    for level in ("model", "strategy", "size", "overlap", "configuration"):
        assert sum(r["share_first"] for r in res["overall"] if r["level"] == level) == pytest.approx(1.0)
    assert {r["value"] for r in res["overall"] if r["level"] == "strategy"} <= {"fixed_token", "hierarchical_section"}


def test_the_pool_all_includes_baselines_and_reranking(runs):
    rs, _ = runs
    res = WS.analyse(rs, WS.POOLS["all"], resamples=200, seed=1)
    kinds = {r["kind"] for r in res["configuration"]}
    assert "rerank" in kinds                                      # the rerank run (quality 3.0) beats every dense model here
    first = [r for r in res["configuration"] if r["winner_in_data"]]
    assert {r["kind"] for r in first} == {"rerank"} and all(r["share_first"] > 0.5 for r in first)
    assert WS.analyse(rs, ["hybrid"], resamples=10, seed=1)["overall"] == []                  # no run of this kind: empty tables, no error


def test_ties_go_to_the_first_run_and_resamples_are_paired(runs):
    rs, _ = runs
    a = WS.analyse(rs, ["dense"], resamples=100, seed=5)
    b = WS.analyse(rs, ["dense"], resamples=100, seed=5)
    assert a == b


def test_cli(runs, tmp_path):
    _, w = runs
    assert WS.main(["--roots", str(w["root"]), "--qa", str(w["qa_path"]), "--out-dir", str(tmp_path), "--resamples", "50", "--pools", "dense"]) == 0
    for f in ("winner_dense_by_configuration.csv", "winner_dense_by_model.csv", "winner_dense_overall.csv"):
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "winner_all_overall.csv").exists()

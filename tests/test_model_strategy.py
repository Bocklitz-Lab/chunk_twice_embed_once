"""tools/model_strategy.py: mean (not maximum) Geom@10 per model x strategy with paper-level intervals, winner shares, table and figure."""
import json

import numpy as np
import pytest

from analysis_world import build_world
from tools import analysis_data as AD, model_strategy as MS, winner_stability


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("ms"))


def test_cells_are_means_over_the_runs_with_ordered_intervals_and_shares(world):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    rows = MS.analyse(runs, resamples=200, seed=3)
    by = {(r["model"], r["strategy"]): r for r in rows}
    assert len(rows) == 3 * 2 and all(r["n_runs"] == 2 for r in rows)                            # 3 dense models x 2 strategies, 2 sizes each
    g = {runs.meta[i]["key"]: v for i, v in zip(runs.select(kind="dense"), runs.geom(runs.select(kind="dense")))}
    for (m, s), r in by.items():
        vals = [v for k, v in g.items() if k.startswith(f"{m}|{s}|")]
        assert r["mean_geom_at_10"] == pytest.approx(np.mean(vals)) and r["mean_geom_at_10"] != pytest.approx(max(vals))      # the mean, not the maximum
        assert r["ci_low"] <= r["mean_geom_at_10"] <= r["ci_high"]
    for s in ("fixed_token", "hierarchical_section"):
        assert sum(r["share_best_model_in_strategy"] for r in rows if r["strategy"] == s) == pytest.approx(1.0)
    assert sum(r["winner_share_overall"] for r in rows) == pytest.approx(1.0)                  # the best run of all is in exactly one cell
    assert min(r["rank_in_strategy"] for r in rows) == 1
    # the winner shares are those of winner_stability (same resamples)
    top = next(r for r in winner_stability.analyse(runs, ["dense"], resamples=200, seed=3)["overall"] if r["level"] == "run")
    cell = by[(top["value"].split(" | ")[0], top["value"].split(" | ")[1].split(" ")[0])]
    assert cell["winner_share_overall"] >= top["share_first"] - 1e-12


def test_run_writes_table_and_figure_with_sidecar(world, tmp_path):
    res = MS.run([world["root"]], world["qa_path"], tmp_path, resamples=100, seed=1)
    assert res["rows"] == 6 and (tmp_path / "table_model_strategy.csv").exists()
    for ext in ("png", "pdf", "json"):
        assert (tmp_path / f"fig_model_strategy.{ext}").exists()
    side = json.loads((tmp_path / "fig_model_strategy.json").read_text())
    assert side["resamples"] == 100 and side["n_dense_runs"] == 12 and "not the maximum" in side["cell"]

"""tools/figures.py: panels follow the data, the threshold is the 99th percentile, bands contain the estimate, sizes with one overlap get no panel, the baseline layer needs the flag."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from analysis_world import build_world
from tools import analysis_data as AD, figures as F

CONFIGS = [(c, s, o) for c in ("fixed_token", "hierarchical_section") for s, o in ((128, 0), (128, 16), (256, 0), (256, 16), (512, 0))]


def table(path, models=("org/m_a", "org/m_b", "org/m_c"), with_base=True):
    rng = np.random.default_rng(0)
    rows = []
    for m in models:
        for c, s, o in CONFIGS:
            rows.append({"model": m, "kind": "dense", "chunker": c, "size": s, "overlap": o, "span_geom_at_10": 0.5 + 0.1 * rng.random(), "evaluation_time_s": 10 + 5 * rng.random()})
    if with_base:
        for c, s, o in CONFIGS:
            rows.append({"model": "bm25", "kind": "bm25", "chunker": c, "size": s, "overlap": o, "span_geom_at_10": 0.9, "evaluation_time_s": 2})
            rows.append({"model": "hybrid_rrf60_org_m_a_bm25", "kind": "hybrid", "chunker": c, "size": s, "overlap": o, "span_geom_at_10": 0.95, "evaluation_time_s": 14})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def sidecar(out, name):
    return json.loads((out / f"{name}.json").read_text())


def test_panels_follow_the_models_and_strategies_in_the_data(tmp_path):
    runs = table(tmp_path / "r.csv", models=("org/m_a", "org/m_b", "org/m_c", "org/m_d"))
    F.draw_all(runs, tmp_path / "o")
    s = sidecar(tmp_path / "o", "fig_tradeoff_by_model")
    assert s["panels"] == ["org/m_a", "org/m_b", "org/m_c", "org/m_d"] and s["strategies"] == ["fixed_token", "hierarchical_section"]
    assert sidecar(tmp_path / "o", "fig_tradeoff_by_strategy")["panels"] == ["fixed_token", "hierarchical_section"]
    for ext in ("png", "pdf", "json"):
        assert (tmp_path / "o" / f"fig_tradeoff_by_model.{ext}").exists()
    assert s["input_sha256"] == F.sha256(runs) and s["n_runs_in_table"] == 4 * 10 + 20 and s["git_commit"]


def test_threshold_is_the_99th_percentile_of_the_dense_runs_unless_asked(tmp_path):
    runs = table(tmp_path / "r.csv")
    F.draw_all(runs, tmp_path / "o")
    rows, _ = F.read_runs(runs)
    dense = [r["geom"] for r in rows if r["kind"] == "dense"]
    s = sidecar(tmp_path / "o", "fig_tradeoff_by_model")
    assert s["threshold_value"] == pytest.approx(np.percentile(dense, 99)) and s["threshold_over"] == "dense" and s["threshold_n_runs"] == len(dense)
    assert s["kinds_included"] == ["dense"]
    F.draw_all(runs, tmp_path / "o2", include_baselines=True, threshold_over="all")
    s2 = sidecar(tmp_path / "o2", "fig_tradeoff_by_model")
    assert s2["threshold_over"] == "all" and s2["threshold_value"] > s["threshold_value"] and set(s2["threshold_kinds"]) == {"dense", "bm25", "hybrid"}
    F.draw_all(runs, tmp_path / "o3", include_baselines=True)                # baselines drawn, threshold still over the dense runs
    s3 = sidecar(tmp_path / "o3", "fig_tradeoff_by_model")
    assert s3["threshold_value"] == pytest.approx(s["threshold_value"]) and s3["kinds_included"] == ["dense", "bm25", "hybrid", "rerank"]


def test_overlap_panels_skip_sizes_with_one_overlap_and_bands_need_per_query_files(tmp_path):
    F.draw_all(table(tmp_path / "r.csv"), tmp_path / "o")
    s = sidecar(tmp_path / "o", "fig_overlap_by_size")
    assert s["panels"] == ["128", "256"] and s["sizes_without_panel"] == {"512": [0]}
    assert s["band"].startswith("none")
    assert sidecar(tmp_path / "o", "fig_chunk_size_by_strategy")["band"].startswith("none")


def test_baseline_layer_only_with_the_flag(tmp_path):
    runs = table(tmp_path / "r.csv")
    F.draw_all(runs, tmp_path / "a")
    F.draw_all(runs, tmp_path / "b", include_baselines=True)
    assert sidecar(tmp_path / "a", "fig_chunk_size_by_strategy")["baseline_layer"] is False and "bm25" not in sidecar(tmp_path / "a", "fig_chunk_size_by_strategy")["kinds_included"]
    assert sidecar(tmp_path / "b", "fig_chunk_size_by_strategy")["kinds_included"] == ["dense", "bm25"]
    # effect series: dense only by default
    rows, _ = F.read_runs(runs)
    ser = F.effect_series([r for r in rows if r["overlap"] == 0], None, lambda r: r["chunker"], lambda r: r["size"], ("dense",))
    assert set(ser) == {"fixed_token", "hierarchical_section"} and [p["x"] for p in ser["fixed_token"]] == [128, 256, 512]
    assert all(p["value"] < 0.7 for pts in ser.values() for p in pts)           # BM25 (0.9) is not in the mean


def test_families_come_from_the_config_not_from_names(tmp_path):
    rows = [{"model": m, "kind": "dense"} for m in ("intfloat/e5-large-v2", "BAAI/bge-m3", "x/unknown")]
    assert F.families(rows) == {"BAAI/bge-m3": "BGE", "intfloat/e5-large-v2": "E5", "x/unknown": "Other"}


def test_palette_never_exceeds_six_colours():
    assert len(set(F.palette(list("abcdef")).values())) == 6
    with pytest.raises(ValueError):
        F.palette(list("abcdefg"))


def test_missing_columns_are_an_error(tmp_path):
    (tmp_path / "r.csv").write_text("model,chunker,size,overlap\na,fixed_token,128,0\n")
    with pytest.raises(ValueError, match="needs a Geom@10 column"):
        F.read_runs(tmp_path / "r.csv")


# ---- bands from per-query files --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("fig"), configs=[("fixed_token", 128, 0), ("fixed_token", 128, 16), ("fixed_token", 512, 0)], with_rerank=False)


def overview_of(world, path):
    runs = AD.load_runs([world["root"]], world["qa_path"])
    rows = [{"model": m["model"], "kind": m["kind"], "chunker": m["chunker"], "size": m["size"], "overlap": m["overlap"],
             "span_geom_at_10": float(g), "evaluation_time_s": 5.0 + i} for i, (m, g) in enumerate(zip(runs.meta, runs.geom()))]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return runs


def test_bands_contain_the_point_estimate_and_it_is_the_mean_of_the_run_geoms(world, tmp_path):
    runs = overview_of(world, tmp_path / "r.csv")
    b = F.Bands([world["root"]], world["qa_path"], resamples=300, seed=3, kinds=["dense"])
    keys = [(m["model"], "fixed_token", 128, 0) for m in runs.meta if m["kind"] == "dense" and m["chunker"] == "fixed_token" and m["size"] == 128 and m["overlap"] == 0]
    got = b.band(keys)
    idx = [runs.index[f"{k[0]}|fixed_token|128|0"] for k in keys]
    assert got["value"] == pytest.approx(runs.geom(idx).mean()) and got["ci_low"] <= got["value"] <= got["ci_high"] and got["n_runs"] == len(keys)
    # the resampled geom of a run is recomputed from its nDCG and Recall means, with the same papers for every run
    g = b.geom_resamples()
    nd, rc = b.pb.resample_means(b.runs.metrics["ndcg_at_10"][:1]), b.pb.resample_means(b.runs.metrics["recall_at_10"][:1])
    assert np.allclose(g[:, 0], np.sqrt(nd[:, 0] * rc[:, 0]))
    F.draw_all(tmp_path / "r.csv", tmp_path / "o", roots=[world["root"]], qa=world["qa_path"], resamples=100, seed=1)
    s = sidecar(tmp_path / "o", "fig_chunk_size_by_strategy")
    assert s["band"].startswith("95 % paper-level") and s["bootstrap_resamples"] == 100
    assert sidecar(tmp_path / "o", "fig_overlap_by_size")["panels"] == ["128"] and sidecar(tmp_path / "o", "fig_overlap_by_size")["sizes_without_panel"] == {"512": [0]}
    ser = F.effect_series([r for r in F.read_runs(tmp_path / "r.csv")[0] if r["overlap"] == 0], b, lambda r: r["chunker"], lambda r: r["size"], ("dense",))
    assert all(p["ci_low"] <= p["value"] <= p["ci_high"] for p in ser["fixed_token"])


def test_published_adapter(tmp_path):
    tests = tmp_path / "tests"
    (tests / "cfg1").mkdir(parents=True)
    (tests / "cfg1" / "ChemQuest.json").write_text(json.dumps({"evaluation_time": 12.5}))
    (tmp_path / "c.csv").write_text("configuration_id,model_hf,chunker,token_size,overlap,geom_at_10,result_file\n"
                                    "cfg1,org/m,fixed_token,128,0,0.6,artifacts/tests/cfg1/ChemQuest.json\ncfg2,org/m,fixed_token,256,0,0.7,artifacts/tests/cfg2/ChemQuest.json\n")
    out = F.adapt_published(tmp_path / "c.csv", tests, tmp_path / "t.csv")
    rows, used = F.read_runs(out)
    assert len(rows) == 1 and rows[0]["time"] == 12.5 and rows[0]["kind"] == "dense" and used["time_column"] == "evaluation_time_s"


def test_panel_grid_grows_for_13_or_more_models():
    from tools import figures as F
    assert F.panel_columns(4, 4) == 4 and F.panel_columns(12, 4) == 4 and F.panel_columns(13, 4) == 5 and F.panel_columns(18, 4) == 5


def test_legend_below_does_not_cover_the_axes():
    """Audit bug 2: the legend of a short one-row figure covered the x labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools import figures as F
    fig, axes = plt.subplots(1, 4, figsize=(F.FULL_WIDTH_MM * F.MM, 55 * F.MM))
    for ax in axes:
        ax.set_xlabel("evaluation time (s, log)")
    handles = [plt.Line2D([], [], marker="o", ls="", label=f"label number {i}") for i in range(10)]
    F.legend_below(fig, handles, ncol=5)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    leg = fig.legends[0].get_window_extent(r)
    lowest = min(ax.xaxis.label.get_window_extent(r).y0 for ax in axes)
    assert leg.y1 <= lowest + 1
    plt.close(fig)

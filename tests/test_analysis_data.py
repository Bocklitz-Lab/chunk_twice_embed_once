"""tools/analysis_data.py: the shared loader and the paper-level resampling of all runs at once."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis_world import MODELS, build_world
from tools import analysis_data as AD, stats


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("world"))


def test_load_runs_reads_every_kind_and_the_paper_index(world):
    rs = AD.load_runs([world["root"]], world["qa_path"])
    n_cfg, n_models = 4, len(MODELS)
    assert len(rs) == n_cfg * (n_models + 1)                                  # + the rerank run of every configuration
    assert rs.metrics["ndcg_at_10"].shape == (len(rs), 36) and rs.qids == [str(i) for i in range(36)]
    assert {m["kind"] for m in rs.meta} == {"dense", "bm25", "rerank"}
    assert rs.groups[0] == "1" and rs.groups[3] == "2" and len(set(rs.groups)) == 12            # zero-padded ids ("001") are normalised
    assert [m["source_dense"] for m in rs.meta if m["kind"] == "rerank"] == ["org/m_c"] * n_cfg
    assert rs.select(kind="dense", size=512) and all(rs.meta[i]["size"] == 512 for i in rs.select(size=512))
    assert len(rs.select(kind=["dense", "bm25"])) == n_cfg * n_models


def test_geom_is_sqrt_of_the_product_of_the_means(world):
    rs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    g = rs.geom()
    assert np.allclose(g, np.sqrt(rs.metrics["ndcg_at_10"].mean(axis=1) * rs.metrics["recall_at_10"].mean(axis=1)))
    assert np.allclose(rs.geom(rs.select(model="org/m_a")), g[rs.select(model="org/m_a")])


def test_first_root_wins_and_missing_roots_are_ignored(tmp_path, world):
    other = build_world(tmp_path / "second", models={"org/m_a": 0.0}, seed=5, with_rerank=False)
    a = AD.load_runs([world["root"], other["root"], tmp_path / "nothing"], world["qa_path"], kinds=["dense"])
    b = AD.load_runs([other["root"], world["root"]], world["qa_path"], kinds=["dense"])
    i, j = a.index["org/m_a|fixed_token|128|0"], b.index["org/m_a|fixed_token|128|0"]
    assert a.meta[i]["root"] == str(world["root"]) and b.meta[j]["root"] == str(other["root"])
    assert not np.allclose(a.metrics["ndcg_at_10"][i], b.metrics["ndcg_at_10"][j])
    with pytest.raises(FileNotFoundError):
        AD.load_runs([tmp_path / "nothing"], world["qa_path"])


def test_runs_with_different_queries_are_refused(tmp_path):
    w = build_world(tmp_path / "w", configs=[("fixed_token", 128, 0)], models={"org/m_a": 1.0}, with_rerank=False)
    f = next((w["root"] / "tests").glob("*/results/*/per_query_metrics.csv"))
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:-1]) + "\n")
    w2 = build_world(tmp_path / "w2", configs=[("fixed_token", 512, 0)], models={"org/m_a": 1.0}, with_rerank=False)
    with pytest.raises(ValueError, match="same queries"):
        AD.load_runs([w["root"], w2["root"]], w["qa_path"])


def test_overview_metadata_is_joined(tmp_path, world):
    ov = tmp_path / "runs_overview.csv"
    ov.write_text("model,chunker,size,overlap,n_chunks\norg/m_a,fixed_token,128,0,999\n")
    rs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"], overview=ov)
    assert rs.meta[rs.index["org/m_a|fixed_token|128|0"]]["n_chunks"] == "999"


def test_paper_bootstrap_equals_stats_for_the_same_seed(world):
    rs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    boot = AD.PaperBootstrap(rs.groups, 200, 7)
    nd, rc = rs.metrics["ndcg_at_10"], rs.metrics["recall_at_10"]
    mine = boot.resample_means(nd[:3])
    for r in range(3):
        bn, br = stats.resampled_means(nd[r], rc[r], rs.groups, 200, 7)
        assert np.allclose(mine[:, r], bn)
    g = boot.geom_resamples(nd, rc)
    bn, br = stats.resampled_means(nd[4], rc[4], rs.groups, 200, 7)
    assert g.shape == (200, len(rs)) and np.allclose(g[:, 4], np.sqrt(bn * br))
    lo, hi = boot.percentile_ci(g)
    assert (lo <= hi).all() and (lo <= rs.geom() + 0.2).all()
    assert boot.counts.sum(axis=1).min() == boot.G and boot.n_questions.min() > 0


def test_defaults_and_common_args_come_from_the_config():
    d = AD.defaults()
    assert d["resamples"] == 10000 and d["seed"] == stats.DEFAULT_SEED and d["cover_level"] == 0.9
    import argparse
    ap = argparse.ArgumentParser()
    AD.add_common_args(ap)
    a = ap.parse_args(["--roots", "r", "--qa", "q", "--out-dir", "o"])
    assert AD.resolve(a) == (10000, stats.DEFAULT_SEED)
    assert AD.resolve(ap.parse_args(["--roots", "r", "--qa", "q", "--out-dir", "o", "--resamples", "5", "--seed", "3"])) == (5, 3)


def test_write_rows_keeps_first_seen_column_order(tmp_path):
    p = AD.write_rows(tmp_path / "x.csv", [{"a": 1, "b": 0.123456789}, {"a": 2, "c": "z"}])
    assert p.read_text().splitlines() == ["a,b,c", "1,0.123457,", "2,,z"]

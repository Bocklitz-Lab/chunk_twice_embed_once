"""Phase 2 model selection (stage 2): the inclusion-probability rule widens the published top-fraction rule, never narrows it, and the published rule still works."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline_lib import phase1_per_query as PQ
from pipeline_lib import shortlist as SL
from stages import screen_top_k_models as S2
from tools import phase1_intervals as PI
from tests.test_phase1_per_query import FakeTask, NQ, TASK, add_task, make_model

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "temp_files/artifacts_old/embedding_models_screening/validation"
PROMPTED = ROOT / "artifacts/subset_v1/embedding_models_screening/validation_prompts"


def _world(tmp_path, n=10, t2="ChemHotpotQARetrieval"):
    root = tmp_path / "validation"
    for m in range(n):
        out = make_model(root, f"org__m{m}", [1 if i < NQ - m else 11 for i in range(NQ)])
        add_task(out, f"org__m{m}", [1 if i < m + 2 else 11 for i in range(NQ)], task=t2)      # the second task reverses the order
        PQ.write_task(out, FakeTask(), {TASK: "test"})
        PQ.write_task(out, FakeTask(t2), {t2: "test"})
    return root, {TASK: "test", t2: "test"}


def _table(tmp_path, primary=0.5, resamples=400):
    root, split = _world(tmp_path)
    rows = PI.analyse([root], split, top_percent=0.2, resamples=resamples, seed=7)
    return SL.table_from_intervals(rows, primary), rows


def test_observed_top_models_are_always_selected_and_thresholds_are_monotone(tmp_path):
    t, rows = _table(tmp_path)
    obs = {r["model"] for r in t if r["observed_top"]}
    assert obs == {"org__m0", "org__m1", "org__m8", "org__m9"}
    for k in ("selected_p0.4", "selected_p0.5", "selected_p0.6", "selected_primary"):
        assert obs <= {r["model"] for r in t if r[k]}
    n = SL.counts(t, 0.5)
    assert n["selected_top_fraction"] == 4 <= n["selected_p0.6"] <= n["selected_p0.5"] <= n["selected_p0.4"]
    assert {r["model"] for r in SL.select(t)} == {r["model"] for r in t if r["selected_p0.5"]}


def test_a_lower_threshold_can_only_add_models(tmp_path):
    t, _ = _table(tmp_path, primary=0.05)
    assert len(SL.select(t)) > 4                      # models next to the cut have a small but non-zero probability
    assert {r["model"] for r in t if r["selected_p0.4"]} <= {r["model"] for r in SL.select(t)}


def test_validate_selection_rejects_bad_values():
    assert SL.validate_selection(None)["rule"] == "top_fraction"
    with pytest.raises(SL.ShortlistError):
        SL.validate_selection({"rule": "best_five"})
    with pytest.raises(SL.ShortlistError):
        SL.validate_selection({"rule": "inclusion_probability", "min_inclusion_probability": 0})


def _stage2_cfg(tmp_path, root_dir, sub, rule, **extra):
    cfg = {"root_dir": str(root_dir), "results_subdir": sub, "eval_split": {"ChemNQRetrieval": "test", "ChemHotpotQARetrieval": "test"}, "n_queries": {},
           "metric_key": "geom_mean", "geom_mean": {"k": 10, "weights": {"ndcg": 1, "recall": 1, "time": 0}}, "top_percent": 0.2, "model_id_format": "name@rev",
           "selection": {"rule": rule, "top_fraction": 0.2, "min_inclusion_probability": 0.5, "resamples": 300, "max_models": 18, **extra}, "seed": extra.get("seed", 1),
           "output_union_csv": str(tmp_path / "u.csv"), "output_per_dataset_csv": str(tmp_path / "p.csv"), "output_selection_csv": str(tmp_path / "sel.csv")}
    p = tmp_path / "s2.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run(monkeypatch, cfg):
    monkeypatch.setattr(sys, "argv", ["stage2", "--config", str(cfg)])
    S2.main()


@pytest.mark.skipif(not PUBLISHED.is_dir(), reason="published Phase 1 files not in this checkout")
def test_top_fraction_reproduces_the_published_13(tmp_path, monkeypatch):
    cfg = _stage2_cfg(tmp_path, PUBLISHED.parent, PUBLISHED.name, "top_fraction")
    _run(monkeypatch, cfg)
    names = set(pd.read_csv(tmp_path / "p.csv")["model_name"])
    assert len(names) == 13 and {"hkunlp/instructor-xl", "BAAI/bge-m3", "intfloat/e5-large"} <= names and "nomic-ai/nomic-embed-text-v2-moe" not in names
    assert not (tmp_path / "sel.csv").exists()                  # the published files have no per-query files: no probabilities


@pytest.mark.skipif(not PROMPTED.is_dir(), reason="prompted Phase 1 files not in this checkout")
def test_prompted_files_give_the_prompted_13_and_the_rule_never_drops_the_observed_top(tmp_path, monkeypatch):
    for rule in ("top_fraction", "inclusion_probability"):
        cfg = _stage2_cfg(tmp_path, PROMPTED.parent, PROMPTED.name, rule, resamples=10000, seed=20261121)
        _run(monkeypatch, cfg)
        names = set(pd.read_csv(tmp_path / "p.csv")["model_name"])
        assert len(names) == 13 and "nomic-ai/nomic-embed-text-v2-moe" in names and "BAAI/bge-m3" not in names
    sel = pd.read_csv(tmp_path / "sel.csv")
    assert set(sel[sel.observed_top]["model"]) <= names
    assert {"selected_top_fraction", "selected_p0.4", "selected_p0.5", "selected_p0.6", "selected_primary"} <= set(sel.columns)
    assert sel["selected_p0.4"].sum() >= sel["selected_p0.5"].sum() >= sel["selected_p0.6"].sum() >= sel["selected_top_fraction"].sum() == 13


def test_stage2_stops_when_the_rule_selects_too_many(tmp_path, monkeypatch):
    root, split = _world(tmp_path)
    # leaf layout of stage 1: <results>/<folder>/<model>/<rev>/<Task>.json with model_meta.json (written by make_model)
    cfg = _stage2_cfg(tmp_path, tmp_path, "validation", "inclusion_probability", min_inclusion_probability=0.001, max_models=5)
    with pytest.raises(SystemExit, match="STOP"):
        _run(monkeypatch, cfg)
    assert (tmp_path / "sel.csv").exists() and not (tmp_path / "p.csv").exists()

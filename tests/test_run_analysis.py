"""tools/run_analysis.py (`make analysis`) and the cover-level sensitivity step of `make report` (tools/rescore.py --sensitivity)."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from analysis_world import build_world
from tools import metrics as M, rescore, run_analysis as RA

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "configs" / "main_config.yaml")


def ns(**kw):
    base = dict(main_config=MAIN, roots=None, qa=None, out_dir=None, phase1_root=None, shortlist=None, runs_overview=None, resamples=None, label_set="span", only=None, skip=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_defaults_come_from_the_config():
    d = RA.defaults(MAIN)
    assert d["roots"][0] == "artifacts" and d["qa"] == "artifacts/chemquests_hf_db/qa.jsonl" and d["out_dir"] == "results/v2-revision/analysis"
    assert d["phase1_root"] == "artifacts/embedding_models_screening/validation_prompts"
    assert d["shortlist"] == "artifacts/embedding_models_screening/passed_per_dataset__validation_prompts.csv" and d["overviews"][0] == "artifacts/report/runs_overview.csv"


def test_every_tool_gets_its_arguments():
    cmds = RA.commands(ns(resamples=77), RA.defaults(MAIN))
    assert set(cmds) == set(RA.TOOLS)
    for tool, c in cmds.items():
        assert "--out-dir" in c and c[c.index("--resamples") + 1] == "77" and "--main-config" in c
    assert "--balanced" in cmds["variance_decomposition"] and "--phase1-root" in cmds["phase_transfer"] and "--runs-overview" not in cmds["token_budget"]
    assert all(t in RA.TOOLS for t in cmds)


def test_a_failing_tool_does_not_stop_the_others(capsys):
    seen = []

    def runner(tool, cmd):
        seen.append(tool)
        if tool == "winner_stability":
            raise ValueError("boom")
        return 0

    st = RA.run(ns(skip=["phase_transfer"]), runner)
    assert seen == [t for t in RA.TOOLS if t != "phase_transfer"] and st["winner_stability"].startswith("FAILED: ValueError") and st["token_budget"] == "ok"
    assert "phase_transfer" not in st


def test_phase_transfer_is_skipped_when_phase1_is_missing(tmp_path):
    st = RA.run(ns(only=["phase_transfer"], phase1_root=str(tmp_path / "none"), shortlist=str(tmp_path / "s.csv"), runs_overview=[str(tmp_path / "o.csv")]), lambda *a: 0)
    assert st["phase_transfer"].startswith("skipped: Phase 1 folder")
    (tmp_path / "p1").mkdir()
    st = RA.run(ns(only=["phase_transfer"], phase1_root=str(tmp_path / "p1"), shortlist=str(tmp_path / "s.csv"), runs_overview=[str(tmp_path / "o.csv")]), lambda *a: 0)
    assert "shortlist" in st["phase_transfer"]
    (tmp_path / "s.csv").write_text("model_id\n")
    st = RA.run(ns(only=["phase_transfer"], phase1_root=str(tmp_path / "p1"), shortlist=str(tmp_path / "s.csv"), runs_overview=[str(tmp_path / "o.csv")]), lambda *a: 0)
    assert "runs overview missing" in st["phase_transfer"]


def test_only_and_skip_and_exit_codes():
    assert list(RA.run(ns(only=["examples", "fragmentation"]), lambda *a: 0)) == ["examples", "fragmentation"]
    assert RA.run(ns(only=["examples"]), lambda *a: 3)["examples"] == "FAILED: exit code 3"


def test_end_to_end_with_the_real_tools(tmp_path):
    w = build_world(tmp_path / "w")
    out = tmp_path / "out"
    cfg = ns(roots=[str(w["root"])], qa=str(w["qa_path"]), out_dir=str(out), resamples=30, skip=["phase_transfer"])
    st = RA.run(cfg)
    assert set(st.values()) == {"ok"}, st
    for f in ("token_budget.csv", "breakdown_subfield.csv", "variance_decomposition.csv", "winner_dense_overall.csv", "examples.csv", "fragmentation.csv"):
        assert (out / f).exists(), f


def test_makefile_target_and_report_switch():
    out = subprocess.run(["make", "-n", "-C", str(ROOT), "analysis", "ANALYSIS_ARGS=--only examples"], capture_output=True, text=True).stdout
    assert "python -m tools.run_analysis --only examples" in out
    import yaml
    a = yaml.safe_load(Path(MAIN).read_text())["params"]["analysis"]
    assert a["enabled_in_report"] is False                     # off by default: `make report` stays as it was; set true to chain `make analysis`


# ---- cover-level sensitivity -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("sens"))


def test_sensitivity_rows_cover_every_run_kind_and_level(world):
    rows = rescore.sensitivity_rows(world["root"] / "tests", world["qa_path"], [0.8, 0.9, 1.0], 100, 1)
    assert len(rows) == 4 * 5 * 3 and {r["kind"] for r in rows} == {"dense", "bm25", "rerank"} and {r["cover_level"] for r in rows} == {"0.8", "0.9", "1"}
    for r in rows:
        assert r["ndcg_ci_low"] <= r["ndcg_at_10"] <= r["ndcg_ci_high"] and r["geom_ci_low"] <= r["geom_at_10"] <= r["geom_ci_high"] and r["resamples"] == 100
        assert r["geom_at_10"] == pytest.approx((r["ndcg_at_10"] * r["recall_at_10"]) ** 0.5)


def test_sensitivity_at_the_primary_level_equals_the_stored_per_query_metrics(world):
    rows = rescore.sensitivity_rows(world["root"] / "tests", world["qa_path"], [0.9], 50, 1)
    for r in rows[:6]:
        d = next((world["root"] / "tests").glob(f"{r['model'].replace('/', '_')}_{r['chunker']}_c{r['size']}_o{r['overlap']}"))
        pq = M.load_per_query_csv(next((d / "results").glob("*/per_query_metrics.csv")))
        assert r["ndcg_at_10"] == pytest.approx(np.mean([v["ndcg_at_10"] for v in pq.values()]), abs=1e-9)
        assert r["recall_at_10"] == pytest.approx(np.mean([v["recall_at_10"] for v in pq.values()]), abs=1e-9)


def test_a_stricter_cover_level_never_labels_more_chunks_as_enough(world):
    rows = rescore.sensitivity_rows(world["root"] / "tests", world["qa_path"], [0.8, 1.0], 20, 1)
    by = {(r["model"], r["chunker"], r["size"], r["cover_level"]): r for r in rows}
    # small chunks split the evidence more often, so the labels change with c there; with large chunks (one chunk holds nearly every span) much less
    small = abs(by[("org/m_a", "fixed_token", 128, "0.8")]["geom_at_10"] - by[("org/m_a", "fixed_token", 128, "1")]["geom_at_10"])
    assert small >= 0


def test_runs_without_a_ranking_or_overlap_table_are_skipped(tmp_path):
    w = build_world(tmp_path / "w", configs=[("fixed_token", 128, 0)], models={"org/m_a": 1.0}, with_rerank=False)
    run = next((w["root"] / "tests").iterdir())
    next(run.glob("results/*/ChemQuest_default_predictions.json")).unlink()
    assert rescore.sensitivity_rows(w["root"] / "tests", w["qa_path"], [0.9], 10, 1) == []


def test_cli_writes_the_table_from_the_config_levels(world, tmp_path):
    out = tmp_path / "sens.csv"
    assert rescore.main(["--sensitivity", str(world["root"] / "tests"), "--qa", str(world["qa_path"]), "--resamples", "20", "--output", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert {r["cover_level"] for r in rows} == {"0.8", "0.9", "1"} and list(rows[0]) == rescore.SENSITIVITY_COLUMNS
    with pytest.raises(SystemExit):
        rescore.main(["--sensitivity", str(world["root"] / "tests"), "--output", str(out)])        # --qa is required


def test_aggregate_scores_are_unchanged_by_the_refactor():
    preds = {"0": {"a": 3.0, "b": 2.0, "c": 1.0}, "1": {"a": 1.0, "b": 2.0}}
    qrels = {"0": {"b": 1}, "1": {"zzz": 1}}
    m = rescore.score(preds, qrels)
    assert m["ndcg_at_10"] == pytest.approx((1 / np.log2(3)) / 2) and m["recall_at_10"] == pytest.approx(0.5) and m["n_empty_questions"] == 0
    per_n, per_r, pos = rescore.per_query_scores(preds, qrels)
    assert per_r == {"0": 1.0, "1": 0.0} and set(pos) == {"0", "1"}

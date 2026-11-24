"""Phase 1 per-query results: derived from the saved predictions, checked against the MTEB scores, and the table with query-level intervals."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline_lib import phase1_per_query as PQ
from pipeline_lib.phase1_split import Phase1SplitError
from tools import phase1_intervals as PI

TASK = "ChemNQRetrieval"
NQ = 12


class FakeTask:
    def __init__(self, name=TASK, splits=("test",)):
        self.metadata = type("M", (), {"name": name})
        self.queries = {s: {f"q{i}": f"text {i}" for i in range(NQ)} for s in splits}
        self.relevant_docs = {s: {f"q{i}": {f"d{i}": 1} for i in range(NQ)} for s in splits}


def add_task(out, name, ranks, task=TASK, split="test"):
    """Predictions in which the relevant doc of query i sits at rank ranks[i], and the MTEB json of the same ranking."""
    leaf = out / name / "rev"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "model_meta.json").write_text(json.dumps({"name": name, "revision": "rev"}))
    pred, nd, rc = {}, [], []
    for i in range(NQ):
        r = ranks[i]
        docs = {f"x{i}_{k}": 100.0 - k for k in range(20)}
        docs.pop(f"x{i}_{r - 1}")
        docs[f"d{i}"] = 100.0 - (r - 1) - 0.5
        pred[f"q{i}"] = docs
        nd.append(1.0 / np.log2(r + 1) if r <= 10 else 0.0)
        rc.append(1.0 if r <= 10 else 0.0)
    (out / f"{task}_default_predictions.json").write_text(json.dumps(pred))
    (leaf / f"{task}.json").write_text(json.dumps({"task_name": task, "evaluation_time": 1.0,
                                                    "scores": {split: [{"ndcg_at_10": round(float(np.mean(nd)), 5), "recall_at_10": round(float(np.mean(rc)), 5)}]}}))


def make_model(root, name, ranks, split="test"):
    out = root / name
    add_task(out, name, ranks, split=split)
    return out


def test_per_query_files_reproduce_the_mteb_scores(tmp_path):
    out = make_model(tmp_path, "m", [1, 2, 3, 11, 1, 1, 5, 12, 2, 1, 1, 4])
    s = PQ.write_task(out, FakeTask(), {TASK: "test"})
    assert s["split"] == "test" and s["n_queries"] == NQ and all(v["abs_diff"] < PQ.TOLERANCE for v in s["mteb_check"].values())
    rows = list(csv.DictReader(open(out / f"per_query_{TASK}.csv")))
    assert len(rows) == NQ and {"ndcg_at_10", "recall_at_10", "geom_at_10"} <= set(rows[0])
    run = gzip.open(out / f"run_top100_{TASK}.trec.gz", "rt").read().splitlines()
    assert len(run) == NQ * 20 and run[0].split()[1] == "Q0"


def test_predictions_of_another_split_are_refused(tmp_path):
    out = make_model(tmp_path, "m", [1] * NQ)
    scores = json.loads(next(out.glob("**/ChemNQRetrieval.json")).read_text())
    scores["scores"]["dev"] = [{"ndcg_at_10": 0.3, "recall_at_10": 0.3}]
    next(out.glob("**/ChemNQRetrieval.json")).write_text(json.dumps(scores))
    t = FakeTask(splits=("test", "dev"))
    with pytest.raises(Phase1SplitError, match="do not reproduce the 'dev' scores"):
        PQ.write_task(out, t, {TASK: "dev"})
    with pytest.raises(Phase1SplitError, match="'train' is not in the result file"):
        PQ.write_task(out, FakeTask(splits=("test", "train")), {TASK: "train"})


def test_missing_predictions_raise(tmp_path):
    out = make_model(tmp_path, "m", [1] * NQ)
    (out / f"{TASK}_default_predictions.json").unlink()
    with pytest.raises(FileNotFoundError):
        PQ.write_task(out, FakeTask(), {TASK: "test"})


def test_stage1_reads_the_switch_and_the_split():
    from stages import model_screening as MS
    raw = {"models": [{"name": "a/b"}], "tasks": ["ChemNQRetrieval"], "save_per_query": True, "eval_split": {"ChemNQRetrieval": "test"}}
    cfg = MS.parse_app_config(raw)
    assert cfg.save_per_query is True and cfg.eval_split == {"ChemNQRetrieval": "test"}
    assert MS.parse_app_config({k: v for k, v in raw.items() if k not in ("save_per_query", "eval_split")}).save_per_query is False      # old configs: unchanged


def test_tracked_stage1_file_switches_it_on():
    import yaml
    s1 = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "stages" / "stage1.yaml").read_text())
    assert s1["save_per_query"] is True and s1["eval_split"] == {"ChemNQRetrieval": "test", "ChemHotpotQARetrieval": "test"}


# ---- intervals and shortlist stability -----------------------------------------------------------------------------
def _world(tmp_path, n_models=10):
    root = tmp_path / "validation"
    for m in range(n_models):                                       # model m: relevant docs at rank 1 for the first (NQ - m) queries, 11 (a miss) for the rest
        ranks = [1 if i < NQ - m else 11 for i in range(NQ)]
        out = make_model(root, f"org__m{m}", ranks)
        PQ.write_task(out, FakeTask(), {TASK: "test"})
    return root


def test_table_has_intervals_ranks_and_top_share(tmp_path):
    root = _world(tmp_path)
    rows = PI.analyse([root], {TASK: "test"}, top_percent=0.2, resamples=500, seed=3)
    assert len(rows) == 10 and {r["level"] for r in rows} == {"query"} and {(r["split"], r["n_queries"]) for r in rows} == {("test", NQ)}
    by = {r["model"]: r for r in rows}
    assert by["org__m0"]["rank"] == 1 and by["org__m9"]["rank"] == 10 and by["org__m0"]["in_top"] and by["org__m1"]["in_top"] and not by["org__m2"]["in_top"]
    assert all(r["ci_low"] <= r["geom_at_10"] <= r["ci_high"] for r in rows)
    assert by["org__m0"]["top_share"] > 0.9 and by["org__m9"]["top_share"] < 0.05 and 0 < by["org__m2"]["top_share"] < 0.7
    # same seed: same table
    assert rows == PI.analyse([root], {TASK: "test"}, top_percent=0.2, resamples=500, seed=3)


def test_union_rows_with_two_tasks(tmp_path):
    root, t2 = tmp_path / "validation", "ChemHotpotQARetrieval"
    for m in range(10):
        out = make_model(root, f"org__m{m}", [1 if i < NQ - m else 11 for i in range(NQ)])
        add_task(out, f"org__m{m}", [1 if i < m + 2 else 11 for i in range(NQ)], task=t2)           # second task: the order of the models is reversed
        PQ.write_task(out, FakeTask(), {TASK: "test"})
        PQ.write_task(out, FakeTask(t2), {t2: "test"})
    rows = PI.analyse([root], {TASK: "test", t2: "test"}, top_percent=0.2, resamples=300, seed=1)
    union = [r for r in rows if r["task"] == "union"]
    assert len(union) == 10 and len(rows) == 30 and sum(r["in_top"] for r in union) == 4              # top 2 of each task: m0, m1 and m8, m9
    assert union[0]["union_size_low"] <= union[0]["union_size_mean"] <= union[0]["union_size_high"] and union[0]["union_size_mean"] >= 4
    assert next(r for r in union if r["model"] == "org__m0")["top_share"] > 0.8


def test_a_per_query_file_of_another_split_stops_the_table(tmp_path):
    root = _world(tmp_path, 3)
    with pytest.raises(Phase1SplitError, match="another split"):
        PI.analyse([root], {TASK: "dev"}, top_percent=0.2, resamples=50, seed=1)


def test_no_per_query_files_gives_no_rows(tmp_path):
    assert PI.analyse([tmp_path / "nothing"], {TASK: "test"}, top_percent=0.2, resamples=50, seed=1) == []

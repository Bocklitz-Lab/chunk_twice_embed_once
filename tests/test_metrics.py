"""tools/metrics.py: hand-computed example, TREC round trip, MTEB cross-check, empty questions, run_eval outputs."""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools import metrics as M

ROOT = Path(__file__).resolve().parents[1]

QRELS = {"q1": {"a": 1, "c": 1}, "q2": {"x": 0}, "q3": {"b": 1}}
RUN = {"q1": {"a": 0.9, "b": 0.8, "c": 0.7}, "q2": {"x": 0.5, "y": 0.4}, "q3": {"a": 0.9, "c": 0.8, "b": 0.7}}


def test_hand_computed_values_and_empty_question_stays_in_the_mean():
    pq = M.per_query_metrics(RUN, QRELS, (1, 10, 100))
    ndcg1 = 1.5 / (1 + 1 / math.log2(3))
    assert pq["q1"]["ndcg_at_10"] == pytest.approx(ndcg1)
    assert pq["q1"]["map_at_10"] == pytest.approx((1 + 2 / 3) / 2)
    assert pq["q1"]["precision_at_10"] == pytest.approx(0.2) and pq["q1"]["mrr_at_10"] == 1.0
    assert pq["q3"]["ndcg_at_10"] == pytest.approx(0.5) and pq["q3"]["mrr_at_10"] == pytest.approx(1 / 3)
    assert pq["q3"]["recall_at_1"] == 0.0 and pq["q1"]["recall_at_1"] == 0.5
    assert pq["q2"]["ndcg_at_10"] == 0.0 and pq["q2"]["n_relevant"] == 0
    agg = M.aggregate(pq)
    assert agg["ndcg_at_10"] == pytest.approx((ndcg1 + 0.5) / 3)
    assert agg["recall_at_10"] == pytest.approx(2 / 3) and agg["n_empty_questions"] == 1 and agg["n_queries"] == 3
    assert agg["geom_at_10"] == pytest.approx(math.sqrt(agg["ndcg_at_10"] * agg["recall_at_10"]))


def test_missing_query_and_bad_label_raise(tmp_path):
    with pytest.raises(ValueError):
        M.per_query_metrics({"q1": RUN["q1"]}, QRELS)
    f = tmp_path / "q.jsonl"; f.write_text(json.dumps({"query_id": "1", "doc_id": "a", "relevance": 2}) + "\n")
    with pytest.raises(ValueError):
        M.load_qrels(f)


def test_trec_round_trip_depth_and_tie_break(tmp_path):
    run = {"1": {"a": 1.0, "b": 1.0, "c": 0.5, "d": 0.1}}
    p = tmp_path / "r.trec.gz"
    assert M.write_trec_run(run, p, tag="t", depth=3) == 3
    lines = gzip.open(p, "rt").read().splitlines()
    assert lines[0].split()[:4] == ["1", "Q0", "b", "1"]        # tie: document id descending (trec_eval)
    assert [l.split()[2] for l in lines] == ["b", "a", "c"]
    assert M.load_run(p) == {"1": {"b": 1.0, "a": 1.0, "c": 0.5}}
    j = tmp_path / "preds.json"; j.write_text(json.dumps(run))
    assert M.load_run(j) == run


def test_cross_check_matches_mteb_on_random_data(tmp_path):
    pytest.importorskip("mteb")
    from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator
    from mteb.evaluation.evaluators.utils import mrr

    rng = np.random.default_rng(0)
    docs = [f"d{i}" for i in range(60)]
    qrels, run = {}, {}
    for q in range(40):
        rel = list(rng.choice(docs, size=int(rng.integers(0, 3)), replace=False))
        qrels[str(q)] = {d: 1 for d in rel} if rel else {"d0": 0}
        run[str(q)] = {d: float(s) for d, s in zip(docs, rng.random(len(docs)))}
    ks = [1, 3, 5, 10, 20, 100]
    ndcg, _map, recall, prec, _ = RetrievalEvaluator.evaluate(qrels, run, ks)
    mr = mrr(qrels, run, ks)
    stored = {}
    for k in ks:
        stored.update({f"ndcg_at_{k}": ndcg[f"NDCG@{k}"], f"map_at_{k}": _map[f"MAP@{k}"],
                       f"recall_at_{k}": recall[f"Recall@{k}"], f"precision_at_{k}": prec[f"P@{k}"],
                       f"mrr_at_{k}": mr[f"MRR@{k}"]})
    cq = tmp_path / "ChemQuest.json"; cq.write_text(json.dumps({"scores": {"test": [stored]}}))
    cc = M.cross_check(run, qrels, cq)
    assert cc["ok"] and cc["n_metrics"] == 30 and cc["max_abs_diff"] <= 5e-6
    stored["ndcg_at_10"] += 0.01
    cq.write_text(json.dumps({"scores": {"test": [stored]}}))
    assert not M.cross_check(run, qrels, cq)["ok"]


def test_cli_writes_per_query_csv_for_two_label_sets(tmp_path):
    run = tmp_path / "run.trec.gz"; M.write_trec_run(RUN, run, depth=100)
    q1 = tmp_path / "qrels_a.jsonl"; q2 = tmp_path / "qrels_b.jsonl"
    q1.write_text("".join(json.dumps({"query_id": q, "doc_id": d, "relevance": r}) + "\n" for q, ds in QRELS.items() for d, r in ds.items()))
    q2.write_text("".join(json.dumps({"query_id": q, "doc_id": "b", "relevance": 1}) + "\n" for q in QRELS))
    assert M.main(["--run", str(run), "--qrels", str(q1), str(q2), "--out-dir", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "per_query_metrics__qrels_a.csv").exists() and (tmp_path / "o" / "summary.json").exists()
    back = M.load_per_query_csv(tmp_path / "o" / "per_query_metrics__qrels_b.csv")
    assert set(back) == {"q1", "q2", "q3"} and back["q1"]["recall_at_10"] == 1.0


def test_run_eval_writes_run_file_per_query_csv_and_cross_check(tmp_path, monkeypatch):
    pytest.importorskip("mteb"); pytest.importorskip("sentence_transformers")
    from stages import run_eval
    data = tmp_path / "task"; data.mkdir()
    (data / "queries.jsonl").write_text("".join(json.dumps({"id": q, "text": "t"}) + "\n" for q in RUN))
    (data / "qrels_x.jsonl").write_text("".join(json.dumps({"query_id": q, "doc_id": d, "relevance": r}) + "\n" for q, ds in QRELS.items() for d, r in ds.items()))
    stored = M.aggregate(M.per_query_metrics(RUN, QRELS, (1, 3, 5, 10, 20, 100)))
    stored = {k: round(v, 5) for k, v in stored.items() if "_at_" in k}

    class FakeMTEB:
        def __init__(self, tasks): pass
        def run(self, model, **kw):
            out = Path(kw["output_folder"]); (out / "ChemQuest_default_predictions.json").write_text(json.dumps(RUN))
            d = out / "m" / "r"; d.mkdir(parents=True); (d / "ChemQuest.json").write_text(json.dumps({"scores": {"test": [stored]}}))
            return []

    cfg = {"models": [{"name": "m/x", "revision": "r"}], "output_root": str(tmp_path / "res"), "top_k": 100,
           "tasks": [{"class": "x.Y", "kwargs": {"data_dir": str(data), "qrels_file": "qrels_x.jsonl"}}]}
    p = tmp_path / "s6.yaml"; p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(run_eval, "MTEB", FakeMTEB)
    monkeypatch.setattr(run_eval, "SentenceTransformer", lambda *a, **k: object())
    monkeypatch.setattr(run_eval, "resolve_tasks", lambda specs, mods: [object()])
    monkeypatch.setattr(sys, "argv", ["run_eval", "--config", str(p)])
    run_eval.main()
    out = tmp_path / "res" / "m__x_r"
    assert (out / "run_top100.trec.gz").exists() and (out / "per_query_metrics.csv").exists()
    ms = json.loads((out / "metrics_summary.json").read_text())
    assert ms["cross_check"]["ok"] and ms["aggregate"]["n_queries"] == 3
    assert json.loads((out / "run_manifest.json").read_text())["per_query_outputs"]["cross_check"]["ok"]


def test_trec_run_file_is_byte_reproducible(tmp_path):
    a, b = tmp_path / "a.trec.gz", tmp_path / "b.trec.gz"
    M.write_trec_run(RUN, a); M.write_trec_run(RUN, b)
    assert a.read_bytes() == b.read_bytes()
    assert M.load_run(a)["q1"] == RUN["q1"]

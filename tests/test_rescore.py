"""tools/rescore.py: metrics against hand-computed values, MTEB's evaluator, and the overlap table vs the qrels file."""
import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("pytrec_eval")

from pipeline_lib.dataset_pin import sha256_file  # noqa: E402
from test_make_span_qrels import _build  # noqa: E402
from tools.rescore import (  # noqa: E402
    label_set_of, load_predictions, load_qrels, qrels_from_overlap_table, load_overlap_table, rescore_config, score,
)


# ---------------------------------------------------------------- hand-made example
# q0: one relevant chunk, ranked 2nd           nDCG@10 = 1/log2(3)                = 0.6309297536, Recall@10 = 1
# q1: two relevant chunks, ranked 1st and 12th nDCG@10 = 1/(1 + 1/log2(3))        = 0.6131471928, Recall@10 = 1/2
# q2: one relevant chunk, ranked 1st           nDCG@10 = 1,                         Recall@10 = 1
# q3: no relevant chunk (relevance-0 row)      nDCG@10 = 0,                         Recall@10 = 0
NDCG_HAND = (0.6309297536 + 0.6131471928 + 1.0 + 0.0) / 4  # 0.5610192366
RECALL_HAND = (1.0 + 0.5 + 1.0 + 0.0) / 4                  # 0.625


def _hand_example():
    chunks = [f"d{i:02d}" for i in range(14)]

    def ranking(order):  # best first; scores strictly decreasing
        return {c: 100.0 - i for i, c in enumerate(order)}

    preds = {
        "0": ranking(["d05", "d00"] + [c for c in chunks if c not in ("d05", "d00")]),
        "1": ranking(["d01"] + [c for c in chunks if c not in ("d01", "d02")][:10] + ["d02"]),
        "2": ranking(["d03"] + [c for c in chunks if c != "d03"]),
        "3": ranking(chunks),
    }
    assert list(preds["1"]).index("d02") == 11  # rank 12
    qrels = {"0": {"d00": 1}, "1": {"d01": 1, "d02": 1}, "2": {"d03": 1}, "3": {"d09": 0}}
    return preds, qrels


def test_ndcg_and_recall_match_the_hand_computed_values():
    preds, qrels = _hand_example()
    m = score(preds, qrels)
    assert m["n_queries"] == 4 and m["n_empty_questions"] == 1
    assert m["ndcg_at_10"] == pytest.approx(NDCG_HAND, abs=1e-9)
    assert m["recall_at_10"] == pytest.approx(RECALL_HAND, abs=1e-12)
    assert m["geom_at_10"] == pytest.approx(math.sqrt(NDCG_HAND * RECALL_HAND), abs=1e-9)
    per_query = (math.sqrt(0.6309297536 * 1) + math.sqrt(0.6131471928 * 0.5) + 1.0 + 0.0) / 4
    assert m["geom_at_10_query_mean"] == pytest.approx(per_query, abs=1e-9)


def test_unsorted_predictions_are_sorted_by_score_first(tmp_path):
    preds, qrels = _hand_example()
    shuffled = {}
    for q, docs in preds.items():
        items = list(docs.items())
        random.Random(7).shuffle(items)
        shuffled[q] = dict(items)
    p = tmp_path / "ChemQuest_default_predictions.json"
    p.write_text(json.dumps(shuffled))
    loaded = load_predictions(p)
    for q in preds:
        assert list(loaded[q]) == list(preds[q])  # best first
    assert score(loaded, qrels) == score(preds, qrels)


def test_an_empty_question_stays_in_the_mean_and_counts_zero():
    preds, qrels = _hand_example()
    without = {q: v for q, v in qrels.items() if q != "3"}
    m_all = score(preds, qrels)
    m_drop = score({q: v for q, v in preds.items() if q != "3"}, without)
    assert m_drop["ndcg_at_10"] == pytest.approx(m_all["ndcg_at_10"] * 4 / 3)  # dropping it would inflate the mean
    # an empty dict (no row at all) is the same as a relevance-0 row
    assert score(preds, {**qrels, "3": {}}) == m_all


def test_predictions_and_labels_must_cover_the_same_queries():
    preds, qrels = _hand_example()
    with pytest.raises(ValueError, match="different queries"):
        score({q: v for q, v in preds.items() if q != "2"}, qrels)
    with pytest.raises(ValueError, match="different queries"):
        score(preds, {q: v for q, v in qrels.items() if q != "2"})


def test_matches_the_mteb_evaluator_on_random_data():
    from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator

    rng = random.Random(3)
    docs = [f"c{i}" for i in range(60)]
    qrels = {}
    for q in range(40):
        rel = rng.sample(docs, rng.choice([0, 1, 1, 2, 3]))
        qrels[str(q)] = {d: 1 for d in rel} if rel else {"c0": 0}
    preds = {str(q): {d: rng.random() for d in rng.sample(docs, 40)} for q in range(40)}
    m = score(preds, qrels)
    ndcg, _, recall, _, _ = RetrievalEvaluator.evaluate(qrels, preds, [10])
    assert abs(m["ndcg_at_10"] - ndcg["NDCG@10"]) <= 5e-6
    assert abs(m["recall_at_10"] - recall["Recall@10"]) <= 5e-6


def test_label_set_of():
    assert label_set_of("x/qrels_span_c080.jsonl") == ("span", "0.8")
    assert label_set_of("qrels_span_c100.jsonl") == ("span", "1")
    assert label_set_of("qrels_span_c087p5.jsonl") == ("span", "0.875")
    assert label_set_of("qrels_lexical.jsonl") == ("lexical", "")
    assert label_set_of("other.jsonl") == ("other", "")


# ---------------------------------------------------------------- overlap table vs qrels file
def _predictions_for(d, out, seed=1):
    corpus = [json.loads(l)["id"] for l in (out / "corpus.jsonl").read_text().splitlines()]
    n_q = len([l for l in (d / "qa.jsonl").read_text().splitlines() if l.strip()])
    rng = random.Random(seed)
    return {str(q): {c: rng.random() for c in corpus} for q in range(n_q)}


def test_overlap_table_and_qrels_file_give_identical_results(tmp_path):
    d = _build(tmp_path)
    r = subprocess.run([sys.executable, "-m", "stages.make_span_qrels", "--qa-jsonl", str(d / "qa.jsonl"),
                        "--chunks-jsonl", str(d / "chunks.jsonl"), "--full-text-jsonl", str(d / "full_text.jsonl"),
                        "--output-folder", str(d / "out"), "--qa-sha256", sha256_file(d / "qa.jsonl"),
                        "--cover-level", "0.5", "0.8", "0.9", "1.0"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = d / "out"
    pred_path = d / "ChemQuest_default_predictions.json"
    preds = _predictions_for(d, out)
    pred_path.write_text(json.dumps(preds))
    from_qrels = {}
    for tag in ("c050", "c080", "c090", "c100"):
        (row,) = rescore_config("cfg", pred_path, qrels_files=[out / f"qrels_span_{tag}.jsonl"])
        from_qrels[row["cover_level"]] = row
    rows = rescore_config("cfg", pred_path, overlap_table=out / "overlap_table.jsonl", cover_levels=["0.5", "0.8", "0.9", "1.0"])
    assert [r["cover_level"] for r in rows] == ["0.5", "0.8", "0.9", "1"] == list(from_qrels)
    for row in rows:
        ref = from_qrels[row["cover_level"]]
        for key in ("n_queries", "n_empty_questions", "ndcg_at_10", "recall_at_10", "geom_at_10", "geom_at_10_query_mean"):
            assert row[key] == ref[key], (row["cover_level"], key)
        assert row["label_set"] == ref["label_set"] == "span"
    # the labels themselves are equal too
    table = load_overlap_table(out / "overlap_table.jsonl")
    for tag, c in (("c050", 0.5), ("c090", 0.9), ("c100", 1.0)):
        q_file = {q: {k: v for k, v in rels.items() if v} for q, rels in load_qrels(out / f"qrels_span_{tag}.jsonl").items()}
        q_tab = qrels_from_overlap_table(table, [str(i) for i in range(8)], c)
        assert q_file == q_tab


def test_cli_writes_one_row_per_config_label_set_and_c(tmp_path):
    d = _build(tmp_path)
    r = subprocess.run([sys.executable, "-m", "stages.make_span_qrels", "--qa-jsonl", str(d / "qa.jsonl"),
                        "--chunks-jsonl", str(d / "chunks.jsonl"), "--full-text-jsonl", str(d / "full_text.jsonl"),
                        "--output-folder", str(d / "out"), "--qa-sha256", sha256_file(d / "qa.jsonl"),
                        "--cover-level", "0.8"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = d / "out"
    # a pipeline folder layout: <cfg>/mteb_task, <cfg>/results/<model>/ChemQuest_default_predictions.json (+ ChemQuest.json)
    cfgs = []
    for name, seed in (("cfgA", 1), ("cfgB", 2)):
        base = tmp_path / "tests" / name
        (base / "results" / "m").mkdir(parents=True)
        (base / "mteb_task").mkdir()
        for f in out.iterdir():
            (base / "mteb_task" / f.name).write_text(f.read_text())
        preds = _predictions_for(d, out, seed)
        (base / "results/m/ChemQuest_default_predictions.json").write_text(json.dumps(preds))
        lab = load_qrels(base / "mteb_task/qrels_span_c080.jsonl")
        m = score(preds, lab)
        (base / "results/m/ChemQuest.json").write_text(json.dumps({"scores": {"test": [
            {"ndcg_at_10": round(m["ndcg_at_10"], 5), "recall_at_10": round(m["recall_at_10"], 5)}]}}))
        cfgs.append(str(base))
    csv_path = tmp_path / "res.csv"
    r = subprocess.run([sys.executable, "-m", "tools.rescore", "--config-dir", *cfgs, "--cover-level", "0.8", "0.5", "1.0",
                        "--qrels-file", "qrels_span_c080.jsonl", "--with-stored", "--output", str(csv_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 2 * (1 + 3)  # per config: one qrels file + three cover levels
    # one row per (config, label set, c) and label source; c = 0.8 was requested from the qrels file AND from the table
    keys = [(r["config"], r["label_set"], r["cover_level"], r["labels_source"]) for r in rows]
    assert len(set(keys)) == len(keys)
    for cfg in ("cfgA", "cfgB"):
        a, b = [r for r in rows if r["config"] == cfg and r["cover_level"] == "0.8"]
        assert (a["ndcg_at_10"], a["recall_at_10"], a["geom_at_10"]) == (b["ndcg_at_10"], b["recall_at_10"], b["geom_at_10"])
    for r_ in rows:
        assert float(r_["geom_at_10"]) == pytest.approx(math.sqrt(float(r_["ndcg_at_10"]) * float(r_["recall_at_10"])), abs=1e-9)
        if r_["cover_level"] == "0.8":  # the label set that MTEB stored: reproduced within the 5-decimal rounding
            assert float(r_["abs_diff_ndcg_at_10"]) <= 5e-6 and float(r_["abs_diff_recall_at_10"]) <= 5e-6


def test_cli_rejects_a_bad_cover_level_and_missing_inputs(tmp_path):
    base = tmp_path / "x"
    base.mkdir()
    (base / "p.json").write_text("{}")
    (base / "t.jsonl").write_text("")
    for args in (["--predictions", str(base / "p.json"), "--overlap-table", str(base / "t.jsonl"), "--cover-level", "1.5"],
                 ["--predictions", str(base / "p.json"), "--overlap-table", str(base / "t.jsonl"), "--cover-level", "0"],
                 ["--predictions", str(base / "p.json")]):
        r = subprocess.run([sys.executable, "-m", "tools.rescore", *args, "--output", str(base / "o.csv")],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode != 0 and "ERROR" in r.stderr, args

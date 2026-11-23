"""tools/fragmentation.py (fragmented evidence vs gain at the large size) and tools/examples.py (named success / failure cases)."""
import csv
import json
from fractions import Fraction

import numpy as np
import pytest

from analysis_world import build_world
from tools import analysis_data as AD, examples as EX, fragmentation as FR, metrics as M


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("fx"))


def test_fragmented_questions_use_exact_fractions(tmp_path):
    t = tmp_path / "overlap_table.jsonl"
    rows = [{"query_id": "0", "overlap_chars": 90, "span_len": 100}, {"query_id": "0", "overlap_chars": 10, "span_len": 100},
            {"query_id": "1", "overlap_chars": 89, "span_len": 100}, {"query_id": "2", "overlap_chars": 100, "span_len": 100}]
    t.write_text("".join(json.dumps(r) + "\n" for r in rows))
    f = FR.fragmented_questions(t, ["0", "1", "2", "3"], Fraction(9, 10))
    assert f.tolist() == [False, True, False, True]                                  # 90/100 = c is enough; 89/100 is not; a question without a row counts as fragmented
    assert FR.fragmented_questions(t, ["0"], Fraction(1)).tolist() == [True]


def test_group_difference_sign_and_interval(world):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    boot = AD.PaperBootstrap(runs.groups, 300, 1)
    rng = np.random.default_rng(0)
    frag = rng.random(36) < 0.4
    gain = np.where(frag, 0.3, 0.05) + rng.normal(size=36) * 0.02
    r = FR.group_difference(gain, frag, boot)
    assert r["mean_gain_fragmented"] == pytest.approx(gain[frag].mean()) and r["difference"] == pytest.approx(gain[frag].mean() - gain[~frag].mean())
    assert 0.15 < r["difference"] and r["difference_ci_low"] > 0 and r["p_bootstrap"] < 0.05
    none = FR.group_difference(rng.normal(size=36), np.zeros(36, dtype=bool), boot)
    assert np.isnan(none["mean_gain_fragmented"]) and np.isnan(none["difference_ci_low"])


def test_fragmentation_end_to_end(world, tmp_path):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    rows = FR.analyse(runs, Fraction(9, 10), [[128, 512]], 0, resamples=200, seed=2)
    assert {r["strategy"] for r in rows} == {"fixed_token", "hierarchical_section"} and {r["metric"] for r in rows} == set(FR.METRICS)
    assert len(rows) == 2 * 3 * (3 + 1)                                              # strategies x metrics x (3 models + all)
    r = rows[0]
    assert r["n_fragmented"] + r["n_not_fragmented"] == 36 and r["small"] == 128 and r["large"] == 512
    # the count equals the number of questions in Step 2 or 3 of the small prepared folder
    stats_ = json.loads((world["prepared"][("fixed_token", 128, 0)] / "mteb_task" / "label_stats_c090.json").read_text())["questions_per_step"]
    assert r["n_fragmented"] == stats_["step2"] + stats_["step3"] + stats_["step0_no_overlapping_chunk"]
    one = next(x for x in rows if x["model"] == "org/m_a" and x["strategy"] == "fixed_token" and x["metric"] == "ndcg_at_10")
    i, j = runs.index["org/m_a|fixed_token|128|0"], runs.index["org/m_a|fixed_token|512|0"]
    gain = runs.metrics["ndcg_at_10"][j] - runs.metrics["ndcg_at_10"][i]
    frag = FR.fragmented_questions(world["prepared"][("fixed_token", 128, 0)] / "mteb_task" / "overlap_table.jsonl", runs.qids, Fraction(9, 10))
    assert one["mean_gain_fragmented"] == pytest.approx(gain[frag].mean()) and one["difference"] == pytest.approx(gain[frag].mean() - gain[~frag].mean())
    out = tmp_path / "o"
    assert FR.main(["--roots", str(world["root"]), "--qa", str(world["qa_path"]), "--out-dir", str(out), "--resamples", "30"]) == 0


def test_fragmentation_settings_come_from_the_config():
    st = FR.settings()
    assert st["overlap"] == 0 and [128, 512] in st["pairs"]


# ---- examples --------------------------------------------------------------------------------------------------
ST = {"works_ndcg": 0.5, "margin": 0.5, "size_pairs": [[128, 512]], "excerpt_chars": 100, "top_chunks": 3, "chunk_chars": 80, "n_per_rule": 5}


def test_case_rules_mean_what_they_say(world):
    runs = AD.load_runs([world["root"]], world["qa_path"])
    nd, rc = runs.metrics["ndcg_at_10"], runs.metrics["recall_at_10"]
    cases = EX.find_cases(runs, ST)
    assert set(cases) == set(EX.RULES)
    for q, (i, j) in cases["fails_small_works_large"]:
        assert runs.meta[i]["size"] == 128 and runs.meta[j]["size"] == 512 and runs.meta[i]["model"] == runs.meta[j]["model"] and runs.meta[i]["chunker"] == runs.meta[j]["chunker"]
        assert rc[i][q] == 0 and rc[j][q] == 1 and nd[j][q] >= 0.5
    for q, (d, b) in cases["dense_wins_vs_bm25"]:
        assert runs.meta[d]["kind"] == "dense" and runs.meta[b]["kind"] == "bm25" and nd[d][q] - nd[b][q] >= 0.5
    for q, (b, d) in cases["bm25_wins_vs_dense"]:
        assert nd[b][q] - nd[d][q] >= 0.5
    for q, ids in cases["all_fail"]:
        assert len({(runs.meta[i]["chunker"], runs.meta[i]["size"]) for i in ids}) == 1 and all(rc[i][q] == 0 for i in ids)
    for q, (d, r) in cases["rerank_rescues_dense_miss"]:
        assert runs.meta[r]["kind"] == "rerank" and rc[d][q] == 0 and rc[r][q] == 1
    assert cases["fails_small_works_large"] and cases["dense_wins_vs_bm25"] and cases["rerank_rescues_dense_miss"]


def test_sampling_is_seeded_and_one_case_per_question(world):
    runs = AD.load_runs([world["root"]], world["qa_path"])
    cases = EX.find_cases(runs, ST)
    a, b = EX.sample_cases(cases, 4, seed=1), EX.sample_cases(cases, 4, seed=1)
    assert a == b and a != EX.sample_cases(cases, 4, seed=2)
    for rule, cs in a.items():
        assert len(cs) <= 4 and len({q for q, _ in cs}) == len(cs)


def test_export_rows_carry_span_chunks_and_labels(world, tmp_path):
    runs = AD.load_runs([world["root"]], world["qa_path"])
    chosen = EX.sample_cases(EX.find_cases(runs, ST), 2, seed=1)
    mat = EX.Material(runs, 0.9)
    rows = []
    for rule, cs in chosen.items():
        for k, (q, ids) in enumerate(cs, 1):
            rows += EX.case_rows(rule, k, q, ids, mat, world["qa"], ST)
    r = next(x for x in rows if x["rule"] == "fails_small_works_large")
    assert len(r["span_excerpt"]) <= 104 and r["question"] and "top1" in r and "top3" in r and "top4" not in r
    assert r["top1"].startswith(("[RELEVANT]", "[not relevant]")) and len(r["top1"]) <= 130
    two = [x for x in rows if x["case"] == r["case"]]
    assert len(two) == 2 and two[0]["configuration"] != two[1]["configuration"] and two[0]["recall_at_10"] == 0 and two[1]["recall_at_10"] == 1
    assert two[1]["first_relevant_rank"] <= 10 and (two[0]["first_relevant_rank"] == "not in top 100" or two[0]["first_relevant_rank"] > 10)
    md = EX.to_markdown(rows, 3)
    assert md.startswith("# Success and failure examples") and "fails_small_works_large-1" in md and "**Evidence span:**" in md


def test_examples_cli(world, tmp_path):
    out = tmp_path / "o"
    assert EX.main(["--roots", str(world["root"]), "--qa", str(world["qa_path"]), "--out-dir", str(out), "--n-per-rule", "2"]) == 0
    rows = list(csv.DictReader(open(out / "examples.csv")))
    assert {r["rule"] for r in rows} <= set(EX.RULES) and (out / "examples.md").read_text().count("## ") >= 1


def test_examples_settings_from_config():
    st = EX.settings()
    assert st["n_per_rule"] == 10 and st["works_ndcg"] == 0.5 and st["top_chunks"] == 3

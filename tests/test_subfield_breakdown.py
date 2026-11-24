"""tools/subfield_breakdown.py: official question type and subfields (multi-label and fractional), automatic features, paper-level intervals, paired comparisons,
contrasts between groups and Holm families. No question type is guessed from the text."""
import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from analysis_world import WORLD_SUBFIELDS, WORLD_TYPES, build_world, patch_metadata_constants
from pipeline_lib import question_features as QF, question_metadata as QM
from pipeline_lib.question_metadata import QuestionMetadataError
from tools import analysis_data as AD, stats, subfield_breakdown as SB

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("sb"))


@pytest.fixture()
def labelled(world, monkeypatch):
    patch_metadata_constants(monkeypatch)
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense", "bm25"])
    meta = AD.load_question_metadata(world["qa_path"])
    idx = [int(q) for q in runs.qids]
    feats = [QF.question_features(world["qa"][i]["question"]) for i in idx]
    fam = SB.build_groups([meta.question_type[i] for i in idx], [meta.subfields[i] for i in idx], feats, meta.subfield_names, min_questions=5)
    return {"runs": runs, "meta": meta, "feats": feats, "fam": fam, "idx": idx}


def by_name(groups):
    return {(g.name, g.attrs.get("label_rule")): g for g in groups}


# ---- no guessing ---------------------------------------------------------------------------------------------------
def test_no_heuristic_question_type_code_is_left():
    code = re.sub(r'""".*?"""', "", (ROOT / "tools" / "subfield_breakdown.py").read_text(), count=1, flags=re.S)     # without the module docstring
    for gone in ("question_tags", "qword", "yes_no", "QWORDS", "YES_NO_WORDS", "UNIT_RE", "read_fields", "fields_of_study", "UNLABELLED", "tag:"):
        assert gone not in code, gone
    assert not hasattr(SB, "question_tags") and not hasattr(SB, "read_fields") and not hasattr(SB, "is_formula")
    for f in list((ROOT / "tools").glob("*.py")) + list((ROOT / "pipeline_lib").glob("*.py")):
        assert "question_tags" not in f.read_text(), f.name


def test_question_type_comes_from_the_metadata_file(labelled, world):
    meta, runs = labelled["meta"], labelled["runs"]
    types = {g.attrs["question_type"]: g for g in labelled["fam"]["question_type"][0]}
    assert set(types) == set(WORLD_TYPES)
    for i, q in enumerate(runs.qids):
        want = meta.question_type[int(q)]
        assert [t for t, g in types.items() if g.weights[i] == 1] == [want]
    first = json.loads((Path(world["qa_path"]).with_name("question_metadata.jsonl")).read_text().splitlines()[0])
    assert meta.question_type[0] == first["question_type"]


def test_missing_metadata_stops_with_the_download_command(world, tmp_path):
    qa = tmp_path / "qa.jsonl"
    qa.write_text(Path(world["qa_path"]).read_text())
    with pytest.raises(QuestionMetadataError, match="download_chemquests"):
        SB.main(["--roots", str(world["root"]), "--qa", str(qa), "--out-dir", str(tmp_path / "o"), "--resamples", "20"])
    assert not (tmp_path / "o" / "breakdown_question_type.csv").exists()


def test_a_metadata_file_that_does_not_match_qa_stops_the_tool(world, tmp_path, monkeypatch):
    patch_metadata_constants(monkeypatch)
    for n in ("qa.jsonl", "question_metadata.jsonl"):
        (tmp_path / n).write_text((Path(world["qa_path"]).with_name(n)).read_text())
    rows = [json.loads(l) for l in (tmp_path / "question_metadata.jsonl").read_text().splitlines()]
    rows[3]["question"] += " in this study?"
    (tmp_path / "question_metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(QuestionMetadataError, match="question text differs"):
        SB.main(["--roots", str(world["root"]), "--qa", str(tmp_path / "qa.jsonl"), "--out-dir", str(tmp_path / "o"), "--resamples", "20"])


# ---- groups ----------------------------------------------------------------------------------------------------
def test_types_are_exclusive_subfields_are_multi_label_and_fractional(labelled):
    n = len(labelled["runs"].qids)
    tg = labelled["fam"]["question_type"][0]
    assert np.all(sum(g.weights for g in tg) == 1) and sum(g.n_questions for g in tg) == n
    sg = labelled["fam"]["subfield"][0]
    multi = [g for g in sg if g.attrs["label_rule"] == SB.MULTI]
    frac = [g for g in sg if g.attrs["label_rule"] == SB.FRACTIONAL]
    assert [g.name for g in multi] == [f"subfield:{s}" for s in WORLD_SUBFIELDS]
    n_labels = np.array([len(labelled["meta"].subfields[i]) for i in labelled["idx"]])
    assert sum(g.n_questions for g in multi) == n_labels.sum() > n
    assert np.allclose(sum(g.weights for g in frac), 1.0)                      # the weights of one question add up to 1
    assert sum(g.weights.sum() for g in frac) == pytest.approx(n)              # so the totals add up to the number of questions
    for g in frac:
        m = next(x for x in multi if x.name == g.name)
        assert np.array_equal(g.weights > 0, m.weights > 0)
        assert set(np.round(g.weights[g.weights > 0], 6)) <= {1.0, 0.5}


def test_every_question_has_a_subfield(labelled):
    assert all(len(s) >= 1 for s in labelled["meta"].subfields)
    assert np.all(sum(g.weights for g in labelled["fam"]["subfield"][0] if g.attrs["label_rule"] == SB.MULTI) >= 1)


def test_feature_groups_are_flagged_and_not_flagged_and_labelled_as_rule_based(labelled):
    groups, contrasts = labelled["fam"]["question_features"]
    for feat in SB.FEATURES:
        fl = next(g for g in groups if g.name == f"{feat}:flagged")
        nf = next(g for g in groups if g.name == f"{feat}:not_flagged")
        assert np.all(fl.weights + nf.weights == 1) and fl.attrs["feature_source"] == "rule_based" and fl.attrs["feature"] == feat
        flags = np.array([getattr(f, feat) for f in labelled["feats"]])
        assert np.array_equal(fl.weights > 0, flags)
    assert not any(k in g.attrs for g in groups for k in ("type", "category"))        # the features are never a type
    assert all(g.attrs.get("question_type") in (None, *WORLD_TYPES) for g in groups)
    cell = next(g for g in groups if g.name == "auto_has_number:flagged|question_type:Applied")
    assert cell.attrs == {"feature": "auto_has_number", "feature_source": "rule_based", "flag": "flagged", "question_type": "Applied"}


# ---- statistics ---------------------------------------------------------------------------------------------------
def test_group_values_intervals_and_weights(labelled):
    runs, fam = labelled["runs"], labelled["fam"]
    rows, paired, sizes = SB.analyse(runs, fam["question_type"][0], fam["question_type"][1], resamples=200, seed=3, min_questions=5)
    g = next(x for x in fam["question_type"][0] if x.name == "question_type:Applied")
    i = runs.index["org/m_a|fixed_token|128|0"]
    r = next(x for x in rows if x["group"] == "question_type:Applied" and x["model"] == "org/m_a" and x["size"] == 128 and x["chunker"] == "fixed_token"
             and x["metric"] == "ndcg_at_10" and x["row_kind"] == "group")
    assert r["value"] == pytest.approx(runs.metrics["ndcg_at_10"][i][g.weights > 0].mean())
    assert r["ci_low"] <= r["value"] <= r["ci_high"] and r["ci_width"] == pytest.approx(r["ci_high"] - r["ci_low"]) and r["n_questions"] == 9
    sz = next(s for s in sizes if s["group"] == "question_type:Applied")
    assert sz["n_questions"] == 9 and sz["n_papers"] == 9 and 0 < sz["max_paper_share"] <= 1 and sz["weight_sum"] == 9 and sz["median_ci_width_geom"] > 0
    geom = next(x for x in rows if x["group"] == "question_type:Applied" and x["model"] == "org/m_a" and x["size"] == 128 and x["metric"] == "geom_at_10" and x["chunker"] == "fixed_token")
    nd, rc = (runs.metrics[m][i][g.weights > 0].mean() for m in ("ndcg_at_10", "recall_at_10"))
    assert geom["value"] == pytest.approx(np.sqrt(nd * rc))


def test_fractional_value_is_the_weighted_mean_and_totals_are_n(labelled):
    runs, fam = labelled["runs"], labelled["fam"]
    frac = [g for g in fam["subfield"][0] if g.attrs["label_rule"] == SB.FRACTIONAL]
    rows, _, sizes = SB.analyse(runs, frac, resamples=100, seed=3, min_questions=5)
    g = frac[0]
    i = runs.index["org/m_a|fixed_token|128|0"]
    r = next(x for x in rows if x["group"] == g.name and x["model"] == "org/m_a" and x["size"] == 128 and x["chunker"] == "fixed_token" and x["metric"] == "recall_at_10")
    assert r["value"] == pytest.approx((runs.metrics["recall_at_10"][i] * g.weights).sum() / g.weights.sum()) and r["ci_low"] <= r["value"] <= r["ci_high"]
    assert sum(s["weight_sum"] for s in sizes) == pytest.approx(len(runs.qids))


def test_weighted_bootstrap_with_unit_weights_equals_the_unweighted_one(labelled):
    runs = labelled["runs"]
    w = (np.arange(len(runs.qids)) % 2 == 0).astype(float)
    a = SB.Resampler({"g": w}, runs.groups, 100, 7)
    b = SB.Resampler({"g": 0.5 * w}, runs.groups, 100, 7)                 # a common factor changes nothing
    v = runs.metrics["ndcg_at_10"]
    assert np.allclose(a.resampled("g", v), b.resampled("g", v)) and np.allclose(a.point("g", v), v[:, w > 0].mean(axis=1))


def test_paired_comparisons_and_holm_over_the_four_types(labelled):
    runs, fam = labelled["runs"], labelled["fam"]
    rows, paired, _ = SB.analyse(runs, fam["question_type"][0], fam["question_type"][1], resamples=300, seed=3, min_questions=5)
    key = lambda x: (x["model"], x["chunker"], x["size"], x["overlap"], x["metric"], x["comparison"])      # noqa: E731
    assert {p["comparison"] for p in paired} >= {"dense_minus_bm25"} and not any(p["model"] == "bm25" for p in paired)
    pr = [p for p in paired if p["row_kind"] == "group" and key(p) == ("org/m_a", "fixed_token", 128, 0, "ndcg_at_10", "dense_minus_bm25")]
    assert [p["group"] for p in pr] == [f"question_type:{t}" for t in WORLD_TYPES]                          # Holm over the 4 types
    assert [p["p_holm"] for p in pr] == pytest.approx(stats.holm([p["p_bootstrap"] for p in pr]))
    i, j = runs.index["org/m_a|fixed_token|128|0"], runs.index["bm25|fixed_token|128|0"]
    g = next(x for x in fam["question_type"][0] if x.name == pr[0]["group"])
    nd = runs.metrics["ndcg_at_10"]
    assert pr[0]["difference"] == pytest.approx(nd[i][g.weights > 0].mean() - nd[j][g.weights > 0].mean())
    assert all(0 < p["p_bootstrap"] <= 1 and p["ci_low"] <= p["difference"] <= p["ci_high"] and p["p_holm"] >= p["p_bootstrap"] - 1e-12 for p in pr)


def test_interaction_between_types_is_the_difference_of_the_two_differences(labelled):
    runs, fam = labelled["runs"], labelled["fam"]
    _, paired, _ = SB.analyse(runs, fam["question_type"][0], fam["question_type"][1], resamples=300, seed=3, min_questions=5)
    con = [p for p in paired if p["row_kind"] == "contrast" and (p["model"], p["chunker"], p["size"], p["metric"], p["comparison"]) ==
           ("org/m_a", "fixed_token", 128, "ndcg_at_10", "dense_minus_bm25")]
    assert len(con) == 6 and len({p["group"] for p in con}) == 6                                         # all 6 pairs of types
    assert [p["p_holm"] for p in con] == pytest.approx(stats.holm([p["p_bootstrap"] for p in con]))      # Holm over the 6 pairs
    p = con[0]
    a, b = (next(g for g in fam["question_type"][0] if g.name == n) for n in p["group"].split(" - "))
    i, j = runs.index["org/m_a|fixed_token|128|0"], runs.index["bm25|fixed_token|128|0"]
    nd = runs.metrics["ndcg_at_10"]
    d = lambda g: nd[i][g.weights > 0].mean() - nd[j][g.weights > 0].mean()                              # noqa: E731
    assert p["difference"] == pytest.approx(d(a) - d(b)) and p["ci_low"] <= p["difference"] <= p["ci_high"]


def test_group_contrast_rows_per_run(labelled):
    runs, fam = labelled["runs"], labelled["fam"]
    rows, _, _ = SB.analyse(runs, fam["question_type"][0], fam["question_type"][1], resamples=200, seed=3, min_questions=5)
    c = [r for r in rows if r["row_kind"] == "contrast" and r["model"] == "org/m_a" and r["size"] == 128 and r["chunker"] == "fixed_token" and r["metric"] == "recall_at_10"]
    assert len(c) == 6 and all(0 < r["p_bootstrap"] <= 1 and r["p_holm"] >= r["p_bootstrap"] - 1e-12 for r in c)


def test_small_cells_get_numbers_but_no_test(labelled):
    runs, fam = labelled["runs"], labelled["fam"]
    rows, paired, _ = SB.analyse(runs, fam["question_type"][0], fam["question_type"][1], resamples=100, seed=3, min_questions=30)      # every type has 9
    assert [r for r in rows if r["row_kind"] == "group"] and not [r for r in rows if r["row_kind"] == "contrast"]
    assert paired and all(np.isnan(p["p_bootstrap"]) and np.isnan(p["p_holm"]) for p in paired)
    assert all(not np.isnan(p["difference"]) and not np.isnan(p["ci_low"]) for p in paired)
    groups, contrasts = SB.build_groups(*(lambda m: ([m.question_type[i] for i in labelled["idx"]], [m.subfields[i] for i in labelled["idx"]]))(labelled["meta"]),
                                        labelled["feats"], labelled["meta"].subfield_names, min_questions=30)["question_features"]
    assert contrasts and all("|question_type:" not in c.name for c in contrasts)                           # no cell contrast below 30 questions


def test_feature_contrast_is_flagged_minus_not_flagged(labelled):
    runs = labelled["runs"]
    groups, contrasts = labelled["fam"]["question_features"]
    contrasts = [c for c in contrasts if c.name == "auto_has_number:flagged - auto_has_number:not_flagged"]
    rows, paired, _ = SB.analyse(runs, [g for g in groups if "|" not in g.name and g.attrs["feature"] == "auto_has_number"], contrasts, resamples=200, seed=3, min_questions=5)
    r = next(x for x in rows if x["row_kind"] == "contrast" and x["model"] == "org/m_a" and x["size"] == 128 and x["chunker"] == "fixed_token" and x["metric"] == "ndcg_at_10")
    i = runs.index["org/m_a|fixed_token|128|0"]
    fl = next(g for g in groups if g.name == "auto_has_number:flagged").weights > 0
    assert r["value"] == pytest.approx(runs.metrics["ndcg_at_10"][i][fl].mean() - runs.metrics["ndcg_at_10"][i][~fl].mean())
    assert r["feature_source"] == "rule_based" and r["ci_low"] <= r["value"] <= r["ci_high"]
    assert {p["feature"] for p in paired} == {"auto_has_number"} and all(p["feature_source"] == "rule_based" for p in paired)


def test_a_group_with_one_paper_has_no_interval(world):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    w = np.zeros(36)
    w[:3] = 1                                                      # the three questions of paper 1
    rows, _, sizes = SB.analyse(runs, [SB.Group("one", w, "x")], resamples=50, seed=1, min_questions=1)
    assert np.isnan(rows[0]["ci_low"]) and sizes[0]["n_papers"] == 1 and sizes[0]["max_paper_share"] == 1 and not np.isnan(rows[0]["value"])


def test_an_empty_group_is_skipped_without_error(world):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    rows, _, sizes = SB.analyse(runs, [SB.Group("none", np.zeros(36), "x")], resamples=20, seed=1)
    assert rows == [] and sizes[0]["n_questions"] == 0


def test_run_pairs_follow_the_families(world):
    runs = AD.load_runs([world["root"]], world["qa_path"])
    comps = {c for _, _, c in SB.run_pairs(runs)}
    assert "dense_minus_bm25" in comps and comps <= {"dense_minus_bm25", "hybrid_minus_dense", "hybrid_minus_bm25", "rerank_minus_dense", "rerank_minus_bm25", "rerank_minus_hybrid"}
    assert "rerank_minus_dense" in comps                                  # the world has a rerank run of org/m_c


# ---- outputs ---------------------------------------------------------------------------------------------------
def read(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def test_cli_writes_all_tables(world, tmp_path, monkeypatch):
    patch_metadata_constants(monkeypatch)
    out = tmp_path / "o"
    assert SB.main(["--roots", str(world["root"]), "--qa", str(world["qa_path"]), "--out-dir", str(out), "--resamples", "50"]) == 0
    for f in ("breakdown_question_type.csv", "breakdown_question_features.csv", "breakdown_subfield.csv", "breakdown_question_type_paired.csv",
              "breakdown_question_features_paired.csv", "breakdown_subfield_paired.csv", "breakdown_groups.csv", "question_features.csv"):
        assert (out / f).exists(), f
    assert not (out / "breakdown_tag.csv").exists() and not (out / "breakdown_tag_paired.csv").exists()
    groups = read(out / "breakdown_groups.csv")
    names = {(g["group"], g["label_rule"]) for g in groups}
    assert ("subfield:Sub A", "multi_label") in names and ("subfield:Sub A", "fractional") in names and ("question_type:Applied", "") in names
    frac = [g for g in groups if g["label_rule"] == "fractional"]
    assert sum(float(g["weight_sum"]) for g in frac) == pytest.approx(36)
    assert not any("unlabelled" in g["group"] for g in groups)
    feat = read(out / "breakdown_question_features.csv")
    assert {r["feature_source"] for r in feat} == {"rule_based"} and {r["feature"] for r in feat} == {"auto_has_number", "auto_has_formula"}
    for f in ("breakdown_question_type.csv", "breakdown_question_features.csv", "breakdown_subfield.csv", "breakdown_groups.csv", "question_features.csv"):
        head = (out / f).read_text().splitlines()[0].split(",")
        assert "type" not in head and "category" not in head, f                                  # the features never go into a type column
    qf = read(out / "question_features.csv")
    assert len(qf) == 36 and set(qf[0]) == {"query_id", "question", "question_type", "auto_has_number", "auto_has_formula", "number_tokens", "formula_tokens", "feature_source"}
    assert qf[0]["question_type"] == json.loads(Path(world["qa_path"]).with_name("question_metadata.jsonl").read_text().splitlines()[0])["question_type"]
    row = next(r for r in qf if "CO2" in r["question"])
    assert row["auto_has_formula"] == "1" and "CO2" in row["formula_tokens"] and row["feature_source"] == "rule_based"

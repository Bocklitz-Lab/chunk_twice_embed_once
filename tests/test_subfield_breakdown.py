"""tools/subfield_breakdown.py: subfield groups (all listed subfields), rule-based question tags, paper-level intervals inside a group, paired dense - BM25."""
import json

import numpy as np
import pytest

from analysis_world import FIELDS, build_world
from tools import analysis_data as AD, subfield_breakdown as SB


@pytest.mark.parametrize("token,expected", [("NaCl", True), ("CO2", True), ("TiO2", True), ("LiFePO4", True), ("H2O", True), ("O2", True), ("ZnO", True),
                                              ("NMR", False), ("DNA", False), ("PEG", False), ("UV", False), ("In", False), ("As", False), ("He", False), ("Xyz", False)])
def test_formula_detection(token, expected):
    assert SB.is_formula(token) is expected


def tags(q):
    return SB.question_tags(q)


def test_question_words_are_exclusive_and_documented():
    assert tags("What is the yield?")["qword"] == "what" and tags("which catalyst works?")["qword"] == "which"
    assert tags("How many cycles were run?")["qword"] == "how" and tags("Why does it crack?")["qword"] == "why"
    assert tags("When was it measured?")["qword"] == "other" and tags("  ")["qword"] == "other" and tags("Describe the setup.")["qword"] == "other"


def test_yes_no_questions():
    for q in ("Is the sample stable?", "Does NaCl dissolve?", "Can it be reused?", "Were the films annealed?"):
        assert tags(q)["yes_no"] and tags(q)["qword"] == "other"
    assert not tags("What is stable?")["yes_no"] and not tags("How does it work?")["yes_no"]


def test_numbers_and_units():
    for q in ("What happens at 300 K?", "Which film has 12 layers?", "What shifts by 25 nm?", "What is the pH?" + " 7", "Which sample reaches 90%?", "How is 10 mg dissolved?",
              "What happens at 25 °C?", "Which peak lies at 5 eV?"):
        assert tags(q)["number_or_unit"], q
    for q in ("What is the yield?", "Which catalyst is best?", "How does it work?"):
        assert not tags(q)["number_or_unit"], q
    assert not tags("What does the Amino acid do?")["number_or_unit"]                    # "mol" inside a word is not a unit


def test_formulas_in_questions():
    assert tags("What does NaCl do to the rate?")["chemical_formula"] and tags("Which CO2 uptake is highest?")["chemical_formula"]
    assert not tags("What does NMR show?")["chemical_formula"] and not tags("In this study, what is measured?")["chemical_formula"]
    assert not tags("What is the DNA yield?")["chemical_formula"]


def test_groups_use_every_listed_subfield_and_normalise_ids():
    qa = [{"question": "What is it?", "corpus_id": "003"}, {"question": "Is it?", "corpus_id": "3"}, {"question": "How?", "corpus_id": "7"}, {"question": "Why 5 mg?", "corpus_id": "008"}]
    meta = {"3": ["Organic", "Catalysis"], "7": [], "8": ["Organic"]}
    g = SB.question_groups(qa, meta)
    assert g["subfield"]["subfield:Organic"].tolist() == [True, True, False, True] and g["subfield"]["subfield:Catalysis"].tolist() == [True, True, False, False]
    assert g["subfield"]["subfield:unlabelled"].tolist() == [False, False, True, False]
    assert g["tag"]["tag:yes_no"].tolist() == [False, True, False, False] and g["tag"]["tag:no_yes_no"].tolist() == [True, False, True, True]
    assert g["tag"]["tag:qword:what"].sum() == 1 and sum(g["tag"][k].sum() for k in g["tag"] if k.startswith("tag:qword:")) == 4      # exactly one qword per question
    assert g["tag"]["tag:number_or_unit"].tolist() == [False, False, False, True]


def test_read_fields_normalises_ids(tmp_path):
    f = tmp_path / "m.jsonl"
    f.write_text(json.dumps({"corpus_id": "012", "fields_of_study": ["A"]}) + "\n" + json.dumps({"corpus_id": "13"}) + "\n")
    assert SB.read_fields(f) == {"12": ["A"], "13": []}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("sb"))


def test_group_values_and_intervals(world):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense", "bm25"])
    qa = world["qa"]
    meta = {str(p): FIELDS[p % 4] for p in range(1, 13)}
    groups = SB.question_groups(qa, meta)
    rows, paired, sizes = SB.analyse(runs, groups["subfield"], resamples=200, seed=3)
    assert {s["group"]: s["n_questions"] for s in sizes}["subfield:Physical Chemistry"] == 9          # papers 1, 5, 9 x 3 questions
    r = next(x for x in rows if x["group"] == "subfield:Physical Chemistry" and x["model"] == "org/m_a" and x["size"] == 128 and x["chunker"] == "fixed_token" and x["metric"] == "ndcg_at_10")
    i = runs.index["org/m_a|fixed_token|128|0"]
    mask = groups["subfield"]["subfield:Physical Chemistry"]
    assert r["value"] == pytest.approx(runs.metrics["ndcg_at_10"][i][mask].mean()) and r["n_papers"] == 3
    assert r["ci_low"] <= r["value"] <= r["ci_high"]
    g = next(x for x in rows if x["group"] == "subfield:unlabelled" and x["metric"] == "geom_at_10" and x["model"] == "org/m_a" and x["size"] == 128)
    assert g["n_questions"] == 9 and g["n_papers"] == 3
    # paired dense - BM25 in the same configuration
    pr = next(x for x in paired if x["group"] == "subfield:Physical Chemistry" and x["model"] == "org/m_a" and x["size"] == 128 and x["chunker"] == "fixed_token" and x["metric"] == "ndcg_at_10")
    j = runs.index["bm25|fixed_token|128|0"]
    assert pr["dense_minus_bm25"] == pytest.approx(runs.metrics["ndcg_at_10"][i][mask].mean() - runs.metrics["ndcg_at_10"][j][mask].mean())
    assert 0 < pr["p_bootstrap"] <= 1 and pr["ci_low"] <= pr["dense_minus_bm25"] <= pr["ci_high"] and "p_holm" in pr and pr["p_holm"] >= pr["p_bootstrap"] - 1e-12
    assert not any(x["model"] == "bm25" for x in paired)


def test_a_group_with_one_paper_has_no_interval(world):
    runs = AD.load_runs([world["root"]], world["qa_path"], kinds=["dense"])
    mask = np.zeros(36, dtype=bool)
    mask[:3] = True                                                  # the three questions of paper 1
    rows, _, sizes = SB.analyse(runs, {"one paper": mask}, resamples=50, seed=1)
    assert np.isnan(rows[0]["ci_low"]) and sizes[0]["n_papers"] == 1 and not np.isnan(rows[0]["value"])


def test_cli_writes_all_tables(world, tmp_path):
    out = tmp_path / "o"
    assert SB.main(["--roots", str(world["root"]), "--qa", str(world["qa_path"]), "--out-dir", str(out), "--resamples", "50"]) == 0
    for f in ("breakdown_subfield.csv", "breakdown_tag.csv", "breakdown_subfield_paired.csv", "breakdown_tag_paired.csv", "breakdown_groups.csv"):
        assert (out / f).exists(), f
    text = (out / "breakdown_groups.csv").read_text()
    assert "subfield:Materials Science" in text and "tag:qword:what" in text and "subfield:unlabelled" in text

"""pipeline_lib/question_features.py: the two automatic, rule-based features (never question types)."""
import pytest

from pipeline_lib import question_features as QF


def feats(q):
    return QF.question_features(q)


@pytest.mark.parametrize("q", ["What happens at 300 K?", "Which film has 12 layers?", "What shifts by 25 nm?", "What shifts by 25nm?", "Which sample reaches 90%?",
                               "How is 10 mg dissolved?", "What happens at 25 °C?", "Which peak lies at 0.5 eV?", "What is the yield between 5-10 %?",
                               "What changed in 2019?", "What is the ratio at 2,5 mol?", "Which pH 7 buffer?"])
def test_numbers_and_units(q):
    assert feats(q).auto_has_number, q


@pytest.mark.parametrize("q", ["What is the yield?", "Which catalyst gives CO2?", "What does H2O do?", "What is the 3D structure?", "How does 1H NMR show it?",
                               "What does 13C NMR show?", "Which SO4^2- ions?", "What is 1-butanol used for?", "What is 2-methylpentane?", "What does SARS-CoV-2 do?",
                               "What is Ti3C2?", "What is C60?", "Why does (NH4)2SO4 dissolve?", "Why does Ca(OH)2 dissolve?"])
def test_digits_in_names_and_formulas_are_not_numbers(q):
    assert not feats(q).auto_has_number, q


def test_number_tokens_are_reported():
    assert feats("What shifts by 25 nm at 300 K?").number_tokens == ("25 nm", "300 K")
    assert feats("What is 0.5 eV?").number_tokens == ("0.5 eV",)


@pytest.mark.parametrize("q,tok", [("What is the density of H2O?", "H2O"), ("Why does NaCl dissolve?", "NaCl"), ("Which catalyst gives CO2?", "CO2"),
                                   ("What is the role of Fe(II) here?", "Fe(II)"), ("How does SO4^2- bind?", "SO4^2-"), ("How does SO₄²⁻ bind?", "SO4^2-"),
                                   ("Why does TiO₂ absorb?", "TiO2"), ("Which Fe3+ site?", "Fe3+"), ("What does Ca(OH)2 do?", "Ca(OH)2"),
                                   ("What does (NH4)2SO4 do?", "(NH4)2SO4"), ("Which NaCl-based salt?", "NaCl"), ("How does LiFePO4 age?", "LiFePO4"),
                                   ("What is the role of O2 here?", "O2"), ("What reacts with CO?", "CO")])
def test_formulas(q, tok):
    f = feats(q)
    assert f.auto_has_formula and tok in f.formula_tokens, (q, f)


@pytest.mark.parametrize("q", ["What does UV light do?", "What is PVC used for?", "What does NMR show?", "Which IR band appears?", "What does the NIR spectrum show?",
                               "In this study, what is measured?", "As shown, what happens?", "No sample was used. What happened?", "He measured the yield. Why?",
                               "What is the DNA yield?", "What is the role of Fe nanoparticles?", "How does the (SF) process work?", "What is the VOC level?",
                               "Which type II sample?", "What does SARS-CoV-2 do?", "What is the role of PEG?"])
def test_abbreviations_and_words_are_not_formulas(q):
    assert not feats(q).auto_has_formula, (q, feats(q))


def test_a_listed_abbreviation_is_excluded_explicitly():
    for a in ("UV", "PVC", "NMR", "IR"):
        assert not QF.is_formula(a)
    assert {"UV", "IR", "PVC"} <= QF.NOT_FORMULAS
    for w in ("In", "As", "No", "He", "I", "C", "Fe"):
        assert not QF.is_formula(w), w


def test_features_are_deterministic_and_use_only_the_text():
    q = "What is the yield of Fe3O4 at 25 nm?"
    assert feats(q) == feats(q) and feats(q).auto_has_number and feats(q).auto_has_formula


def test_the_features_never_look_like_a_question_type():
    import dataclasses
    names = [f.name for f in dataclasses.fields(QF.QuestionFeatures)]
    assert all(("type" not in n and "category" not in n) for n in names) and {"auto_has_number", "auto_has_formula"} <= set(names)


def test_span_features_use_the_same_rules_as_the_question_features():
    for text in ["The yield was 85 % at 25 nm.", "CO2 and Fe(II) were used.", "No number or formula here.", "In 2020 the UV lamp (SF) was on."]:
        q, sp = QF.question_features(text), QF.span_features(text)
        assert (q.auto_has_number, q.auto_has_formula) == (sp.auto_span_has_number, sp.auto_span_has_formula)
        assert (q.number_tokens, q.formula_tokens) == (sp.number_tokens, sp.formula_tokens)
    a = QF.all_features("What is H2O?", "Water (H2O) boils at 100 K.")
    assert (a.auto_has_number, a.auto_has_formula, a.auto_span_has_number, a.auto_span_has_formula) == (False, True, True, True)


def test_span_texts_are_cut_from_the_full_text_and_checked_against_content(tmp_path):
    import json
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({"corpus_id": "007", "start_index": 4, "end_index": 9, "content": "CO2 a"}) + "\n")
    (tmp_path / "full_text.jsonl").write_text(json.dumps({"corpus_id": "7", "text": "abc CO2 and 5 K"}) + "\n")
    assert QF.span_texts(qa, [json.loads(qa.read_text())]) == ["CO2 a"]
    qa.write_text(json.dumps({"corpus_id": "7", "start_index": 4, "end_index": 9, "content": "wrong"}) + "\n")
    with pytest.raises(ValueError, match="differs"):
        QF.span_texts(qa, [json.loads(qa.read_text())])

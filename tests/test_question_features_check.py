"""tools/question_features_check.py: the seeded sample for the hand check and the precision / recall of the automatic features."""
import csv
import json

import pytest

from tools import question_features_check as C

NUM_ONLY = ["How long at {} K?", "Which sample at {} nm?", "What happens after {} hours?", "Describe step {} min of annealing."]
FORM_ONLY = ["What does {} do?"]
BOTH = ["Why does NaCl change at {} K?", "How does H2O behave at {} nm?"]
NEITHER = ["What is the role of the ligand {}?", "Why does the catalyst fail {}?"]


def questions(n_each=40):
    qs = []
    for i in range(n_each):
        qs += [NUM_ONLY[i % 4].format(10 + i), FORM_ONLY[0].format(["CO2", "TiO2", "Fe(II)", "NaOH", "H2O"][i % 5]), BOTH[i % 2].format(20 + i),
               NEITHER[i % 2].format("x" * (i % 3 + 1)) + f" ({'abc'[i % 3]})"]
    return qs


def test_sample_has_25_flagged_and_25_not_flagged_per_feature():
    qs = questions()
    ids = C.draw_sample(qs, 50, seed=7)
    assert len(ids) == len(set(ids)) == 50
    for feat in C.FEATURES:
        flagged = sum(C.flags(qs[i])[feat] for i in ids)
        assert flagged == 25 and 50 - flagged == 25
    assert ids == C.draw_sample(qs, 50, seed=7) and ids != C.draw_sample(qs, 50, seed=8)


def test_infeasible_sample_is_an_error():
    with pytest.raises(ValueError, match="cannot draw"):
        C.draw_sample(["What is it?"] * 100, 50, seed=1)


def test_sheet_has_empty_manual_columns_and_does_not_show_the_automatic_flags(tmp_path):
    qs = questions()
    path = C.write_sheet(tmp_path / "s.csv", qs, C.draw_sample(qs, 50, 1))
    rows = list(csv.DictReader(open(path)))
    assert list(rows[0]) == ["query_id", "question", "manual_has_number", "manual_has_formula"] and len(rows) == 50
    assert all(r["manual_has_number"] == "" and r["manual_has_formula"] == "" for r in rows)
    assert not any(k.startswith("auto_") for k in rows[0])
    assert rows[0]["question"] == qs[int(rows[0]["query_id"])]


def test_unfilled_or_invalid_sheet_is_refused(tmp_path):
    qs = questions()
    path = C.write_sheet(tmp_path / "s.csv", qs, C.draw_sample(qs, 50, 1))
    with pytest.raises(ValueError, match="not filled in"):
        C.score(path)
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["manual_has_number"] = r["manual_has_formula"] = "maybe"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError, match="expected 1 / 0"):
        C.score(path)


def write_filled(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=C.SHEET_COLUMNS)
        w.writeheader()
        for i, (q, n, fo) in enumerate(rows):
            w.writerow({"query_id": i, "question": q, "manual_has_number": n, "manual_has_formula": fo})


def test_precision_recall_and_agreement_on_a_filled_fixture(tmp_path):
    # number flag: 4 flagged questions of which 3 truly have a number (1 FP: "compound" label counted by hand as no); 2 unflagged, 1 of them has a number by hand (FN)
    rows = [("What happens at 300 K?", "1", "0"), ("Which film has 12 layers?", "yes", "0"), ("What shifts by 25 nm?", "1", "0"), ("What is compound 1?", "0", "0"),
            ("What is the yield?", "1", "0"), ("What is the role of the ligand?", "0", "0"),
            # formula flag: NaCl flagged and true, "UV" not flagged and not a formula, Fe nanoparticles not flagged but marked by hand as a formula (FN)
            ("Why does NaCl dissolve?", "0", "1"), ("What does UV do?", "0", "0"), ("What are Fe nanoparticles?", "0", "1")]
    write_filled(tmp_path / "f.csv", rows)
    res = {r["feature"]: r for r in C.score(tmp_path / "f.csv")}
    n = res["auto_has_number"]
    assert (n["tp"], n["fp"], n["fn"], n["tn"], n["n"]) == (3, 1, 1, 4, 9)                    # flagged by the rules: 300 K, 12, 25 nm, compound 1
    assert n["precision"] == pytest.approx(3 / 4) and n["recall"] == pytest.approx(3 / 4) and n["agreement"] == pytest.approx(7 / 9)
    f = res["auto_has_formula"]
    assert (f["tp"], f["fp"], f["fn"], f["tn"]) == (1, 0, 1, 7)
    assert f["precision"] == 1.0 and f["recall"] == 0.5 and f["agreement"] == pytest.approx(8 / 9) and f["feature_source"] == "rule_based"


def test_weighted_scores_use_the_population(tmp_path):
    qa = tmp_path / "qa.jsonl"
    qs = ["What at 5 K?"] * 10 + ["What is it?"] * 90
    qa.write_text("".join(json.dumps({"question": q}) + "\n" for q in qs))
    rows = [("What at 5 K?", "1", "0")] * 2 + [("What is it?", "1", "0")] * 1 + [("What is it?", "0", "0")] * 1       # 2 of 2 flagged right; 1 of 2 unflagged is a FN
    write_filled(tmp_path / "f.csv", rows)
    r = {x["feature"]: x for x in C.score(tmp_path / "f.csv", qa)}["auto_has_number"]
    assert r["recall"] == pytest.approx(2 / 3) and r["precision"] == 1.0
    # weights: 10 flagged / 2 sampled = 5 each; 90 unflagged / 2 sampled = 45 each: TP 10, FN 45 -> weighted recall 10 / 55
    assert r["weighted_recall"] == pytest.approx(10 / 55) and r["weighted_precision"] == 1.0


def test_cli_draw_and_score(tmp_path, capsys):
    qa = tmp_path / "qa.jsonl"
    qs = questions()
    doc, rows, pos = "", [], 0
    for q in qs:                                                 # the evidence span of a question is its own sentence: the span flags equal the question flags
        rows.append({"question": q, "corpus_id": "0", "start_index": pos, "end_index": pos + len(q), "content": q})
        doc, pos = doc + q + " ", pos + len(q) + 1
    qa.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "full_text.jsonl").write_text(json.dumps({"corpus_id": "0", "text": doc}) + "\n")
    assert C.main(["draw", "--qa", str(qa), "--out-dir", str(tmp_path / "v")]) == 0
    sheet = tmp_path / "v" / "question_features_check.csv"
    rows = list(csv.DictReader(open(sheet)))
    assert len(rows) == 50
    for r in rows:                                                                        # fill in as the rules say: a perfect annotator
        f = C.flags(r["question"])
        r["manual_has_number"], r["manual_has_formula"] = str(int(f["number"])), str(int(f["formula"]))
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    assert C.main(["score", "--sheet", str(sheet), "--qa", str(qa)]) == 0
    out = capsys.readouterr().out
    assert "auto_has_number: n=50" in out and "precision=1.000 recall=1.000" in out
    assert (tmp_path / "v" / "question_features_check_scores.csv").exists()


def test_span_sheet_is_drawn_blind_and_scored_with_the_span_rules(tmp_path, capsys):
    qa = tmp_path / "qa.jsonl"
    qs = questions()
    doc, rows, pos = "", [], 0
    for q in qs:
        rows.append({"question": q, "corpus_id": "0", "start_index": pos, "end_index": pos + len(q), "content": q})
        doc, pos = doc + q + " ", pos + len(q) + 1
    qa.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "full_text.jsonl").write_text(json.dumps({"corpus_id": "0", "text": doc}) + "\n")
    assert C.main(["draw", "--qa", str(qa), "--out-dir", str(tmp_path / "v")]) == 0
    sheet = tmp_path / "v" / "span_features_check.csv"
    srows = list(csv.DictReader(open(sheet)))
    assert len(srows) == 50 and list(srows[0]) == C.SPAN_SHEET_COLUMNS and not any(r["manual_span_has_number"] or r["manual_span_has_formula"] for r in srows)
    spans = C.read_spans(qa)
    for feat in C.FEATURES:                                      # 25 flagged and 25 not flagged for each feature, by the SPAN rules
        assert sum(C.flags(spans[int(r["query_id"])], "span")[feat] for r in srows) == 25
    for r in srows:
        f = C.flags(r["span_text"], "span")
        r["manual_span_has_number"], r["manual_span_has_formula"] = str(int(f["number"])), str(int(f["formula"]))
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)
    assert C.main(["score", "--sheet", str(sheet), "--qa", str(qa), "--kind", "span"]) == 0
    assert "auto_span_has_number: n=50" in capsys.readouterr().out and (tmp_path / "v" / "span_features_check_scores.csv").exists()
    assert draw_is_seeded(qa, tmp_path)


def draw_is_seeded(qa, tmp_path):
    spans = C.read_spans(qa)
    return C.draw_sample(spans, 50, 7, "span") == C.draw_sample(spans, 50, 7, "span") != C.draw_sample(spans, 50, 8, "span")

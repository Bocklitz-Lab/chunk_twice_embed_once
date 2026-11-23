"""B9: tools/validation_sample.py (stratified sample, sheets) and tools/kappa.py (agreement of two annotators)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tools import kappa as K, validation_sample as V

CONFIGS = [{"chunker": "fixed_token", "size": 128, "overlap": 0}, {"chunker": "fixed_token", "size": 512, "overlap": 0},
           {"chunker": "hierarchical_section", "size": 512, "overlap": 0}]


# ---- allocation and sampling --------------------------------------------------------------------------------
def test_allocation_gives_every_stratum_its_minimum_then_proportional():
    sizes = {"a": 400, "b": 300, "c": 200, "d": 90, "e": 8, "f": 1}
    a = V.allocate(sizes, 100, 2)
    assert sum(a.values()) == 100 and a["f"] == 1 and a["e"] >= 2 and all(a[s] >= 2 for s in "abcde")
    assert a["a"] > a["b"] > a["c"] > a["d"] and all(a[s] <= sizes[s] for s in sizes)


def test_allocation_when_the_minimum_does_not_fit_or_the_data_is_small():
    sizes = {f"s{i}": 10 for i in range(8)}
    a = V.allocate(sizes, 10, 2)                                     # 8 x 2 = 16 > 10: the minimum is lowered
    assert sum(a.values()) == 10 and max(a.values()) <= 2 and min(a.values()) >= 1
    assert sum(V.allocate({"a": 3, "b": 4}, 100, 2).values()) == 7   # asks for more than exists: everything


def _qa(n_papers=30, per=4):
    return [{"question": f"q{p}-{j}", "corpus_id": f"{p:03d}", "content": "w", "start_index": 0, "end_index": 1} for p in range(n_papers) for j in range(per)]


def _fields(n_papers=30):
    names = ["Organic", "Physical", "Analytical", "Materials", "Polymer"]
    return {f"{p}": names[p % 5] for p in range(1, n_papers)} | {"0": V.UNLABELLED}      # paper 0 has no label (ids are normalised: 000 -> 0)


def test_sample_is_seeded_stratified_and_shuffled():
    qa, sub = _qa(), _fields()
    s1 = V.draw_sample(qa, sub, 20, seed=5, min_per=2)
    assert s1 == V.draw_sample(qa, sub, 20, seed=5, min_per=2) and s1 != V.draw_sample(qa, sub, 20, seed=6, min_per=2)
    assert len(s1) == len(set(s1)) == 20
    strata = [sub.get(str(int(qa[i]["corpus_id"])), V.UNLABELLED) for i in s1]
    assert {s: strata.count(s) for s in set(strata)}.keys() == set(sub.values()) and all(strata.count(s) >= 2 for s in set(strata))
    assert s1 != sorted(s1)                                          # shuffled, not in stratum or file order


def test_first_listed_subfield_and_unlabelled(tmp_path):
    f = tmp_path / "meta.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [{"corpus_id": "7", "fields_of_study": ["Organic Chemistry", "Catalysis"]},
                                                    {"corpus_id": "008", "fields_of_study": []}, {"corpus_id": "9"}]))
    assert V.first_subfields(f) == {"7": "Organic Chemistry", "8": V.UNLABELLED, "9": V.UNLABELLED}


def test_span_context_is_cut_at_the_document_ends():
    assert V.span_with_context("0123456789", 3, 5, 2) == ("12", "34", "56")
    assert V.span_with_context("0123456789", 0, 2, 5) == ("", "01", "23456")
    assert V.span_with_context("0123456789", 8, 10, 5) == ("34567", "89", "")


# ---- end to end on a tiny corpus ------------------------------------------------------------------------------
def _world(tmp_path):
    docs = {str(p): " ".join(f"w{p}_{i}" for i in range(200)) for p in range(1, 7)}
    text_of = lambda p: docs[str(p)]
    qa, meta, full = [], [], []
    for p in range(1, 7):
        meta.append({"corpus_id": f"{p:03d}", "title": f"Paper {p}", "fields_of_study": [["Organic", "Physical", "Analytical"][p % 3], "Other"]})
        full.append({"corpus_id": f"{p:03d}", "text": text_of(p)})
        for j in range(3):
            s = 100 + 40 * j
            qa.append({"question": f"what about {p}-{j}?", "corpus_id": f"{p:03d}", "content": text_of(p)[s:s + 30], "start_index": s, "end_index": s + 30})
    d = tmp_path / "chem"
    d.mkdir()
    for name, rows in (("qa.jsonl", qa), ("metadata.jsonl", meta), ("full_text.jsonl", full)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    prep = tmp_path / "prepared"
    for c in CONFIGS:
        size = c["size"]
        folder = prep / f"{c['chunker']}_c{size}_o0__abc123"
        (folder / "mteb_task").mkdir(parents=True)
        (folder / "chunks").mkdir()
        (folder / "manifest.json").write_text(json.dumps({"key": "abc123"}))
        chunks, qrels = [], []
        step = 60 if size == 128 else 140
        for p in range(1, 7):
            t = text_of(p)
            for k, a in enumerate(range(0, len(t), step)):
                chunks.append({"chunk_id": f"{p}:{k}", "source_doc_id": str(p), "start_char": a, "end_char": min(len(t), a + step), "text": t[a:a + step]})
        for qid, r in enumerate(qa):
            p = int(r["corpus_id"])
            for ch in chunks:
                if ch["source_doc_id"] == str(p) and max(ch["start_char"], r["start_index"]) < min(ch["end_char"], r["end_index"]):
                    qrels.append({"query_id": str(qid), "doc_id": ch["chunk_id"], "relevance": 1})
        (folder / "chunks" / "chunks.jsonl").write_text("".join(json.dumps(c_) + "\n" for c_ in chunks))
        (folder / "mteb_task" / "qrels_span_c090.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qrels))
    return d, prep


def test_end_to_end_sheets(tmp_path):
    d, prep = _world(tmp_path)
    out = tmp_path / "out"
    summ = V.run(d / "qa.jsonl", d / "metadata.jsonl", d / "full_text.jsonl", prep, out, n=9, seed=3, min_per=2, configs=CONFIGS, context_chars=40,
                 annotators=["ann_a", "ann_b"], cover_level=0.9)
    assert summ["n_questions"] == 9 and set(summ["questions_per_subfield"]) == {"Organic", "Physical", "Analytical"} and min(summ["questions_per_subfield"].values()) >= 2
    master = list(csv.DictReader(open(out / "validation_sample.csv")))
    a, b = (list(csv.DictReader(open(out / f"validation_sheet_{n}.csv"))) for n in ("ann_a", "ann_b"))
    assert [r["query_id"] for r in a] == [r["query_id"] for r in b] == [r["query_id"] for r in master] == summ["query_ids"]
    r = master[0]
    for col in ("span_correct", "span_sufficient", "chunks_sufficient_fixed_token_c128_o0", "chunks_sufficient_fixed_token_c512_o0",
                "chunks_sufficient_hierarchical_section_c512_o0", "comment"):
        assert col in r and r[col] == ""                                    # the answers are empty
    assert len(r["context_before"]) <= 40 and len(r["context_after"]) <= 40 and r["span_text"] and r["paper_title"].startswith("Paper")
    qa_rows = [json.loads(l) for l in open(d / "qa.jsonl")]
    q = qa_rows[int(r["query_id"])]
    assert r["span_text"] == q["content"] and r["question"] == q["question"]
    for c in CONFIGS:
        n = V.cfg_name(c)
        assert int(r[f"chunks_{n}_n"]) >= 1 and r[f"chunks_{n}_text"] and 0 < float(r[f"chunks_{n}_share_of_span"]) <= 1.0001
        assert r[f"chunks_{n}_text"].count("--- next labelled chunk ---") == int(r[f"chunks_{n}_n"]) - 1
    md = (out / "validation_sheet_ann_a.md").read_text()
    assert "⟦" in md and "Item 1 " in md and "Labelled chunks, fixed_token_c128_o0" in md and md.count("## Item") == 9
    assert json.loads((out / "validation_sample_summary.json").read_text())["configurations"]["fixed_token_c128_o0"]["manifest_sha256"]
    # same seed: the same sample
    out2 = tmp_path / "out2"
    V.run(d / "qa.jsonl", d / "metadata.jsonl", d / "full_text.jsonl", prep, out2, n=9, seed=3, min_per=2, configs=CONFIGS, context_chars=40, annotators=["x"], cover_level=0.9)
    assert (out2 / "validation_sample.csv").read_text() == (out / "validation_sample.csv").read_text()


def test_prepared_folder_must_be_unique(tmp_path):
    _, prep = _world(tmp_path)
    assert V.find_prepared(prep, CONFIGS[0]).name == "fixed_token_c128_o0__abc123"
    with pytest.raises(FileNotFoundError):
        V.find_prepared(prep, {"chunker": "fixed_token", "size": 192, "overlap": 0})
    dup = prep / "fixed_token_c128_o0__ffffff"
    (dup / "mteb_task").mkdir(parents=True)
    (dup / "manifest.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="2 complete"):
        V.find_prepared(prep, CONFIGS[0])


def test_the_configured_defaults_exist():
    s = V.settings()
    assert s["n_questions"] == 100 and s["min_per_subfield"] == 2 and s["context_chars"] == 300 and len(s["annotators"]) == 2
    assert [V.cfg_name(c) for c in s["configurations"]] == ["fixed_token_c128_o0", "fixed_token_c512_o0", "hierarchical_section_c512_o0"]


# ---- kappa ----------------------------------------------------------------------------------------------------
def _sheet(path, answers, papers=None, comments=None, extra_cols=("span_sufficient",)):
    cols = ["item", "query_id", "corpus_id", "question", "span_correct", *extra_cols, "comment"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, a in enumerate(answers):
            row = {"item": i + 1, "query_id": str(i), "corpus_id": (papers or [str(i // 2)] * len(answers))[i], "question": f"q{i}", "span_correct": a,
                   "comment": (comments or [""] * len(answers))[i]}
            for c in extra_cols:
                row[c] = a
            w.writerow(row)


def test_kappa_known_value_and_weighted_equals_unweighted_for_two_levels():
    counts = np.array([[20, 5, 0], [10, 15, 0], [0, 0, 0]])
    assert K.kappa_from_counts(counts, K._weights("none")) == pytest.approx(0.4)
    assert K.kappa_from_counts(counts, K._weights("linear")) == pytest.approx(0.4)


def test_kappa_matches_sklearn_and_penalises_far_disagreements_less_when_weighted():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    a = rng.integers(0, 3, 200)
    b = np.where(rng.random(200) < .6, a, rng.integers(0, 3, 200))
    cm = K.confusion(a, b)
    assert K.kappa_from_counts(cm, K._weights("none")) == pytest.approx(sk.cohen_kappa_score(a, b))
    assert K.kappa_from_counts(cm, K._weights("linear")) == pytest.approx(sk.cohen_kappa_score(a, b, weights="linear"))


def test_compare_two_sheets(tmp_path):
    ans_a = ["yes"] * 10 + ["partial"] * 6 + ["no"] * 4
    ans_b = ["yes"] * 8 + ["partial"] * 2 + ["partial"] * 5 + ["yes"] + ["no"] * 3 + ["partial"]
    papers = [str(i // 2) for i in range(20)]
    _sheet(tmp_path / "a.csv", ans_a, papers, comments=["c_a"] * 20)
    _sheet(tmp_path / "b.csv", ans_b, papers, comments=["c_b"] * 20)
    rows, dis, invalid = K.compare(tmp_path / "a.csv", tmp_path / "b.csv", 300, 1)
    r = {x["column"]: x for x in rows}["span_correct"]
    assert r["n"] == 20 and r["percent_agreement"] == pytest.approx(100 * 16 / 20) and 0 < r["kappa"] < 1
    assert r["kappa_ci_low"] <= r["kappa"] <= r["kappa_ci_high"] and r["kappa_weighted_ci_low"] <= r["kappa_weighted"] <= r["kappa_weighted_ci_high"]
    assert r["n_papers"] == 10 and r["A_yes__B_yes"] == 8 and r["A_yes__B_partial"] == 2
    assert len(dis) == 2 * 4 and {d["column"] for d in dis} == {"span_correct", "span_sufficient"} and dis[0]["comment_A"] == "c_a" and dis[0]["comment_B"] == "c_b"
    assert invalid == []


def test_undefined_kappa_empty_and_invalid_answers(tmp_path):
    _sheet(tmp_path / "a.csv", ["yes"] * 6)
    _sheet(tmp_path / "b.csv", ["yes", "yes", "Yes ", "maybe", "", "yes"])
    rows, dis, invalid = K.compare(tmp_path / "a.csv", tmp_path / "b.csv", 50, 1)
    r = rows[0]
    assert r["n"] == 4 and r["percent_agreement"] == 100.0 and np.isnan(r["kappa"])           # one level only: chance agreement 1
    assert len(invalid) == 2 and "maybe" in invalid[0]                                         # two columns carry the invalid answer
    assert dis == []


def test_sheets_must_hold_the_same_items(tmp_path):
    _sheet(tmp_path / "a.csv", ["yes", "no"])
    _sheet(tmp_path / "b.csv", ["yes", "no", "no"])
    with pytest.raises(ValueError, match="same items"):
        K.compare(tmp_path / "a.csv", tmp_path / "b.csv", 10, 1)


def test_cli_writes_both_tables(tmp_path, capsys):
    _sheet(tmp_path / "a.csv", ["yes", "no", "partial", "yes", "no", "yes"])
    _sheet(tmp_path / "b.csv", ["yes", "no", "no", "yes", "yes", "yes"])
    assert K.main(["--sheets", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--out-dir", str(tmp_path / "o"), "--resamples", "100"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "o" / "kappa.csv")))
    assert [r["column"] for r in rows] == ["span_correct", "span_sufficient"] and float(rows[0]["percent_agreement"]) == pytest.approx(100 * 4 / 6, abs=0.01)
    assert len(list(csv.DictReader(open(tmp_path / "o" / "kappa_disagreements.csv")))) == 4
    assert "agreement" in capsys.readouterr().out

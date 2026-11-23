"""B8: table_label_stats.csv from label_stats_c*.json and coverage_diagnostics.csv of every prepared configuration."""
import csv
import json
from pathlib import Path

import pytest

from tools import coverage_diagnostics as CD, label_stats_table as LS


def _label_stats(c, s1, s2, s3, s0=0, n=100, mean=1.2, mx=3, empty=0, n_chunks=500):
    return {"cover_level": c, "n_questions": n, "n_chunks": n_chunks, "questions_per_step": {"step1": s1, "step2": s2, "step3": s3, "step0_no_overlapping_chunk": s0},
            "labels_per_question": {"mean": mean, "max": mx, "total": int(mean * n)}, "n_empty_questions": empty}


def _prepared(root, name="fixed_token_c128_o0__abcd1234", median=True, levels=((0.9, 60, 38, 2),), manifest=True, chunker="fixed_token", size=128):
    d = root / name
    (d / "mteb_task").mkdir(parents=True)
    (d / "chunks").mkdir()
    if manifest:
        (d / "manifest.json").write_text(json.dumps({"chunker": chunker, "size": size, "overlap": 0, "tokenizer": "bert-base-uncased", "tokenizer_revision": "r"}))
    cols = ["n_chunks", "mean_chunk_tokens", "max_chunk_tokens", "pct_span_inside_one_chunk"] + (["median_chunk_tokens"] if median else [])
    vals = [500, 120.5, 130, 55.0] + ([128.0] if median else [])
    with open(d / "coverage_diagnostics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerow(vals)
    for c, s1, s2, s3 in levels:
        (d / "mteb_task" / f"label_stats_c{int(c * 100):03d}.json").write_text(json.dumps(_label_stats(c, s1, s2, s3)))
    (d / "chunks" / "chunks.jsonl").write_text("".join(json.dumps({"chunk_id": f"0:{i}", "text": "w " * (i + 1)}) + "\n" for i in range(5)))
    return d


def test_rows_have_the_requested_columns_and_values(tmp_path):
    _prepared(tmp_path)
    rows = LS.build_table(tmp_path)
    assert len(rows) == 1 and list(rows[0]) == LS.COLUMNS
    r = rows[0]
    assert (r["chunker"], r["size"], r["overlap"], r["cover_level"]) == ("fixed_token", 128, 0, 0.9)
    assert r["n_chunks"] == 500 and r["chunk_tokens_median"] == 128.0 and r["chunk_tokens_mean"] == 120.5 and r["chunk_tokens_max"] == 130.0
    assert (r["questions_step1"], r["questions_step2"], r["questions_step3"], r["questions_step0"]) == (60, 38, 2, 0)
    assert r["pct_split_over_chunks"] == pytest.approx(45.0) and r["pct_step2_or_3"] == pytest.approx(40.0)
    assert r["mean_relevant_per_question"] == 1.2 and r["max_relevant_per_question"] == 3 and r["n_empty_questions"] == 0


def test_every_cover_level_present_gets_a_row_and_rows_are_sorted(tmp_path):
    _prepared(tmp_path, levels=((0.8, 70, 30, 0), (0.9, 60, 38, 2), (1.0, 50, 30, 20)))
    _prepared(tmp_path, name="fixed_token_c64_o0__ffff0000", size=64)
    rows = LS.build_table(tmp_path)
    assert [(r["size"], r["cover_level"]) for r in rows] == [(64, 0.9), (128, 0.8), (128, 0.9), (128, 1.0)]


def test_missing_median_is_counted_from_the_chunks_with_the_manifest_tokenizer(tmp_path, monkeypatch):
    _prepared(tmp_path, median=False)
    seen = {}

    def fake(texts, name, rev):
        seen.update(name=name, rev=rev, n=len(texts))
        return [len(t.split()) for t in texts]

    monkeypatch.setattr(CD, "_token_lengths", fake)
    r = LS.build_table(tmp_path)[0]
    assert seen == {"name": "bert-base-uncased", "rev": "r", "n": 5} and r["chunk_tokens_median"] == 3.0
    assert r["chunk_tokens_mean"] == 120.5 and r["chunk_tokens_max"] == 130.0              # values the diagnostics have are kept
    none = LS.build_table(tmp_path, recompute_lengths=False)[0]
    assert none["chunk_tokens_median"] is None


def test_incomplete_folders_and_missing_roots_are_skipped(tmp_path):
    _prepared(tmp_path, name="fixed_token_c128_o0__aaaa0000", manifest=False)
    assert LS.build_table(tmp_path) == [] and LS.build_table(tmp_path / "nothing") == []


def test_folder_name_gives_the_configuration_when_the_manifest_lacks_it(tmp_path):
    d = _prepared(tmp_path, name="hierarchical_section_c512_o0__12345678")
    (d / "manifest.json").write_text("{}")
    r = LS.build_table(tmp_path)[0]
    assert (r["chunker"], r["size"], r["overlap"]) == ("hierarchical_section", 512, 0)


def test_cli_writes_the_csv(tmp_path):
    _prepared(tmp_path / "prep")
    out = tmp_path / "out" / "table_label_stats.csv"
    assert LS.main(["--prepared-root", str(tmp_path / "prep"), "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0]) == LS.COLUMNS and rows[0]["questions_step2"] == "38"


def test_diagnostics_now_carry_the_median(tmp_path):
    texts = ["a b", "a b c d", "a b c d e f"]
    task = tmp_path / "task"
    task.mkdir()
    (task / "queries.jsonl").write_text(json.dumps({"id": "0", "text": "q"}) + "\n")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("".join(json.dumps({"chunk_id": f"0:{i}", "source_doc_id": "0", "text": t}) + "\n" for i, t in enumerate(texts)))
    orig = CD._token_lengths
    CD._token_lengths = lambda t, n, r: [len(x.split()) for x in t]
    try:
        row = CD.diagnose_config(task, chunks, "tok", "rev")
    finally:
        CD._token_lengths = orig
    assert row["median_chunk_tokens"] == 4.0 and row["mean_chunk_tokens"] == 4.0 and row["max_chunk_tokens"] == 6


def test_make_report_writes_the_table(tmp_path):
    from test_make_report import _run
    from tools import make_report as R
    root = tmp_path / "art"
    (root / "tests").mkdir(parents=True)
    task = _prepared(root / "prepared") / "mteb_task"
    # make_report also compares the lexical and the span labels of every prepared folder: give it a complete folder
    (task / "queries.jsonl").write_text(json.dumps({"id": "0", "title": "", "text": "q"}) + "\n")
    (task / "corpus.jsonl").write_text(json.dumps({"id": "0:0", "title": "", "text": "w"}) + "\n")
    (task / "qrels_lexical.jsonl").write_text(json.dumps({"query_id": "0", "doc_id": "0:0", "relevance": 1, "score": 1.0}) + "\n")
    (task / "qrels_span_c090.jsonl").write_text(json.dumps({"query_id": "0", "doc_id": "0:0", "relevance": 1}) + "\n")
    (task / "overlap_table.jsonl").write_text(json.dumps({"query_id": "0", "chunk_id": "0:0", "overlap_chars": 1}) + "\n")
    qa = root / "qa.jsonl"
    qa.write_text(json.dumps({"question": "q", "corpus_id": "0", "content": "q", "start_index": 0, "end_index": 1}) + "\n")
    _run(root / "tests", "bm25_fixed_token_c128_o0", "bm25", "fixed_token", 128, {"0": (0.5, 0.5)})
    out = tmp_path / "out"
    m = R.build_report(root, out, qa, resamples=5, overlap=False, verbose=False)
    rows = list(csv.DictReader(open(out / "table_label_stats.csv")))
    assert rows[0]["questions_step2"] == "38" and "table_label_stats.csv" in m["files"]

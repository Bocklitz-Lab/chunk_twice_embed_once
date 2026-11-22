"""stages/make_span_qrels.py on a small synthetic document set (integration), cover levels and failure cases."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_lib.dataset_pin import sha256_file  # noqa: E402
from stages.make_span_qrels import SpanQrelsError, build_span_qrels  # noqa: E402


def _words(n):
    return "".join(f"w{i:03d} " for i in range(n))  # 5 characters per word


DOCS = {"0": _words(60), "1": _words(60), "2": _words(56)}  # 300, 300, 280 characters


def _chunk_spec():
    """(doc, [a, b)) per chunk. Doc 0: no overlap; doc 1: a gap between the chunks; doc 2: overlap."""
    return {
        "0": [(0, 100), (100, 200), (200, 300)],
        "1": [(0, 100), (130, 230)],  # [100, 130) is in no chunk
        "2": [(0, 120), (80, 200), (160, 280)],  # overlap zones [80,120) and [160,200)
    }


# question index -> (doc, start, end): the comment says what it is for
QUESTIONS = [
    ("0", 110, 150),  # q0: inside one chunk (c1)
    ("0", 80, 140),   # q1: split 20 / 40 characters over c0 and c1
    ("1", 90, 140),   # q2: crosses the gap: 10 characters in c0, 10 in c1, 30 in the gap
    ("1", 250, 290),  # q3: no chunk overlaps the span
    ("2", 90, 110),   # q4: inside the overlap region of c0 and c1
    ("2", 100, 180),  # q5: 20 of 80 in c0 ... c1 holds [100,180) completely
    ("0", 0, 10),     # q6: first characters of the document
    ("0", 190, 210),  # q7: split exactly 10 / 10
]


def _build(tmp_path, chunks=None, questions=None, qa_overrides=None, docs=None, name="data"):
    d = tmp_path / name
    d.mkdir()
    docs = docs or DOCS
    questions = questions or QUESTIONS
    with open(d / "full_text.jsonl", "w") as f:
        for doc, text in docs.items():
            f.write(json.dumps({"corpus_id": doc, "text": text}) + "\n")
    with open(d / "qa.jsonl", "w") as f:
        for i, (doc, s, e) in enumerate(questions):
            row = {"question": f"What happens in passage number {i}?", "corpus_id": doc,
                   "content": docs[doc][s:e], "start_index": s, "end_index": e}
            row.update((qa_overrides or {}).get(i, {}))
            f.write(json.dumps(row) + "\n")
    spec = chunks or _chunk_spec()
    with open(d / "chunks.jsonl", "w") as f:
        for doc, ivs in spec.items():
            for k, (a, b) in enumerate(ivs):
                f.write(json.dumps({"source_doc_id": doc, "chunk_index": k, "chunk_id": f"{doc}:{k}",
                                    "text": docs[doc][a:b], "start_char": a, "end_char": b}) + "\n")
    return d


def _run(d, *extra, check=True):
    out = d / "out"
    cmd = [sys.executable, "-m", "stages.make_span_qrels", "--qa-jsonl", str(d / "qa.jsonl"),
           "--chunks-jsonl", str(d / "chunks.jsonl"), "--full-text-jsonl", str(d / "full_text.jsonl"),
           "--output-folder", str(out), "--qa-sha256", sha256_file(d / "qa.jsonl"), *extra]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check:
        assert p.returncode == 0, p.stdout + p.stderr
    return p, out


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _labels(out, name):
    res = {}
    for r in _jsonl(out / name):
        res.setdefault(int(r["query_id"]), {})[r["doc_id"]] = r["relevance"]
    return res


def test_outputs_for_three_cover_levels(tmp_path):
    d = _build(tmp_path)
    p, out = _run(d, "--cover-level", "0.8", "0.9", "1.0")
    names = sorted(x.name for x in out.iterdir())
    assert names == sorted([
        "corpus.jsonl", "queries.jsonl", "overlap_table.jsonl",
        "qrels_span_c080.jsonl", "qrels_span_c090.jsonl", "qrels_span_c100.jsonl",
        "label_stats_c080.json", "label_stats_c090.json", "label_stats_c100.json",
    ])

    # queries: id = row index, text = the question only
    q = _jsonl(out / "queries.jsonl")
    assert [x["id"] for x in q] == [str(i) for i in range(len(QUESTIONS))]
    assert q[3]["text"] == "What happens in passage number 3?"
    for i, (doc, s, e) in enumerate(QUESTIONS):
        assert DOCS[doc][s:e] not in q[i]["text"]

    # corpus: same format as the lexical stage 5
    corpus = _jsonl(out / "corpus.jsonl")
    assert len(corpus) == 8 and corpus[0] == {"id": "0:0", "title": "", "text": DOCS["0"][0:100]}   # empty title: embed_chunk_title is false
    assert [c["id"] for c in corpus] == ["0:0", "0:1", "0:2", "1:0", "1:1", "2:0", "2:1", "2:2"]

    # qrels at c = 0.9 (a chunk with >= 90 % of the span, else greedy, else all overlapping)
    lab = _labels(out, "qrels_span_c090.jsonl")
    assert lab[0] == {"0:1": 1}                       # inside one chunk
    assert set(lab[1]) == {"0:0", "0:1"}              # split 20/40: both (step 2)
    assert set(lab[2]) == {"1:0", "1:1"} and all(v == 1 for v in lab[2].values())  # gap: step 3, both overlapping
    assert list(lab[3].values()) == [0]               # no overlapping chunk: relevance-0 row
    assert set(lab[4]) == {"2:0", "2:1"}              # span in the overlap region: both chunks
    assert set(lab[5]) == {"2:1"}                     # [100,180) lies in c1 = [80,200)
    assert set(lab[6]) == {"0:0"}
    assert set(lab[7]) == {"0:1", "0:2"}              # split exactly 10/10 -> both
    assert all(v in (0, 1) for qq in lab.values() for v in qq.values())

    # every query has at least one row, and there is exactly one zero row (query 3)
    assert set(lab) == set(range(8))
    assert [i for i, qq in lab.items() if not any(qq.values())] == [3]

    # c = 0.8 and 1.0 differ where they should: q1 has 40/60 = 67 % in c1: step 2 at 0.8 and 1.0
    lab80, lab100 = _labels(out, "qrels_span_c080.jsonl"), _labels(out, "qrels_span_c100.jsonl")
    assert set(lab80[1]) == {"0:0", "0:1"} and set(lab100[1]) == {"0:0", "0:1"}
    assert set(lab80[2]) == {"1:0", "1:1"} and set(lab100[2]) == {"1:0", "1:1"}  # unreachable: step 3

    # overlap table: one row per (question, chunk) pair that overlaps the span; independent of c
    table = _jsonl(out / "overlap_table.jsonl")
    keys = [(int(r["query_id"]), r["chunk_id"]) for r in table]
    assert len(keys) == len(set(keys))
    assert (3, "1:1") not in keys and not [k for k in keys if k[0] == 3]
    row = next(r for r in table if r["query_id"] == "1" and r["chunk_id"] == "0:0")
    assert (row["overlap_chars"], row["span_len"], row["contains_span"]) == (20, 60, False)
    assert row["fraction"] == pytest.approx(20 / 60)
    assert (row["overlap_start"], row["overlap_end"], row["chunk_start"], row["chunk_end"]) == (80, 100, 0, 100)
    full = next(r for r in table if r["query_id"] == "0")
    assert (full["overlap_chars"], full["span_len"], full["contains_span"], full["fraction"]) == (40, 40, True, 1.0)
    assert sum(1 for r in table if r["query_id"] == "4") == 2 and all(r["contains_span"] for r in table if r["query_id"] == "4")
    # table does not depend on c
    _, out2 = _run(_build(tmp_path, name="data2"), "--cover-level", "0.5")
    assert (out2 / "overlap_table.jsonl").read_text() == (out / "overlap_table.jsonl").read_text()

    # label stats: c, steps, labels per question, empty questions
    s90 = json.loads((out / "label_stats_c090.json").read_text())
    assert s90["cover_level"] == 0.9 and s90["cover_level_exact"] == "9/10" and s90["cover_tag"] == "c090"
    assert s90["n_questions"] == 8 and s90["n_empty_questions"] == 1 and s90["empty_question_ids"] == [3]
    assert s90["questions_per_step"] == {"step1": 4, "step2": 2, "step3": 1, "step0_no_overlapping_chunk": 1}
    # labels: q0 1, q1 2, q2 2, q3 0, q4 2, q5 1, q6 1, q7 2
    assert s90["labels_per_question"] == {"mean": 11 / 8, "max": 2, "total": 11}
    assert s90["step3_question_ids"] == [2]
    for c, tag in ((0.8, "c080"), (1.0, "c100")):
        st = json.loads((out / f"label_stats_{tag}.json").read_text())
        assert st["cover_level"] == c and st["cover_tag"] == tag


def test_step_counts_move_with_c(tmp_path):
    d = _build(tmp_path)
    _, out = _run(d, "--cover-level", "0.5", "0.7", "0.9", "1.0")
    steps = {}
    for tag in ("c050", "c070", "c090", "c100"):
        st = json.loads((out / f"label_stats_{tag}.json").read_text())["questions_per_step"]
        steps[tag] = (st["step1"], st["step2"], st["step3"])
    s1 = [steps[t][0] for t in ("c050", "c070", "c090", "c100")]
    s23 = [steps[t][1] + steps[t][2] for t in ("c050", "c070", "c090", "c100")]
    assert s1 == sorted(s1, reverse=True) and s1[0] > s1[-1]       # higher c: fewer step 1
    assert s23 == sorted(s23) and s23[0] < s23[-1]                 # higher c: more step 2/3
    # the number of questions per step adds up
    assert all(sum(v) == 7 for v in steps.values())                # 8 questions, 1 without a chunk


def test_a_value_outside_0_1_raises(tmp_path):
    d = _build(tmp_path)
    for bad in ("0", "1.2", "-0.1", "abc"):
        p, out = _run(d, "--cover-level", bad, check=False)
        assert p.returncode != 0, bad
        assert "cover level" in (p.stdout + p.stderr).lower(), p.stderr
    # one bad value among good ones: nothing is written
    p, out = _run(d, "--cover-level", "0.8", "1.5", check=False)
    assert p.returncode != 0 and not out.exists()
    with pytest.raises(ValueError):
        build_span_qrels(str(d / "qa.jsonl"), str(d / "chunks.jsonl"), str(tmp_path / "o"), [0.0],
                         qa_sha256=sha256_file(d / "qa.jsonl"))


def test_exact_boundary_chunk_holding_exactly_c_counts(tmp_path):
    """Span of 50 characters; one chunk holds exactly 45 (= 90 %), another case exactly 44."""
    docs = {"0": _words(60)}
    qs = [("0", 100, 150), ("0", 100, 150)]
    d1 = _build(tmp_path, chunks={"0": [(0, 145), (145, 300)]}, questions=qs[:1], docs=docs, name="exact")
    _, o1 = _run(d1, "--cover-level", "0.9", "0.91")
    assert set(_labels(o1, "qrels_span_c090.jsonl")[0]) == {"0:0"}          # 45/50 >= 9/10: step 1, only c0
    assert set(_labels(o1, "qrels_span_c091.jsonl")[0]) == {"0:0", "0:1"}    # 45/50 < 0.91: step 2 needs both
    d2 = _build(tmp_path, chunks={"0": [(0, 144), (144, 300)]}, questions=qs[:1], docs=docs, name="below")
    _, o2 = _run(d2, "--cover-level", "0.9")
    assert set(_labels(o2, "qrels_span_c090.jsonl")[0]) == {"0:0", "0:1"}    # 44/50 < 9/10


# ------------------------------------------------------------------ failure cases
def test_fails_on_a_chunk_without_positions(tmp_path):
    d = _build(tmp_path)
    rows = _jsonl(d / "chunks.jsonl")
    rows[2]["start_char"] = None
    rows[2]["end_char"] = None
    (d / "chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p, _ = _run(d, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "no exact position" in p.stderr and "0:2" in p.stderr
    # a chunk with no position keys at all
    rows[2].pop("start_char"), rows[2].pop("end_char")
    (d / "chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p, _ = _run(d, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "no exact position" in p.stderr


def test_fails_on_a_chunk_whose_text_does_not_match_its_positions(tmp_path):
    d = _build(tmp_path)
    rows = _jsonl(d / "chunks.jsonl")
    rows[1]["start_char"] += 1  # the position is shifted by one character
    rows[1]["end_char"] += 1
    (d / "chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p, _ = _run(d, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "does not equal the chunk text" in p.stderr and "0:1" in p.stderr
    # the text was changed instead of the position
    d2 = _build(tmp_path, name="d2")
    rows = _jsonl(d2 / "chunks.jsonl")
    rows[0]["text"] = rows[0]["text"].upper()
    (d2 / "chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p, _ = _run(d2, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "does not equal the chunk text" in p.stderr


def test_fails_on_a_span_whose_text_does_not_match_the_full_text(tmp_path):
    d = _build(tmp_path, qa_overrides={5: {"content": "not the text of the span"}})
    p, _ = _run(d, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "question 5" in p.stderr and "span text" in p.stderr
    d2 = _build(tmp_path, qa_overrides={2: {"start_index": 91, "end_index": 141}}, name="d2")  # shifted indices
    p, _ = _run(d2, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "question 2" in p.stderr


def test_fails_on_an_empty_question_or_a_question_containing_the_span(tmp_path):
    d = _build(tmp_path, qa_overrides={1: {"question": "  "}})
    p, _ = _run(d, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "question text is empty" in p.stderr
    span = DOCS["0"][80:140]
    d2 = _build(tmp_path, qa_overrides={1: {"question": f"Explain: {span} please"}}, name="d2")
    p, _ = _run(d2, "--cover-level", "0.8", check=False)
    assert p.returncode != 0 and "contains the evidence span text" in p.stderr


def test_fails_when_qa_is_not_the_pinned_release(tmp_path):
    d = _build(tmp_path)
    cmd = [sys.executable, "-m", "stages.make_span_qrels", "--qa-jsonl", str(d / "qa.jsonl"),
           "--chunks-jsonl", str(d / "chunks.jsonl"), "--full-text-jsonl", str(d / "full_text.jsonl"),
           "--output-folder", str(d / "out"), "--cover-level", "0.8"]  # default pin: the synthetic qa.jsonl differs
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and "SHA256 mismatch" in p.stderr


def test_python_api_raises_span_qrels_error(tmp_path):
    d = _build(tmp_path, qa_overrides={0: {"content": "x"}})
    with pytest.raises(SpanQrelsError):
        build_span_qrels(str(d / "qa.jsonl"), str(d / "chunks.jsonl"), str(tmp_path / "o"), [0.8],
                         qa_sha256=sha256_file(d / "qa.jsonl"), verbose=False)


# ------------------------------------------------------------------ same corpus/queries as the lexical stage 5
def test_queries_and_corpus_equal_the_lexical_stage_5(tmp_path):
    d = _build(tmp_path)
    _, out = _run(d, "--cover-level", "0.8")
    cfg = {"qa_jsonl_file": str(d / "qa.jsonl"), "chunks_jsonl_file": str(d / "chunks.jsonl"),
           "output_folder": str(d / "lex"), "qa_sha256": sha256_file(d / "qa.jsonl")}
    (d / "lex.yaml").write_text(yaml.safe_dump(cfg))
    p = subprocess.run([sys.executable, "-m", "stages.make_mteb_task", "--config", str(d / "lex.yaml")],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    for name in ("queries.jsonl", "corpus.jsonl"):
        assert (out / name).read_bytes() == (d / "lex" / name).read_bytes(), name

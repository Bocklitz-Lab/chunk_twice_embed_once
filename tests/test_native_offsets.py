"""B1: every chunker returns (text, start, end) with exact positions in the original text."""
import json
import os

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")
pytest.importorskip("tqdm")

from transformers import AutoTokenizer

from pipeline_lib.fixed_token_chunker import FixedTokenChunker
from pipeline_lib.hierarchical_section_chunker import HierarchicalSectionChunker
from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker
from pipeline_lib.semantic_chunker import SemanticSplitter
from stages import chunking
from stages.chunking import ChunkOffsetError, check_chunk_offsets, locate_chunk_offsets, process_docs

TOK = os.environ.get("OFFSET_TEST_MODEL", "bert-base-uncased")
SEM = os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# A periodic text: the same 3-word cycle again and again. A forward text search for a chunk finds
# an earlier copy; native positions must not.
PERIODIC = " ".join(["alpha beta gamma"] * 60)
PARA = ("Furfural is produced from biomass. It is used for fuels and polymers. "
        "The yield was measured by UV spectroscopy.")
REPEATED_PARAS = "\n\n".join([PARA, "Unique intro sentence about sugarcane.", PARA, PARA,
                              "Another unique closing remark about costs.", PARA])
TILDES = "Start of the paper. " + "~/" * 400 + " Middle text with words. " + "~/" * 400 + " End."


def _check(text, triples):
    prev = -1
    for chunk, s, e in triples:
        assert chunk and text[s:e] == chunk
        assert s >= prev
        prev = s


def _uncovered_non_ws(text, triples):
    covered = [False] * len(text)
    for _, s, e in triples:
        for i in range(s, e):
            covered[i] = True
    return [i for i, c in enumerate(covered) if not c and not text[i].isspace()]


# ---------------- fixed-token ----------------
@pytest.mark.parametrize("overlap", [0, 2, 5])
def test_fixed_offsets_equal_tokenizer_truth_where_search_fails(overlap):
    ch = FixedTokenChunker(model_name=TOK, chunk_size=8, chunk_overlap=overlap)
    triples = ch.split_text_with_offsets(PERIODIC)
    _check(PERIODIC, triples)
    # independent truth from the tokenizer
    tok = AutoTokenizer.from_pretrained(TOK, use_fast=True)
    offs = tok(PERIODIC, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    step = 8 - overlap
    truth = []
    for t in range(0, len(offs), step):
        a, b = offs[t][0], offs[min(len(offs), t + 8) - 1][1]
        truth.append((a, b))
        if min(len(offs), t + 8) == len(offs):
            break
    assert [(s, e) for _, s, e in triples] == truth
    # the legacy search is wrong on this text when windows overlap or the step is not the period
    guessed = locate_chunk_offsets(PERIODIC, [c for c, _, _ in triples])
    if step % 3:                         # the text has a period of 3 tokens
        assert guessed != truth


def test_fixed_split_text_is_unchanged():
    ch = FixedTokenChunker(model_name=TOK, chunk_size=16, chunk_overlap=4)
    assert ch.split_text(REPEATED_PARAS) == [c for c, _, _ in ch.split_text_with_offsets(REPEATED_PARAS)]


def test_fixed_strip_whitespace_moves_start_past_leading_blanks():
    text = "   \n\n  hello world   foo bar baz   "
    ch = FixedTokenChunker(model_name=TOK, chunk_size=2, chunk_overlap=0, strip_whitespace=True)
    triples = ch.split_text_with_offsets(text)
    _check(text, triples)
    assert triples[0][0] == "hello world"


# ---------------- recursive ----------------
@pytest.mark.parametrize("overlap", [0, 8])
@pytest.mark.parametrize("text", [REPEATED_PARAS, TILDES, PERIODIC], ids=["paras", "tildes", "periodic"])
def test_recursive_offsets_exact_and_cover_the_text(text, overlap):
    ch = RecursiveTokenChunker(model_name=TOK, chunk_size=24, chunk_overlap=overlap,
                               separators=["\n\n", "\n", ".", "?", "!", " ", ""])
    triples = ch.split_text_with_offsets(text)
    assert len(triples) > 3
    _check(text, triples)
    # LangChain chunks cover the whole text, so a chunk put at an earlier duplicate would leave a hole
    assert _uncovered_non_ws(text, triples) == []
    assert [c for c, _, _ in triples] == ch.split_text(text)


def test_recursive_overlap_zero_chunks_tile_the_text():
    ch = RecursiveTokenChunker(model_name=TOK, chunk_size=24, chunk_overlap=0)
    triples = ch.split_text_with_offsets(REPEATED_PARAS)
    for (_, _, e1), (_, s2, _) in zip(triples, triples[1:]):
        assert REPEATED_PARAS[e1:s2].strip() == ""   # nothing but stripped whitespace in between


# ---------------- hierarchical ----------------
def _hier(overlap=0, size=24):
    return HierarchicalSectionChunker(
        model_name=TOK, chunk_size=size, chunk_overlap=overlap,
        use_latex=False, use_numbered_headings=False, use_common_chem_headings=False,
        use_generic_heading_heuristics=False,
        custom_heading_regexes=[r"^\s*section\s+[a-z]\s*$"],
        join_short_sections=False,
    )


def test_hierarchical_offsets_with_identical_sections():
    body = " ".join(f"Sentence number {i % 4} about furfural yields." for i in range(12))
    text = "Section A\n" + body + "\n\nSection B\n" + body + "\n\nSection C\n" + body
    for overlap in (0, 6):
        triples = _hier(overlap).split_text_with_offsets(text)
        _check(text, triples)
        b_start, c_start = text.index("Section B"), text.index("Section C")
        in_b = [t for t in triples if b_start <= t[1] < c_start]
        in_c = [t for t in triples if t[1] >= c_start]
        assert in_b and in_c
        assert min(t[1] for t in in_b) == b_start      # the first chunk of section B starts at its heading
        assert min(t[1] for t in in_c) == c_start
        assert _uncovered_non_ws(text, triples) == []


def test_hierarchical_no_headings_and_split_text_unchanged():
    ch = _hier(overlap=4)
    text = "no headings here " * 30
    triples = ch.split_text_with_offsets(text)
    _check(text, triples)
    assert ch.split_text(text) == [c for c, _, _ in triples]


# ---------------- semantic ----------------
def _sem(use_recursive, overlap, size=16):
    return SemanticSplitter(model_name=SEM, chunk_size=size, chunk_overlap=overlap, use_recursive=use_recursive,
                            breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80.0)


@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("overlap", [0, 4])
def test_semantic_offsets_inside_their_segment_with_identical_segments(use_recursive, overlap):
    sp = _sem(use_recursive, overlap)
    text = PARA + "\n\n" + PARA + "\n\n" + PARA
    third = 2 * (len(PARA) + 2)
    spans = [(0, len(PARA)), (len(PARA) + 2, 2 * len(PARA) + 2), (third, len(text))]
    sp._segment_spans = lambda t: spans          # fixed segments: three identical texts
    triples = sp.split_text_with_offsets(text)
    _check(text, triples)
    for a, b in spans:                           # every segment gets chunks and they stay inside it
        inside = [t for t in triples if a <= t[1] and t[2] <= b]
        assert inside, (a, b)
    assert all(any(a <= s and e <= b for a, b in spans) for _, s, e in triples)
    assert [c for c, _, _ in triples] == sp.split_text(text)


def test_semantic_real_segmentation_is_exact():
    from tests.test_semantic_verbatim import TEXT
    for use_recursive in (False, True):
        for overlap in (0, 6):
            sp = _sem(use_recursive, overlap, size=24)
            triples = sp.split_text_with_offsets(TEXT)
            assert len(triples) > 1
            _check(TEXT, triples)


# ---------------- stage 4 gate ----------------
def test_gate_accepts_good_and_rejects_bad_positions():
    text = "abcdefghij"
    check_chunk_offsets(text, [("abc", 0, 3), ("cde", 2, 5), ("jj"[:1], 9, 10)])
    with pytest.raises(ChunkOffsetError, match="text != full_text"):
        check_chunk_offsets(text, [("abc", 1, 4)])
    with pytest.raises(ChunkOffsetError, match="document order"):
        check_chunk_offsets(text, [("def", 3, 6), ("abc", 0, 3)])
    with pytest.raises(ChunkOffsetError, match="invalid span"):
        check_chunk_offsets(text, [("abc", 8, 11)])
    with pytest.raises(ChunkOffsetError):
        check_chunk_offsets(text, [("", 2, 2)])


class _FakeChunker:
    def __init__(self, triples):
        self.triples = triples

    def split_text(self, text):          # must not be used when offsets are available
        raise AssertionError("split_text called")

    def split_text_with_offsets(self, text):
        return self.triples


def _run(tmp_path, monkeypatch, chunker, text="hello world, hello world"):
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"corpus_id": "d1", "text": text}) + "\n")
    monkeypatch.setattr(chunking.ChunkerFactory, "create", staticmethod(lambda n, p: chunker))
    out = tmp_path / "out" / "chunks.jsonl"
    process_docs(str(inp), str(out), id_field="corpus_id", strategy_name="x", compute_offsets=True)
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_stage4_uses_native_positions_not_search(tmp_path, monkeypatch):
    # "hello world" occurs twice; the chunker says the 2nd chunk is the second copy.
    rows = _run(tmp_path, monkeypatch, _FakeChunker([("hello world", 0, 11), ("hello world", 13, 24)]))
    assert [(r["start_char"], r["end_char"], r["offset_exact"], r["offset_source"]) for r in rows] == [
        (0, 11, True, "native"), (13, 24, True, "native")]
    stats = json.loads((tmp_path / "out" / "chunking_stats.json").read_text())
    assert stats["offset_sources"] == {"native": 2}


def test_stage4_fails_on_wrong_or_unordered_native_positions(tmp_path, monkeypatch):
    with pytest.raises(ChunkOffsetError):
        _run(tmp_path, monkeypatch, _FakeChunker([("hello world", 1, 12)]))
    with pytest.raises(ChunkOffsetError):
        _run(tmp_path, monkeypatch, _FakeChunker([("hello world", 13, 24), ("hello world", 0, 11)]))


def test_stage4_with_a_real_chunker(tmp_path, monkeypatch):
    ch = FixedTokenChunker(model_name=TOK, chunk_size=8, chunk_overlap=3)
    rows = _run(tmp_path, monkeypatch, ch, text=PERIODIC)
    assert rows and all(r["offset_exact"] and r["offset_source"] == "native" for r in rows)
    assert all(PERIODIC[r["start_char"]:r["end_char"]] == r["text"] for r in rows)


def test_legacy_string_chunkers_still_work(tmp_path, monkeypatch):
    class Legacy:
        def split_text(self, text):
            return [text[:5], text[6:]]
    rows = _run(tmp_path, monkeypatch, Legacy(), text="hello world")
    assert [r["offset_source"] for r in rows] == ["search", "search"]


def test_recursive_locate_prefers_the_candidate_that_moves_forward():
    # Paper 41 of the corpus: the one-word chunk "Summary" also occurs inside the previous chunk.
    # A chunk always ends after the previous one, so the first copy cannot be the right one.
    text = "ethics ... cultured meat.\n\n8. Summary and Conclusion\n\nSummary\nWe explore the idea of growing organs."
    ch = RecursiveTokenChunker(model_name=TOK, chunk_size=24, chunk_overlap=8)
    pieces = ["ethics ... cultured meat.\n\n8. Summary and Conclusion", "Summary", "We explore the idea of growing organs."]
    got = ch._locate(text, pieces)
    assert got[1][1] == text.index("Summary\nWe")
    assert [text[s:e] for _, s, e in got] == pieces


def test_recursive_ambiguous_chunks_are_counted_and_logged(caplog):
    ch = RecursiveTokenChunker(model_name=TOK, chunk_size=24, chunk_overlap=8,
                               separators=["\n\n", "\n", ".", "?", "!", " ", ""])
    with caplog.at_level("WARNING"):
        triples = ch.split_text_with_offsets(TILDES)
    assert ch.n_ambiguous > 0
    assert any("occurs" in r.getMessage() for r in caplog.records)
    assert _uncovered_non_ws(TILDES, triples) == []

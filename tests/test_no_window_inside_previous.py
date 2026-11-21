"""Audit finding after B3: with SentencePiece-style offsets (two tokens, same characters) the last token
window of the fixed, semantic-fixed and hierarchical splitters could lie completely inside the previous
chunk (hierarchical, bge-m3, c128 o32: 26 chunks). Such windows are skipped; nothing else changes."""
import os
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")

from pipeline_lib import hierarchical_section_chunker as hsc
from pipeline_lib import semantic_chunker as sem
from pipeline_lib.fixed_token_chunker import FixedTokenChunker, _HFTokenTextSplitter as FixedSplitter


class _SentencePieceLike:
    """Fake fast tokenizer: one token per word, plus one extra token at the very end that carries the
    same character offsets as the last word (like the '▁' / split-character tokens of XLM-R and T5)."""

    def __init__(self, duplicate_last=True):
        self.duplicate_last = duplicate_last

    def __call__(self, text, **kw):
        offs, pos = [], 0
        for w in text.split(" "):
            offs.append((pos, pos + len(w)))
            pos += len(w) + 1
        if self.duplicate_last:
            offs.append(offs[-1])
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}


def _inside_previous(triples):
    return [i for i in range(1, len(triples)) if triples[i][2] <= triples[i - 1][2] and triples[i][1] >= triples[i - 1][1]]


def _text(n_words):
    return " ".join(f"w{i}" for i in range(n_words))


# size 8, overlap 3: windows [0,8) [5,13) ... ; n_tokens = n_words + 1 (the duplicated last token)
@pytest.mark.parametrize("n_words", [8, 9, 12, 13, 17, 21, 24])
def test_fixed_splitter_skips_the_window_inside_the_previous(n_words):
    text = _text(n_words)
    sp = FixedSplitter(chunk_size=8, chunk_overlap=3, model_name="gpt2", add_start_index=False, strip_whitespace=True)
    sp.tokenizer = _SentencePieceLike()
    triples = sp.split_text_with_offsets(text)
    assert _inside_previous(triples) == []
    assert triples[-1][2] == len(text)                       # the text is still covered to its end
    assert all(text[s:e] == c for c, s, e in triples)


@pytest.mark.parametrize("n_words", [8, 9, 12, 13, 17, 21, 24])
def test_semantic_splitter_skips_the_window_inside_the_previous(n_words):
    text = _text(n_words)
    sp = sem._HFTokenTextSplitter(tokenizer=_SentencePieceLike(), chunk_size=8, chunk_overlap=3)
    triples = sp.split_text_with_offsets(text)
    assert _inside_previous(triples) == []
    assert triples[-1][2] == len(text)


@pytest.mark.parametrize("n_words", [8, 9, 12, 13, 17, 21, 24])
def test_hierarchical_splitter_skips_the_window_inside_the_previous(n_words):
    text = _text(n_words)
    sp = hsc._HFTokenTextSplitter(chunk_size=8, chunk_overlap=3, tokenizer=_SentencePieceLike())
    triples = sp.split_text_with_offsets(text)
    assert _inside_previous(triples) == []
    assert triples[-1][2] == len(text)


def test_without_duplicate_offsets_nothing_changes():
    # a normal tokenizer (every token has its own characters): the windows are exactly the ones of the
    # token arithmetic, i.e. starts at tokens 0, 5, 10, 15 (end_tok - overlap) and the last one reaches the end
    text = _text(21)
    words = text.split(" ")
    sp = hsc._HFTokenTextSplitter(chunk_size=8, chunk_overlap=3, tokenizer=_SentencePieceLike(duplicate_last=False))
    triples = sp.split_text_with_offsets(text)
    assert [s for _, s, _ in triples] == [text.index(words[i]) for i in (0, 5, 10, 15)]
    assert triples[-1][2] == len(text)
    assert _inside_previous(triples) == []


def test_real_xlmr_tokenizer_doc_103():
    p = Path(__file__).resolve().parents[1] / "temp_files/artifacts_old/chemquests_hf_db/full_text.jsonl"
    if not p.exists():
        pytest.skip("full_text.jsonl not available")
    import json
    docs = {str(json.loads(l)["corpus_id"]): json.loads(l)["text"] for l in p.open()}
    try:
        ch = hsc.HierarchicalSectionChunker(
            model_name="BAAI/bge-m3", chunk_size=128, chunk_overlap=32, use_latex=False,
            use_numbered_headings=True, use_common_chem_headings=True, use_generic_heading_heuristics=True,
            join_short_sections=True, min_section_tokens=128, prefer_merge_forward=True,
            drop_tail_sections_matching=["references"], drop_from_first_match=True)
    except Exception:
        pytest.skip("bge-m3 tokenizer not available")
    for d in ("103", "106"):                                  # the papers where the audit found them
        triples = ch.split_text_with_offsets(docs[d])
        assert _inside_previous(triples) == [], d
        assert all(docs[d][s:e] == c for c, s, e in triples)

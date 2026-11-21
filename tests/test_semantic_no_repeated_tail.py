"""B3: semantic-fixed must not emit an extra tail chunk that lies inside the previous chunk."""
import os

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")

from transformers import AutoTokenizer

from pipeline_lib.fixed_token_chunker import FixedTokenChunker
from pipeline_lib.semantic_chunker import SemanticSplitter, _HFTokenTextSplitter

MODEL = os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _inside_previous(triples):
    return [i for i in range(1, len(triples))
            if triples[i][1] >= triples[i - 1][1] and triples[i][2] <= triples[i - 1][2]]


@pytest.mark.parametrize("n_words", list(range(8, 60, 3)))
@pytest.mark.parametrize("size,overlap", [(10, 4), (12, 3), (8, 7), (10, 0)])
def test_window_loop_stops_at_the_end_of_the_segment(n_words, size, overlap):
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    text = " ".join(f"w{i}" for i in range(n_words))
    sp = _HFTokenTextSplitter(tokenizer=tok, chunk_size=size, chunk_overlap=overlap)
    triples = sp.split_text_with_offsets(text)
    assert _inside_previous(triples) == []
    # same windows as the fixed-token chunker (which stops at the end of the text)
    fixed = FixedTokenChunker(model_name=MODEL, chunk_size=size, chunk_overlap=overlap)
    assert triples == fixed.split_text_with_offsets(text)
    # the last chunk reaches the end of the text
    assert triples[-1][2] == len(text)


def test_exact_case_that_used_to_repeat_the_tail():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    # 14 one-token words, window 10, overlap 4 (step 6): windows start at 0 and 6; the old loop also
    # started a third window at token 12 whose two tokens lie inside the second window.
    text = " ".join(["a"] * 14)
    assert len(tok(text, add_special_tokens=False)["input_ids"]) == 14
    sp = _HFTokenTextSplitter(tokenizer=tok, chunk_size=10, chunk_overlap=4)
    assert len(sp.split_text_with_offsets(text)) == 2


@pytest.mark.parametrize("overlap", [4, 8])
def test_semantic_fixed_end_to_end_no_chunk_inside_the_previous(overlap):
    from tests.test_semantic_verbatim import TEXT
    sp = SemanticSplitter(model_name=MODEL, chunk_size=24, chunk_overlap=overlap, use_recursive=False,
                          breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80.0)
    triples = sp.split_text_with_offsets(TEXT)
    assert len(triples) > 3
    assert _inside_previous(triples) == []
    # at most chunk_size tokens per chunk, as before
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    assert all(len(tok(c, add_special_tokens=False)["input_ids"]) <= 24 for c, _, _ in triples)

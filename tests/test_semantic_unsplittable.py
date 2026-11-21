"""B5: semantic-recursive splits text without separators instead of emitting one oversized chunk."""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")

from transformers import AutoTokenizer

from pipeline_lib.semantic_chunker import SemanticSplitter

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIZE, SLACK = 32, 5          # "a few tokens"


def _sp(size=SIZE, overlap=4):
    return SemanticSplitter(model_name=MODEL, chunk_size=size, chunk_overlap=overlap, use_recursive=True,
                            breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80.0)


def _tokens(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def test_separator_list_ends_with_empty_string():
    sp = _sp()
    assert sp._token_splitter.config["separators"][-1] == ""
    assert sp._token_splitter.config["separators"][:6] == ["\n\n", "\n", ".", "?", "!", " "]


@pytest.mark.parametrize("overlap", [0, 4])
def test_run_without_separators_is_split(overlap):
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    text = "Intro sentence here. " + "~/" * 700 + " Closing words follow."
    sp = _sp(overlap=overlap)
    chunks = sp._token_splitter.split_text_with_offsets(text)
    assert len(chunks) > 5
    assert max(_tokens(tok, c) for c, _, _ in chunks) <= SIZE + SLACK
    assert all(text[s:e] == c for c, s, e in chunks)


def test_without_the_empty_separator_the_run_stays_in_one_chunk():
    # documents the old behaviour (what the fix removes)
    from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    text = "~/" * 700
    old = RecursiveTokenChunker(model_name=MODEL, chunk_size=SIZE, chunk_overlap=4,
                                separators=["\n\n", "\n", ".", "?", "!", " "])
    assert max(_tokens(tok, c) for c in old.split_text(text)) > 10 * SIZE


def _paper(doc_id):
    p = ROOT / "temp_files/artifacts_old/chemquests_hf_db/full_text.jsonl"
    if not p.exists():
        pytest.skip("full_text.jsonl not available")
    for line in p.open():
        d = json.loads(line)
        if str(d["corpus_id"]) == str(doc_id):
            return d["text"]
    pytest.skip(f"paper {doc_id} not found")


def test_paper_91_has_no_oversized_chunk():
    text = _paper(91)
    assert "~/~/~/" in text
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    for overlap in (0, 4):
        sp = _sp(size=128, overlap=overlap)
        # recursive splitter over the whole paper (worst case: one segment) ...
        chunks = sp._token_splitter.split_text_with_offsets(text)
        assert max(_tokens(tok, c) for c, _, _ in chunks) <= 128 + SLACK
    # ... and the real semantic segmentation
    sp = _sp(size=128, overlap=0)
    triples = sp.split_text_with_offsets(text)
    assert max(_tokens(tok, c) for c, _, _ in triples) <= 128 + SLACK
    assert all(text[s:e] == c for c, s, e in triples)

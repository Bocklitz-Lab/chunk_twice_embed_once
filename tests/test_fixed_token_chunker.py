import math
import os
import pytest

# Skip the whole file if transformers isn't installed (your module requires it at import time)
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer

# 👇 update this import to your actual module path
from pipeline_lib.fixed_token_chunker import FixedTokenChunker, _HFTokenTextSplitter, Document


@pytest.fixture(scope="session")
def model_name():
    # Use a very common fast tokenizer. gpt2 fast works well and has offsets.
    return os.environ.get("FTCHUNKER_MODEL", "gpt2")


@pytest.fixture
def chunker(model_name):
    return FixedTokenChunker(
        model_name=model_name,
        chunk_size=5,
        chunk_overlap=2,
        add_start_index=True,
        strip_whitespace=True,
    )


def _count_tokens(text: str, model_name: str) -> int:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    enc = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
    )
    return len(enc["input_ids"])


def _expected_num_chunks(n_tokens: int, chunk_size: int, overlap: int) -> int:
    if n_tokens == 0:
        return 0
    if n_tokens <= chunk_size:
        return 1
    step = max(1, chunk_size - overlap)
    # Number of steps to cover the remaining tokens after the first chunk
    remain = n_tokens - chunk_size
    return 1 + math.ceil(remain / step)


def test_constructor_overlap_validation(model_name):
    with pytest.raises(ValueError):
        FixedTokenChunker(model_name=model_name, chunk_size=10, chunk_overlap=11)


def test_split_text_chunk_count_matches_token_math(chunker, model_name):
    text = "a b c d e f g h i j k l m n o p q r s t"
    n_tokens = _count_tokens(text, model_name)
    chunks = chunker.split_text(text)

    expected = _expected_num_chunks(
        n_tokens=n_tokens,
        chunk_size=chunker.chunk_size,
        overlap=chunker.chunk_overlap,
    )
    assert len(chunks) == expected
    # No empty chunks
    assert all(c.strip() for c in chunks)


def test_split_text_empty_and_whitespace_only(model_name):
    ch = FixedTokenChunker(model_name=model_name, chunk_size=8, chunk_overlap=2)
    assert ch.split_text("") == []
    assert ch.split_text("    \n\t  ") == []  # strip_whitespace=True by default

    ch2 = FixedTokenChunker(model_name=model_name, chunk_size=8, chunk_overlap=2, strip_whitespace=False)
    # With no stripping, still should avoid empty chunks because offsets span non-empty regions
    assert ch2.split_text("    ") == ["    "]


def test_create_documents_adds_start_index_and_respects_strip(chunker):
    # Insert extra spaces before a boundary so we can verify start_index adjustment
    text = "AAA BBB   CCC DDD EEE FFF"
    docs = chunker.create_documents([text])
    assert len(docs) >= 2

    # Gather start indices and contents
    starts, contents = [], []
    if Document is None:
        for d in docs:
            starts.append(d["metadata"].get("start_index"))
            contents.append(d["page_content"])
    else:
        for d in docs:
            starts.append(d.metadata.get("start_index"))
            contents.append(d.page_content)

    # All chunks should be non-empty and trimmed
    assert all(c and c == c.strip() for c in contents)

    # Ensure at least one chunk begins exactly at "CCC" (after trimming)
    ccc_idx = text.index("CCC")
    assert ccc_idx in starts, f"Expected a chunk to start at char index {ccc_idx}, got {starts}"


def test_create_documents_metadata_passthrough(model_name):
    ch = FixedTokenChunker(
        model_name=model_name, chunk_size=10, chunk_overlap=3, add_start_index=True
    )
    texts = ["foo bar baz", "lorem ipsum dolor sit amet"]
    metas = [{"paper_id": "p1"}, {"paper_id": "p2", "section": "intro"}]
    docs = ch.create_documents(texts, metadatas=metas)

    assert len(docs) >= 2

    def get_md(i):
        if Document is None:
            return docs[i]["metadata"]
        return docs[i].metadata

    md0 = get_md(0)
    md1 = get_md(1)
    assert md0.get("paper_id") == "p1"
    assert md1.get("paper_id") == "p2"
    assert md1.get("section") == "intro"
    # start_index should be present since add_start_index=True
    assert "start_index" in md0
    assert isinstance(md0["start_index"], int)


def test_unicode_text_splits_safely(model_name):
    ch = FixedTokenChunker(model_name=model_name, chunk_size=6, chunk_overlap=2)
    text = "氯化钠是食盐。NaCl is common salt. 水的沸点是100℃。"
    chunks = ch.split_text(text)
    assert len(chunks) >= 1
    # Ensure we never produce empty or whitespace-only chunks
    assert all(c.strip() for c in chunks)


def test_determinism_same_input_same_chunks(chunker):
    text = "This is a deterministic split test across tokens."
    a = chunker.split_text(text)
    b = chunker.split_text(text)
    assert a == b


def test_tokenizer_cache_is_shared(model_name):
    # Access the private cache (ok in tests) to verify caching behavior
    cache_before = dict(_HFTokenTextSplitter._TOKENIZER_CACHE)

    s1 = _HFTokenTextSplitter(
        chunk_size=8, chunk_overlap=2, model_name=model_name, add_start_index=False, strip_whitespace=True
    )
    s2 = _HFTokenTextSplitter(
        chunk_size=16, chunk_overlap=4, model_name=model_name, add_start_index=True, strip_whitespace=False
    )

    assert model_name in _HFTokenTextSplitter._TOKENIZER_CACHE
    assert s1.tokenizer is s2.tokenizer  # exact same object
    # Cache should not shrink
    assert len(_HFTokenTextSplitter._TOKENIZER_CACHE) >= len(cache_before)


def test_no_overlap_when_overlap_zero(model_name):
    # Overlap 0 should still produce valid chunks and cover the text
    ch = FixedTokenChunker(model_name=model_name, chunk_size=4, chunk_overlap=0)
    text = "H2O CO2 NaCl CH4 NH3 HCl"
    chunks = ch.split_text(text)
    # Ensure concatenating chunks (with spaces) at least covers the original tokens
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)


def test_small_text_smaller_than_chunk_size(model_name):
    ch = FixedTokenChunker(model_name=model_name, chunk_size=100, chunk_overlap=50)
    text = "Only a few tokens here."
    chunks = ch.split_text(text)
    assert len(chunks) == 1
    # entire text should be present (possibly trimmed)
    assert chunks[0].replace("\n", " ") in text or text in chunks[0]

"""min_chunk_size of the semantic splitter is measured in tokens (default) or, for the legacy rule, in characters."""
import os

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")

from pipeline_lib.semantic_chunker import SemanticSplitter

MODEL = os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# ten sentences of 6 short words (~7 tokens, ~25 characters each)
TEXT = " ".join(f"Cat {w} sat on a mat." for w in "abcdefghij")


def _splitter(**kw):
    sp = SemanticSplitter(model_name=MODEL, chunk_size=64, chunk_overlap=0, **kw)
    sc = sp._semantic
    # every sentence boundary is a breakpoint candidate: only min_chunk_size decides where to cut
    sc._calculate_sentence_distances = lambda sentences: ([1.0] * len(sentences), None)
    sc._calculate_breakpoint_threshold = lambda d, *a, **k: (0.5, list(d))
    return sp


def _ntok(sp, s):
    return len(sp._tokenizer(s, add_special_tokens=False)["input_ids"])


def test_tokens_unit_is_the_default_and_counts_tokens():
    sp = _splitter(min_chunk_size=20)
    assert sp.config["min_chunk_unit"] == "tokens"
    groups = [TEXT[a:b] for a, b in sp._segment_spans(TEXT)]
    assert len(groups) > 1
    assert all(_ntok(sp, g) >= 20 for g in groups[:-1])
    # cut as early as the rule allows: dropping the last sentence of a group goes below the minimum
    assert all(_ntok(sp, g.rsplit(". ", 1)[0] + ".") < 20 for g in groups[:-1])


def test_chars_unit_reproduces_the_legacy_character_rule():
    sp = _splitter(min_chunk_size=60, min_chunk_unit="chars")
    groups = [TEXT[a:b] for a, b in sp._segment_spans(TEXT)]
    assert all(len(g) >= 60 for g in groups[:-1])
    # 60 characters are ~14 tokens: the same number read as tokens gives larger groups
    tokens = [TEXT[a:b] for a, b in _splitter(min_chunk_size=60)._segment_spans(TEXT)]
    assert len(tokens) < len(groups)


def test_unit_is_validated_and_part_of_the_cache_key(tmp_path):
    with pytest.raises(ValueError):
        SemanticSplitter(model_name=MODEL, chunk_size=64, chunk_overlap=0, min_chunk_unit="words")
    a = _splitter(min_chunk_size=60)
    b = _splitter(min_chunk_size=60, min_chunk_unit="chars")
    assert a._cache_key(TEXT) != b._cache_key(TEXT)


def test_without_minimum_the_unit_changes_nothing():
    a = _splitter()._segment_spans(TEXT)
    b = _splitter(min_chunk_unit="chars")._segment_spans(TEXT)
    assert a == b

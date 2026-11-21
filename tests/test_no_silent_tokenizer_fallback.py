"""B6: no chunker silently falls back to another tokenizer or to counting words."""
import inspect

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")
pytest.importorskip("tqdm")

from pipeline_lib import hierarchical_section_chunker as hsc
from pipeline_lib.fixed_token_chunker import FixedTokenChunker
from pipeline_lib.hierarchical_section_chunker import HierarchicalSectionChunker
from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker
from pipeline_lib.semantic_chunker import SemanticSplitter
from stages.chunking import ChunkerFactory


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_model_name_raises(missing):
    with pytest.raises(ValueError, match="model_name"):
        FixedTokenChunker(model_name=missing, chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError, match="model_name"):
        RecursiveTokenChunker(model_name=missing, chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError, match="model_name"):
        HierarchicalSectionChunker(model_name=missing, chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError, match="model_name"):
        SemanticSplitter(model_name=missing, chunk_size=10, chunk_overlap=2)


def test_model_name_not_given_at_all_raises():
    with pytest.raises(ValueError, match="model_name"):
        FixedTokenChunker(chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError, match="model_name"):
        RecursiveTokenChunker(chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError, match="model_name"):
        HierarchicalSectionChunker(chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError, match="model_name"):
        SemanticSplitter(chunk_size=10, chunk_overlap=2)


def test_no_default_model_in_signatures():
    for cls in (FixedTokenChunker, RecursiveTokenChunker, HierarchicalSectionChunker, SemanticSplitter):
        default = inspect.signature(cls.__init__).parameters["model_name"].default \
            if "model_name" in inspect.signature(cls.__init__).parameters else None
        assert default in (None, inspect.Parameter.empty), cls


def test_factory_has_no_default_semantic_model():
    with pytest.raises(ValueError, match="model_name"):
        ChunkerFactory.create("semantic", {"chunk_size": 64, "chunk_overlap": 0})
    for name in ("fixed_token", "recursive_token", "hierarchical_section"):
        with pytest.raises(ValueError, match="model_name"):
            ChunkerFactory.create(name, {"chunk_size": 64, "chunk_overlap": 0})


def test_hierarchical_raises_when_the_tokenizer_cannot_be_loaded(monkeypatch):
    def boom(*a, **k):
        raise OSError("no such model")
    monkeypatch.setattr(hsc.AutoTokenizer, "from_pretrained", boom)
    with pytest.raises(RuntimeError, match="Could not load the tokenizer"):
        HierarchicalSectionChunker(model_name="nobody/nothing", chunk_size=10, chunk_overlap=2)


def test_fixed_and_recursive_raise_for_an_unknown_model():
    with pytest.raises(Exception):
        FixedTokenChunker(model_name="nobody-at-all/no-such-tokenizer-xyz", chunk_size=10, chunk_overlap=2)
    with pytest.raises(ValueError, match="Failed to load"):
        RecursiveTokenChunker(model_name="nobody-at-all/no-such-tokenizer-xyz", chunk_size=10, chunk_overlap=2)


def test_hierarchical_counts_model_tokens_not_words():
    ch = HierarchicalSectionChunker(model_name="bert-base-uncased", chunk_size=16, chunk_overlap=0)
    word = "electroencephalographically"
    assert ch._count_tokens(word) > 1          # word pieces, not one "word"
    text = " ".join([word] * 12)
    sizes = [len(ch._hf_tokenizer.encode(c, add_special_tokens=False)) for c in ch.split_text(text)]
    # a few tokens above nominal can appear when a cut falls inside a word and is re-encoded;
    # a word-count fallback would give 16 WORDS = ~100+ tokens per chunk
    assert max(sizes) <= 16 + 2


def test_hierarchical_splitter_without_tokenizer_raises():
    with pytest.raises(ValueError, match="tokenizer"):
        hsc._HFTokenTextSplitter(chunk_size=10, chunk_overlap=0, tokenizer=None)

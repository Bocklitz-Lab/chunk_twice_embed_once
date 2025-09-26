import os
import pytest

# Skip if required libs aren't installed
transformers = pytest.importorskip("transformers")
try:
    # Try new splitters package first (LC v0.2+)
    pytest.importorskip("langchain_text_splitters")
except pytest.skip.Exception:
    # Fall back to older LC (v0.0–v0.1.*)
    pytest.importorskip("langchain")

from transformers import AutoTokenizer

# 👇 Update this to your actual module path if needed
from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker, Language


@pytest.fixture(scope="session")
def model_name():
    # You can override in CI/local with: export RTCHUNKER_MODEL=gpt2
    return os.environ.get("RTCHUNKER_MODEL", "bert-base-uncased")


@pytest.fixture(scope="session")
def hf_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def token_len(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def test_constructor_validation(model_name):
    # overlap > size
    with pytest.raises(ValueError):
        RecursiveTokenChunker(chunk_size=10, chunk_overlap=11, model_name=model_name)

    # size <= 0
    with pytest.raises(ValueError):
        RecursiveTokenChunker(chunk_size=0, chunk_overlap=0, model_name=model_name)

    # overlap < 0
    with pytest.raises(ValueError):
        RecursiveTokenChunker(chunk_size=10, chunk_overlap=-1, model_name=model_name)


def test_removed_and_unknown_kwargs(model_name):
    # Removed kwargs should error
    for bad in ("encoding_name", "allowed_special", "disallowed_special"):
        with pytest.raises(ValueError):
            RecursiveTokenChunker(chunk_size=10, chunk_overlap=0, **{bad: "x"})

    # Unknown kwargs should error
    with pytest.raises(ValueError) as e:
        RecursiveTokenChunker(chunk_size=10, chunk_overlap=0, foo="bar")
    assert "Unknown argument(s): foo" in str(e.value)


def test_split_text_respects_token_budget(model_name, hf_tokenizer):
    ch = RecursiveTokenChunker(
        model_name=model_name,
        chunk_size=10,
        chunk_overlap=3,
        keep_separator=True,  # default behavior shouldn't matter for the budget
    )

    text = (
        "Water boils at 100 C at 1 atm. Sodium chloride is common salt. "
        "CO2 is a greenhouse gas. Ethanol boils at 78 C."
    )
    chunks = ch.split_text(text)
    assert len(chunks) >= 1

    # Each chunk must not exceed chunk_size in *token* length
    for c in chunks:
        assert token_len(hf_tokenizer, c) <= ch.chunk_size, f"chunk token len exceeded: {c!r}"


def test_split_text_empty_and_whitespace(model_name):
    ch = RecursiveTokenChunker(model_name=model_name, chunk_size=8, chunk_overlap=2)
    assert ch.split_text("") == []

    # LangChain splitter with strip_whitespace=True should drop whitespace-only
    assert ch.split_text("   \n\t   ") == []


def test_unicode_and_mixed_language(model_name, hf_tokenizer):
    ch = RecursiveTokenChunker(model_name=model_name, chunk_size=12, chunk_overlap=4)
    text = "氯化钠是食盐。NaCl is common salt. 水的沸点是100℃。H₂O boils at 100°C."
    chunks = ch.split_text(text)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)
    # sanity: token lens within budget
    for c in chunks:
        assert token_len(hf_tokenizer, c) <= ch.chunk_size


def test_create_documents_adds_start_index_and_passes_metadata(model_name):
    ch = RecursiveTokenChunker(
        model_name=model_name,
        chunk_size=10,
        chunk_overlap=2,
        add_start_index=True,
        keep_separator=True,
    )
    texts = [
        "AAA BBB   CCC DDD. EEE FFF.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    ]
    metas = [{"paper_id": "p1"}, {"paper_id": "p2", "section": "intro"}]

    docs = ch.create_documents(texts, metadatas=metas)
    assert len(docs) >= 2

    # LangChain Document objects expose .page_content and .metadata
    for d in docs:
        assert hasattr(d, "page_content")
        assert hasattr(d, "metadata")
        assert d.page_content.strip()
        # start_index should be populated since add_start_index=True
        assert "start_index" in d.metadata
        assert isinstance(d.metadata["start_index"], int)

    # First chunk should preserve passed metadata
    md = docs[0].metadata
    assert md.get("paper_id") == "p1"


def test_determinism_same_input_same_output(model_name):
    ch = RecursiveTokenChunker(model_name=model_name, chunk_size=9, chunk_overlap=3)
    text = "This is a deterministic split across tokens and separators."
    a = ch.split_text(text)
    b = ch.split_text(text)
    assert a == b


def test_config_exposes_params(model_name):
    ch = RecursiveTokenChunker(
        model_name=model_name,
        chunk_size=15,
        chunk_overlap=5,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=False,
        is_separator_regex=False,
        add_start_index=False,
        strip_whitespace=True,
    )
    cfg = ch.config
    assert cfg["chunk_size"] == 15
    assert cfg["chunk_overlap"] == 5
    assert cfg["keep_separator"] is False
    assert cfg["model_name"] == model_name
    assert isinstance(cfg["separators"], list)
    # property passthroughs
    assert ch.chunk_size == 15
    assert ch.chunk_overlap == 5


def test_language_separators_enum_roundtrip():
    # Pick the first available enum member (robust to different enum contents/names)
    lang = next(iter(Language))
    seps = RecursiveTokenChunker.get_separators_for_language(lang)
    assert isinstance(seps, list) and len(seps) >= 1


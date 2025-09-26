import os
import pytest

# --- Required deps (skip if missing) ---
pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")

from transformers import AutoTokenizer

# 👇 update these imports to your actual module paths if needed
from pipeline_lib.semantic_chunker import SemanticSplitter

# Try to see if your RecursiveTokenChunker is importable for the recursive path
_has_recursive = False
try:
    from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker  # noqa: F401
    _has_recursive = True
except Exception:
    try:
        from recursive_token_chunker import RecursiveTokenChunker  # noqa: F401
        _has_recursive = True
    except Exception:
        _has_recursive = False


@pytest.fixture(scope="session")
def model_name():
    # You can override this to something small/cached:
    # export SEMANTIC_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    # or "sentence-transformers/paraphrase-MiniLM-L3-v2"
    return os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def hf_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def token_len(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def test_constructor_validation(model_name):
    with pytest.raises(ValueError):
        SemanticSplitter(model_name=model_name, chunk_size=0)
    with pytest.raises(ValueError):
        SemanticSplitter(model_name=model_name, chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        SemanticSplitter(model_name=model_name, chunk_size=10, chunk_overlap=11)


def test_split_text_respects_token_budget(model_name, hf_tokenizer):
    sp = SemanticSplitter(
        model_name=model_name,
        chunk_size=16,
        chunk_overlap=4,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
        buffer_size=1,
        min_chunk_size=None,
        strip_whitespace=True,
    )
    text = (
        "Water boils at 100 °C at 1 atm. Sodium chloride is common salt. "
        "CO₂ is a greenhouse gas. Ethanol boils at 78 °C."
    )
    chunks = sp.split_text(text)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)
    for c in chunks:
        assert token_len(hf_tokenizer, c) <= sp.chunk_size


def test_unicode_and_whitespace_only(model_name):
    sp = SemanticSplitter(model_name=model_name, chunk_size=12, chunk_overlap=3)
    assert sp.split_text("") == []
    assert sp.split_text("   \n\t   ") == []

    text = "氯化钠是食盐。NaCl is common salt. 水的沸点是100℃。H₂O boils at 100°C."
    chunks = sp.split_text(text)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_create_documents_metadata_and_start_index(model_name):
    sp = SemanticSplitter(
        model_name=model_name,
        chunk_size=16,
        chunk_overlap=4,
        add_start_index=True,
    )
    texts = [
        "AAA BBB   CCC DDD. EEE FFF.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    ]
    metas = [{"paper_id": "p1"}, {"paper_id": "p2", "section": "intro"}]

    docs = sp.create_documents(texts, metadatas=metas)
    assert len(docs) >= 2

    # LangChain Document: page_content + metadata
    for d in docs:
        assert hasattr(d, "page_content")
        assert hasattr(d, "metadata")
        assert d.page_content.strip()
        # start_index may be approximate; just ensure present and int
        assert "start_index" in d.metadata
        assert isinstance(d.metadata["start_index"], int)

    # Metadata passthrough check
    md0 = docs[0].metadata
    assert md0.get("paper_id") == "p1"


def test_create_documents_metadata_length_validation(model_name):
    sp = SemanticSplitter(model_name=model_name, chunk_size=16, chunk_overlap=4)
    texts = ["A", "B", "C"]
    # len(metadatas) neither 1 nor len(texts) => error
    with pytest.raises(ValueError):
        sp.create_documents(texts, metadatas=[{"a": 1}, {"b": 2}])


def test_determinism_same_input_same_output(model_name):
    sp = SemanticSplitter(model_name=model_name, chunk_size=14, chunk_overlap=5)
    text = "This is a deterministic semantic split across sentences and tokens."
    a = sp.split_text(text)
    b = sp.split_text(text)
    # exact equality can be sensitive to model versions; usually stable enough
    assert a == b


def test_config_exposes_params(model_name):
    sp = SemanticSplitter(
        model_name=model_name,
        chunk_size=21,
        chunk_overlap=7,
        use_recursive=False,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=97.0,
        buffer_size=2,
        min_chunk_size=5,
        add_start_index=False,
        strip_whitespace=True,
    )
    cfg = sp.config
    assert cfg["model_name"] == model_name
    assert cfg["chunk_size"] == 21
    assert cfg["chunk_overlap"] == 7
    assert cfg["use_recursive"] is False
    assert cfg["buffer_size"] == 2
    assert cfg["min_chunk_size"] == 5
    assert sp.chunk_size == 21
    assert sp.chunk_overlap == 7


def test_use_recursive_branch(model_name, hf_tokenizer):
    if _has_recursive:
        sp = SemanticSplitter(
            model_name=model_name,
            chunk_size=16,
            chunk_overlap=4,
            use_recursive=True,
        )
        text = "Water boils at 100 °C. Sodium chloride is common salt. CO2 is a gas."
        chunks = sp.split_text(text)
        assert chunks
        for c in chunks:
            assert len(hf_tokenizer.encode(c, add_special_tokens=False)) <= sp.chunk_size
    else:
        with pytest.raises(ImportError):
            SemanticSplitter(
                model_name=model_name,
                chunk_size=16,
                chunk_overlap=4,
                use_recursive=True,
            )

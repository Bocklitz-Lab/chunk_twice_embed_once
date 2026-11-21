import os
import re

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")
pytest.importorskip("tqdm")

from pipeline_lib.semantic_chunker import SemanticSplitter
from stages.chunking import locate_chunk_offsets

TEXT = (
    "Furfural is a platform chemical produced from lignocellulosic biomass. It is used for biofuels, "
    "polymers, and medicines.\n\n"
    "Sugarcane bagasse is an abundant agricultural waste. It has a high cellulose and hemicellulose content. "
    "The samples were collected from sugar mills in Rahim Yar Khan.\n\n"
    "Furfural yield was determined colorimetrically. The product was verified by UV, FTIR, and NMR spectroscopy. "
    "Yields ranged from 7.5% to 10% for the bagasse samples.\n\n"
    "Costs of raw materials, labor, and equipment were calculated. The process is cost-effective for rural "
    "economies. Economic feasibility was also assessed.\n\n"
    "Density functional theory (DFT) calculations used the B3LYP functional. The π-stacking interaction was "
    "stronger than the hydrogen bond. Binding energies were corrected for basis-set superposition error."
)


@pytest.fixture(scope="module")
def model_name():
    return os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("overlap", [0, 6])
def test_chunks_are_exact_substrings_with_exact_offsets(model_name, use_recursive, overlap):
    sp = SemanticSplitter(
        model_name=model_name,
        chunk_size=24,
        chunk_overlap=overlap,
        use_recursive=use_recursive,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80.0,
    )
    chunks = sp.split_text(TEXT)
    assert len(chunks) > 1
    for c in chunks:
        assert c in TEXT, f"not a substring of the source: {c!r}"
    for c, (s, e) in zip(chunks, locate_chunk_offsets(TEXT, chunks)):
        assert s is not None and TEXT[s:e] == c


def test_segment_grouping_matches_langchain(model_name):
    # Re-joining the sentences of each verbatim segment with " " must reproduce LangChain's own
    # segments exactly: same breakpoints, only the original whitespace is kept.
    sp = SemanticSplitter(
        model_name=model_name,
        chunk_size=4000,
        chunk_overlap=0,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80.0,
    )
    rx = sp._semantic.sentence_split_regex
    mine = [" ".join(re.split(rx, seg)) for seg in sp._semantic_segments(TEXT)]
    assert mine == sp._semantic.split_text(TEXT)
    assert len(mine) > 1

"""Semantic segmentation cache: performance only, must never change an output."""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_community")
pytest.importorskip("sentence_transformers")

from pipeline_lib.semantic_chunker import SemanticSplitter
from tests.test_semantic_verbatim import TEXT

MODEL = os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OTHER_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


def _sp(cache_dir=None, use_recursive=False, size=24, overlap=0, model=MODEL, **kw):
    params = dict(breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80.0)
    params.update(kw)
    return SemanticSplitter(model_name=model, chunk_size=size, chunk_overlap=overlap,
                            use_recursive=use_recursive, cache_dir=cache_dir, **params)


def _forbid_embedding(sp):
    def boom(*a, **k):
        raise AssertionError("embeddings were computed although the segmentation is cached")
    sp._semantic._calculate_sentence_distances = boom


@pytest.mark.parametrize("use_recursive", [False, True])
@pytest.mark.parametrize("overlap", [0, 6])
def test_cache_does_not_change_any_output(tmp_path, use_recursive, overlap):
    plain = _sp(None, use_recursive, overlap=overlap).split_text_with_offsets(TEXT)
    cache = str(tmp_path / "cache")
    cold = _sp(cache, use_recursive, overlap=overlap)
    assert cold.split_text_with_offsets(TEXT) == plain
    assert (cold.cache_hits, cold.cache_misses) == (0, 1)
    warm = _sp(cache, use_recursive, overlap=overlap)
    _forbid_embedding(warm)
    assert warm.split_text_with_offsets(TEXT) == plain
    assert (warm.cache_hits, warm.cache_misses) == (1, 0)


def test_one_entry_serves_every_chunk_size_overlap_and_strategy(tmp_path):
    cache = str(tmp_path / "cache")
    first = _sp(cache, False, size=24, overlap=0)
    first.split_text(TEXT)
    assert len(list(Path(cache).glob("*.json"))) == 1
    for use_recursive in (False, True):
        for size, overlap in [(16, 0), (24, 6), (48, 8)]:
            sp = _sp(cache, use_recursive, size=size, overlap=overlap)
            _forbid_embedding(sp)
            expected = _sp(None, use_recursive, size=size, overlap=overlap).split_text_with_offsets(TEXT)
            assert sp.split_text_with_offsets(TEXT) == expected
            assert sp.cache_hits == 1
    assert len(list(Path(cache).glob("*.json"))) == 1


def test_key_includes_model_text_and_splitter_parameters(tmp_path):
    base = _sp(str(tmp_path))
    k = base._cache_key(TEXT)
    assert k == _sp(str(tmp_path))._cache_key(TEXT)                       # deterministic
    assert k != base._cache_key(TEXT + " More text.")                      # document
    assert k != _sp(str(tmp_path), model=OTHER_MODEL)._cache_key(TEXT)     # model name
    assert k != _sp(str(tmp_path), breakpoint_threshold_amount=95.0)._cache_key(TEXT)
    assert k != _sp(str(tmp_path), breakpoint_threshold_type="standard_deviation",
                    breakpoint_threshold_amount=1.5)._cache_key(TEXT)
    assert k != _sp(str(tmp_path), buffer_size=2)._cache_key(TEXT)
    assert k != _sp(str(tmp_path), min_chunk_size=128)._cache_key(TEXT)
    # not part of the key: things that do not affect the breakpoints
    assert k == _sp(str(tmp_path), size=48, overlap=8)._cache_key(TEXT)
    assert k == _sp(str(tmp_path), use_recursive=True)._cache_key(TEXT)


def test_different_models_do_not_share_entries(tmp_path):
    cache = str(tmp_path / "cache")
    _sp(cache).split_text(TEXT)
    other = _sp(cache, model=OTHER_MODEL)
    other.split_text(TEXT)
    assert other.cache_misses == 1 and other.cache_hits == 0
    assert len(list(Path(cache).glob("*.json"))) == 2


@pytest.mark.parametrize("garbage", ["not json", json.dumps({"spans": [[0, 10 ** 9]]}),
                                     json.dumps({"spans": "x"}), json.dumps({"spans": [[5, 3]]})])
def test_corrupt_or_invalid_entry_is_recomputed(tmp_path, garbage):
    cache = tmp_path / "cache"
    sp = _sp(str(cache))
    expected = _sp(None).split_text_with_offsets(TEXT)
    cache.mkdir()
    (cache / f"{sp._cache_key(TEXT)}.json").write_text(garbage)
    assert sp.split_text_with_offsets(TEXT) == expected
    assert sp.cache_misses == 1
    again = _sp(str(cache))
    _forbid_embedding(again)
    assert again.split_text_with_offsets(TEXT) == expected     # entry was rewritten correctly


def test_cache_disabled_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sp = _sp(None)
    sp.split_text(TEXT)
    assert list(tmp_path.iterdir()) == []
    assert (sp.cache_hits, sp.cache_misses) == (0, 0)


def test_factory_passes_cache_dir(tmp_path):
    from stages.chunking import ChunkerFactory
    sp = ChunkerFactory.create("semantic", {"model_name": MODEL, "chunk_size": 24, "chunk_overlap": 0,
                                            "cache_dir": str(tmp_path / "c")})
    sp.split_text(TEXT)
    assert len(list((tmp_path / "c").glob("*.json"))) == 1

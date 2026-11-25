"""Per-model pooling override (``params.model_pooling``) for models WITHOUT a sentence-transformers ``modules.json``.

sentence-transformers wraps a plain Hugging Face checkpoint in Transformer + Pooling(mean): the LIBRARY DEFAULT, not a statement of the model's authors. Where the
authors define another pooling, the override sets it. Entries: ``{model: {mode: cls | mean | max | lasttoken, source: "<the sentence and where it stands>"}}``.
A model that ships ``modules.json`` already carries its own pooling: an override for it is refused (the card and the files are the truth for it).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

MODES = ("cls", "mean", "max", "lasttoken")


class PoolingConfigError(ValueError):
    """``params.model_pooling`` is malformed or names a model that has its own pooling."""


def resolve(model_name: str, table: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The overriding pooling mode of ``model_name`` or None."""
    entry = (table or {}).get(model_name)
    if entry is None:
        return None
    mode = entry.get("mode") if isinstance(entry, Mapping) else entry
    if mode not in MODES:
        raise PoolingConfigError(f"params.model_pooling[{model_name!r}]: mode must be one of {MODES}, got {mode!r}")
    if isinstance(entry, Mapping) and not entry.get("source"):
        raise PoolingConfigError(f"params.model_pooling[{model_name!r}]: give the source (the sentence of the card or paper that defines the pooling)")
    return mode


def apply(st_model: Any, mode: str, *, has_modules_json: bool, model_name: str = "") -> Any:
    """Replace the Pooling module of a SentenceTransformer by ``mode`` (same word-embedding dimension). Refuses a model with ``modules.json``."""
    if has_modules_json:
        raise PoolingConfigError(f"{model_name}: has a modules.json (its own pooling): params.model_pooling applies only to library-default pooling")
    from sentence_transformers.models import Pooling
    idx = next((i for i, m in enumerate(st_model) if type(m).__name__ == "Pooling"), None)
    if idx is None:
        raise PoolingConfigError(f"{model_name}: no Pooling module to replace")
    dim = st_model[idx].word_embedding_dimension
    st_model._modules[str(idx)] = Pooling(dim, pooling_mode=mode)
    return st_model


def has_modules_json(st_model: Any, model_name: str, revision: Optional[str]) -> bool:
    """Does the snapshot of the loaded model ship ``modules.json``? (cache lookup of ``pipeline_lib.model_info``; unknown counts as yes, so an override is refused)"""
    from pipeline_lib import model_info
    r = model_info.has_modules_json(model_name, revision)
    return True if r is None else bool(r)


def maybe_apply(st_model: Any, model_name: str, revision: Optional[str], table: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Apply the override of ``model_name`` if there is one; returns the mode (None: nothing changed)."""
    mode = resolve(model_name, table)
    if mode is None:
        return None
    apply(st_model, mode, has_modules_json=has_modules_json(st_model, model_name, revision), model_name=model_name)
    return mode

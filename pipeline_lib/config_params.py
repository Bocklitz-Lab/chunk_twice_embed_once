"""Read defaults of the tools from ``configs/main_config.yaml`` (``params`` block), so no tool hard-codes a value the config owns.

The values are the *unresolved* ``params`` entries (no ``${...}`` interpolation); use ``pipeline_lib.split_config.resolve_section`` for
sections that need it. ``load_params`` is cached per file path and modification time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_CONFIG = REPO_ROOT / "configs" / "main_config.yaml"
_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}


def load_params(main_config: Optional[str | Path] = None) -> Dict[str, Any]:
    """``params`` of the main config (default: the repository's ``configs/main_config.yaml``)."""
    path = Path(main_config) if main_config else DEFAULT_MAIN_CONFIG
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key not in _CACHE:
        _CACHE[key] = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("params", {})
    return _CACHE[key]


def reference_tokenizer(main_config: Optional[str | Path] = None) -> Tuple[str, str]:
    """(name, pinned revision) of the tokenizer that counts chunk sizes (``params.chunking_tokenizer`` + ``_revision``).

    Raises ValueError for the legacy setting ``chunking_tokenizer: model`` (no single reference tokenizer exists then)."""
    p = load_params(main_config)
    name, rev = p.get("chunking_tokenizer"), p.get("chunking_tokenizer_revision")
    if not name or name == "model":
        raise ValueError("params.chunking_tokenizer is 'model' (legacy per-model chunking): there is no reference tokenizer")
    if not rev:
        raise ValueError("params.chunking_tokenizer_revision must pin the reference tokenizer")
    return str(name), str(rev)


def section(name: str, main_config: Optional[str | Path] = None) -> Dict[str, Any]:
    """A ``params.<name>`` block (empty dict if absent)."""
    return dict(load_params(main_config).get(name) or {})

"""What a SentenceTransformer model really is: module list, pooling, dimension, maximum length, tokenizer.

Recorded in every run manifest (``model_properties``) because the models differ in these respects and the paper compares them:

* ``pooling_mode``: from the Pooling module ("mean", "cls", "lasttoken", ...). A checkpoint without ``modules.json`` (a plain Hugging Face
  encoder such as SciBERT or ChemBERTa) is wrapped by the library into Transformer + Pooling(mean): ``pooling_is_library_default`` is True then.
* ``embedding_dim`` (the output size of the last module that fixes it), ``max_seq_length`` (what the model truncates at), ``tokenizer_class``.

``describe_model`` needs a loaded model (it runs next to the evaluation). ``describe_snapshot`` (used by ``tools/model_properties.py``) reads the
same facts from the files of a pinned local snapshot, without loading weights and without network.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

POOLING_FLAGS = ("cls_token", "max_tokens", "mean_tokens", "mean_sqrt_len_tokens", "weightedmean_tokens", "lasttoken")
SHORT = {"cls_token": "cls", "max_tokens": "max", "mean_tokens": "mean", "mean_sqrt_len_tokens": "mean_sqrt_len",
         "weightedmean_tokens": "weightedmean", "lasttoken": "lasttoken"}


def pooling_mode_from_flags(flags: Dict[str, Any]) -> Optional[str]:
    """'mean', 'cls', 'cls+mean', ...: the pooling flags that are on (keys with or without the ``pooling_mode_`` prefix)."""
    on = [SHORT[k] for k in POOLING_FLAGS if flags.get(k) or flags.get(f"pooling_mode_{k}")]
    return "+".join(on) if on else None


def snapshot_dir(name: str, revision: Optional[str]) -> Optional[Path]:
    """Local snapshot folder of a pinned Hub model (cache only, never the network); None if it is not cached."""
    if os.path.isdir(name):
        return Path(name)
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(name, revision=revision, local_files_only=True))
    except Exception:
        return None


def has_modules_json(name: str, revision: Optional[str]) -> Optional[bool]:
    """Does the model ship a sentence-transformers ``modules.json``? None if the snapshot is not in the local cache."""
    d = snapshot_dir(name, revision)
    return None if d is None else (d / "modules.json").is_file()


def describe_model(st_model: Any, name: Optional[str] = None, revision: Optional[str] = None) -> Dict[str, Any]:
    """Facts of a loaded SentenceTransformer (see the module docstring). Every field is None when it cannot be read."""
    mods = list(getattr(st_model, "_modules", None) and st_model._modules.values() or [])
    if not mods:
        try:
            mods = list(st_model)
        except Exception:
            mods = []
    types = [type(m).__name__ for m in mods]
    pooling = next((m for m in mods if type(m).__name__ == "Pooling"), None)
    mode = None
    if pooling is not None:
        mode = pooling_mode_from_flags({f"pooling_mode_{k}": getattr(pooling, f"pooling_mode_{k}", False) for k in POOLING_FLAGS})
    tok = getattr(st_model, "tokenizer", None)
    try:
        dim = st_model.get_sentence_embedding_dimension()
    except Exception:
        dim = None
    mj = has_modules_json(name, revision) if name else None
    return {
        "modules": types,
        "pooling_mode": mode,
        "pooling_is_library_default": (None if mj is None else not mj),
        "has_modules_json": mj,
        "embedding_dim": int(dim) if dim else None,
        "max_seq_length": getattr(st_model, "max_seq_length", None),
        "tokenizer_class": type(tok).__name__ if tok is not None else None,
        "tokenizer_name": getattr(tok, "name_or_path", None),
    }


# ---------------------------------------------------------------------------------------------------------------
# from the files of a snapshot (no weights loaded)
# ---------------------------------------------------------------------------------------------------------------
def _json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except Exception:
        return {}


def describe_snapshot(snap: Path) -> Dict[str, Any]:
    """The same facts from ``modules.json`` / ``1_Pooling/config.json`` / ``sentence_bert_config.json`` / ``config.json`` / tokenizer files.

    ``max_seq_length`` follows the library: the value of ``sentence_bert_config.json`` of the Transformer module, else
    ``min(max_position_embeddings, tokenizer model_max_length)``. ``embedding_dim``: the last Dense ``out_features``, else the pooling
    width (word dimension x number of pooling modes), else ``hidden_size``."""
    snap = Path(snap)
    modules: List[Dict[str, Any]] = []
    mj = snap / "modules.json"
    if mj.is_file():
        modules = json.loads(mj.read_text(encoding="utf-8"))
    types = [m.get("type", "").split(".")[-1] for m in modules]
    t_path = next((m.get("path", "") for m in modules if m.get("type", "").endswith("Transformer")), "")
    cfg = _json(snap / t_path / "config.json") or _json(snap / "config.json")
    tok_cfg = _json(snap / t_path / "tokenizer_config.json") or _json(snap / "tokenizer_config.json")
    mode, width, n_modes = None, None, 1
    for m in modules:
        if m.get("type", "").endswith("Pooling"):
            pc = _json(snap / m.get("path", "") / "config.json")
            mode = pooling_mode_from_flags(pc)
            width = pc.get("word_embedding_dimension")
            n_modes = max(1, sum(1 for k in POOLING_FLAGS if pc.get(f"pooling_mode_{k}")))
    dense_out = None
    for m in modules:
        if m.get("type", "").endswith("Dense"):
            dense_out = _json(snap / m.get("path", "") / "config.json").get("out_features", dense_out)
    hidden = cfg.get("hidden_size") or cfg.get("d_model") or cfg.get("n_embd")
    dim = dense_out or (width * n_modes if width else None) or hidden
    msl = _json(snap / t_path / "sentence_bert_config.json").get("max_seq_length") or _json(snap / "sentence_bert_config.json").get("max_seq_length")
    if msl is None:
        pos = cfg.get("max_position_embeddings") or cfg.get("n_positions")
        tml = tok_cfg.get("model_max_length")
        cands = [v for v in (pos, tml) if isinstance(v, (int, float)) and 0 < v < 10 ** 9]
        msl = int(min(cands)) if cands else None
    return {
        "modules": types or ["Transformer", "Pooling"],
        "pooling_mode": mode or ("mean" if not modules else None),
        "pooling_is_library_default": not modules,
        "has_modules_json": bool(modules),
        "embedding_dim": int(dim) if dim else None,
        "max_seq_length": int(msl) if msl else None,
        "tokenizer_class": tok_cfg.get("tokenizer_class") or (cfg.get("tokenizer_class")),
        "architecture": (cfg.get("architectures") or [None])[0],
        "model_type": cfg.get("model_type"),
    }

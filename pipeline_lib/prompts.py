"""Per-model query / document prompts (prefixes or instructions), applied at encoding time only.

The prompts live in the config (``params.model_prompts``, one entry per model; ``params.use_prompts`` switches them
on or off). They are looked up here and applied by ``pipeline_lib.model_wrapper.STNoPrompt.encode`` to the text that
goes into the embedding model. Chunking, token counting and the qrels never see them.

An entry is ``{query: str, passage: str, card: str}``. An empty string means "no prompt for this side". A model
without any prompt has both empty. With ``use_prompts: true`` a model that has no entry is an error: whether a model
needs a prompt must be decided explicitly, never by omission.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

SIDES = ("query", "passage")


class PromptConfigError(ValueError):
    """The prompt table is missing an entry or is malformed."""


def resolve_prompts(
    model_name: str,
    table: Optional[Mapping[str, Any]],
    use_prompts: bool,
    override: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return {"query": str, "passage": str, "card": str} for the model, or None when prompts are off.

    ``override`` (an entry given directly in a model's own config block) wins over the table.
    ``use_prompts: false`` always returns None (the behaviour of all published runs).
    """
    if not use_prompts:
        return None
    entry = override if override is not None else (table or {}).get(model_name)
    if entry is None:
        raise PromptConfigError(
            f"no prompt entry for model {model_name!r} in model_prompts (use_prompts is true). "
            "Add an entry (empty strings for 'no prompt') or set use_prompts: false."
        )
    if not isinstance(entry, Mapping):
        raise PromptConfigError(f"prompt entry of {model_name!r} must be a mapping, got {type(entry).__name__}")
    out: Dict[str, Any] = {}
    for side in SIDES:
        val = entry.get(side, "")
        if val is None:
            val = ""
        if not isinstance(val, str):
            raise PromptConfigError(f"prompt entry of {model_name!r}: {side!r} must be a string")
        out[side] = val
    out["card"] = str(entry.get("card", ""))
    return out


def has_any_prompt(prompts: Optional[Mapping[str, Any]]) -> bool:
    return bool(prompts) and any(prompts.get(s) for s in SIDES)


def side_of(prompt_type: Any) -> Optional[str]:
    """'query' / 'passage' from MTEB's PromptType (an enum with .value) or a plain string; None if unknown."""
    v = getattr(prompt_type, "value", prompt_type)
    return v if v in SIDES else None


def apply_prompt(texts, prompts: Optional[Mapping[str, Any]], prompt_type: Any):
    """Prepend the prompt of ``prompt_type`` to every text. Returns the input unchanged when nothing applies."""
    side = side_of(prompt_type)
    prefix = prompts.get(side) if (prompts and side) else ""
    if not prefix:
        return texts
    if isinstance(texts, str):
        return prefix + texts
    return [prefix + t for t in texts]

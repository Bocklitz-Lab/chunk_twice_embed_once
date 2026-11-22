"""How many texts exceed a model's maximum sequence length when the model embeds them with its OWN tokenizer.

The chunks are cut with the reference tokenizer, but each model tokenizes the text again (plus its prompt and the
document title that MTEB puts in front of the chunk). Texts longer than ``max_seq_length`` are truncated by the model.
This only RECORDS that (it changes nothing): ``truncation_report.json`` next to each run's results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_lib.prompts import apply_prompt


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def embedded_passages(data_dir) -> List[str]:
    """The string MTEB hands to the model for every corpus entry: ``(title + " " + text).strip()``."""
    out = []
    for r in _read_jsonl(Path(data_dir) / "corpus.jsonl"):
        out.append(((r.get("title") + " " + r["text"]) if "title" in r and r.get("title") is not None else r["text"]).strip())
    return out


def embedded_queries(data_dir) -> List[str]:
    return [str(r.get("text") if r.get("text") is not None else r.get("query", "")) for r in _read_jsonl(Path(data_dir) / "queries.jsonl")]


def count(texts: List[str], tokenizer, max_seq_length: Optional[int], batch: int = 2000) -> Dict[str, Any]:
    lens: List[int] = []
    for i in range(0, len(texts), batch):
        lens += [len(x) for x in tokenizer(texts[i:i + batch], add_special_tokens=True, truncation=False)["input_ids"]]
    over = [n for n in lens if max_seq_length and n > max_seq_length]
    return {
        "n_texts": len(lens),
        "n_truncated": len(over),
        "pct_truncated": 100.0 * len(over) / len(lens) if lens else 0.0,
        "mean_tokens": sum(lens) / len(lens) if lens else 0.0,
        "max_tokens": max(lens) if lens else 0,
        "mean_tokens_lost_when_truncated": (sum(n - max_seq_length for n in over) / len(over)) if over else 0.0,
    }


def truncation_report(data_dir, tokenizer, max_seq_length: Optional[int], prompts: Optional[Dict[str, Any]] = None,
                      model_name: str = "") -> Dict[str, Any]:
    """Report for one model on one prepared task folder (prompts included in the counted text)."""
    passages = apply_prompt(embedded_passages(data_dir), prompts, "passage")
    queries = apply_prompt(embedded_queries(data_dir), prompts, "query")
    return {"model": model_name, "max_seq_length": max_seq_length, "counts_special_tokens": True,
            "prompts_included": bool(prompts and (prompts.get("query") or prompts.get("passage"))),
            "passages": count(passages, tokenizer, max_seq_length), "queries": count(queries, tokenizer, max_seq_length)}


def write_report(path, report: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")

"""How many texts exceed a model's maximum sequence length when the model embeds them with its OWN tokenizer.

The chunks are cut with the reference tokenizer, but each model tokenizes the text again (plus its prompt and the
document title that MTEB puts in front of the chunk). Texts longer than ``max_seq_length`` are truncated by the model.
This only RECORDS that (it changes nothing): ``truncation_report.json`` next to each run's results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def report_for_texts(passages: List[str], queries: List[str], tokenizer, max_seq_length: Optional[int],
                     prompts: Optional[Dict[str, Any]] = None, model_name: str = "") -> Dict[str, Any]:
    """Report for one model on given texts (the strings MTEB hands to the model, WITHOUT the prompt; the prompt is added here)."""
    passages = apply_prompt(list(passages), prompts, "passage")
    queries = apply_prompt(list(queries), prompts, "query")
    return {"model": model_name, "max_seq_length": max_seq_length, "counts_special_tokens": True,
            "prompts_included": bool(prompts and (prompts.get("query") or prompts.get("passage"))),
            "passages": count(passages, tokenizer, max_seq_length), "queries": count(queries, tokenizer, max_seq_length)}


def truncation_report(data_dir, tokenizer, max_seq_length: Optional[int], prompts: Optional[Dict[str, Any]] = None,
                      model_name: str = "") -> Dict[str, Any]:
    """Report for one model on one prepared task folder (prompts included in the counted text)."""
    return report_for_texts(embedded_passages(data_dir), embedded_queries(data_dir), tokenizer, max_seq_length, prompts, model_name)


# --------------------------------------------------------------------------------------------------------------------
# Phase 1: the external MTEB tasks (ChemNQRetrieval, ChemHotpotQARetrieval)
# --------------------------------------------------------------------------------------------------------------------
def mteb_passage_string(doc: Any) -> str:
    """What MTEB's dense retrieval encodes for a corpus entry: ``(title + " " + text).strip()`` for a dict with a title, the text of a dict
    without one, or the string itself."""
    if isinstance(doc, dict):
        return ((doc.get("title") + " " + doc["text"]) if "title" in doc and doc.get("title") is not None else doc["text"]).strip()
    return str(doc).strip()


def task_texts(task: Any, split: str) -> Tuple[List[str], List[str]]:
    """(passages, queries) of one evaluation split of a loaded monolingual MTEB retrieval task, as MTEB embeds them. Only queries that have
    relevance judgements are encoded (as in the evaluation)."""
    if getattr(task, "is_multilingual", False):
        raise NotImplementedError("multilingual tasks are not supported")
    corpus = task.corpus[split]
    judged = {str(k) for k in task.relevant_docs[split]}
    passages = [mteb_passage_string(d) for d in corpus.values()]
    qtexts = [q if isinstance(q, str) else str(q) for qid, q in task.queries[split].items() if str(qid) in judged]
    return passages, qtexts


def phase1_report(task: Any, tokenizer, max_seq_length: Optional[int], prompts: Optional[Dict[str, Any]] = None,
                  model_name: str = "", revision: str = "", eval_split: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Truncation of one model on one external task: one report per evaluation split (``splits``), plus the split the stage 2 screening uses
    (``reported_split``: ``params.phase1.eval_split``). ChemHotpotQARetrieval has three splits that share one corpus."""
    out: Dict[str, Any] = {"model": model_name, "revision": revision, "task": task.metadata.name, "max_seq_length": max_seq_length,
                           "counts_special_tokens": True, "splits": {}}
    seen: Dict[int, Dict[str, Any]] = {}                                     # splits that share one corpus are counted once
    for split in task.metadata.eval_splits:
        passages, queries = task_texts(task, split)
        key = hash(tuple(passages))
        rep = report_for_texts([] if key in seen else passages, queries, tokenizer, max_seq_length, prompts, model_name)
        if key in seen:
            rep["passages"] = seen[key]
        else:
            seen[key] = rep["passages"]
        out["splits"][split] = rep
    from pipeline_lib import phase1_split                                      # the split of params.phase1.eval_split, no fallback
    reported = phase1_split.split_of(task.metadata.name, eval_split if eval_split is not None else phase1_split.eval_splits())
    if reported not in out["splits"]:
        raise phase1_split.Phase1SplitError(f"{task.metadata.name}: the configured split {reported!r} is not among the task's splits {sorted(out['splits'])}")
    out["reported_split"] = reported
    out["prompts_included"] = bool(prompts and (prompts.get("query") or prompts.get("passage")))
    return out


PHASE1_COLUMNS = ["model", "revision", "task", "split", "reported", "max_seq_length", "prompts_included", "n_passages", "n_passages_truncated",
                  "pct_passages_truncated", "mean_passage_tokens", "max_passage_tokens", "mean_tokens_lost_when_truncated",
                  "n_queries", "n_queries_truncated", "pct_queries_truncated"]


def phase1_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flat rows (one per split) of a ``phase1_report`` for ``table_truncation_phase1.csv``."""
    rows = []
    for split, r in report["splits"].items():
        p, q = r["passages"], r["queries"]
        rows.append({"model": report["model"], "revision": report.get("revision", ""), "task": report["task"], "split": split,
                     "reported": split == report.get("reported_split"), "max_seq_length": report["max_seq_length"],
                     "prompts_included": report["prompts_included"], "n_passages": p["n_texts"], "n_passages_truncated": p["n_truncated"],
                     "pct_passages_truncated": p["pct_truncated"], "mean_passage_tokens": p["mean_tokens"], "max_passage_tokens": p["max_tokens"],
                     "mean_tokens_lost_when_truncated": p["mean_tokens_lost_when_truncated"], "n_queries": q["n_texts"],
                     "n_queries_truncated": q["n_truncated"], "pct_queries_truncated": q["pct_truncated"]})
    return rows


def collect_phase1(roots: Iterable[Any]) -> List[Dict[str, Any]]:
    """Rows of every ``truncation_report_<Task>.json`` found one level below each Phase 1 results folder in ``roots`` (missing folders skipped)."""
    rows: List[Dict[str, Any]] = []
    for root in roots:
        for p in sorted(Path(root).glob("*/truncation_report_*.json")) if Path(root).is_dir() else []:
            rep = json.loads(p.read_text(encoding="utf-8"))
            if "splits" in rep:
                rows += phase1_rows(rep)
    return rows


def write_report(path, report: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_phase1_reports(tasks: Iterable[Any], tokenizer, max_seq_length: Optional[int], prompts: Optional[Dict[str, Any]], model_name: str,
                         revision: str, out_dir: Any) -> List[Path]:
    """``truncation_report_<Task>.json`` per task in ``out_dir`` (the model's Phase 1 results folder). Records only: a failure is written to
    ``truncation_report_<Task>.error.txt`` and printed, it never stops the evaluation."""
    out_dir = Path(out_dir)
    written = []
    for task in tasks:
        name = task.metadata.name
        try:
            if not getattr(task, "data_loaded", False):
                task.load_data()
            rep = phase1_report(task, tokenizer, max_seq_length, prompts, model_name, revision)
            path = out_dir / f"truncation_report_{name}.json"
            path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
            written.append(path)
        except Exception as e:                      # noqa: BLE001 - recording must not stop a Phase 1 run
            print(f"[WARN] truncation report for {model_name} / {name} failed: {e}")
            (out_dir / f"truncation_report_{name}.error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
    return written

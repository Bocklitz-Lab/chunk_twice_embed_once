"""Phase 2 model selection from Phase 1 (stage 2): the published top-fraction rule and the inclusion-probability rule.

``top_fraction`` (published): a model passes on a task if its rank / n_models <= top_fraction (20 % of 41 = the top 8); the shortlist is the union over the tasks.

``inclusion_probability``: the same union rule made robust to the small Phase 1 query sets (27 and 18 test queries). In every query-level bootstrap resample
(``tools/phase1_intervals.py``: the SAME resampled queries for every model of a task, ``params.seed``) the models are ranked again; the inclusion probability of a model on
a task is the share of resamples in which it is in the top ``top_fraction`` of that task. A model is selected if its probability is >= ``min_inclusion_probability``
on AT LEAST ONE task. The observed top models (point estimate) are always selected as well (``top_fraction`` stays the published one): the rule can only widen the list.

The rule is stated in the config (``params.phase1.selection``) before the results are looked at; ``shortlist_selection_table`` reports the outcome under several thresholds,
it never chooses one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

RULES = ("top_fraction", "inclusion_probability")
REPORT_THRESHOLDS = (0.4, 0.5, 0.6)          # the thresholds that are reported next to the primary one (the primary comes from the config)


class ShortlistError(ValueError):
    """The selection rule is misconfigured or its inputs are inconsistent."""


def validate_selection(sel: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    sel = dict(sel or {})
    rule = sel.get("rule", "top_fraction")
    if rule not in RULES:
        raise ShortlistError(f"params.phase1.selection.rule must be one of {RULES}, got {rule!r}")
    tf = float(sel.get("top_fraction", 0.2))
    if not 0 < tf < 1:
        raise ShortlistError(f"top_fraction must lie in (0, 1), got {tf}")
    mp = float(sel.get("min_inclusion_probability", 0.5))
    if not 0 < mp <= 1:
        raise ShortlistError(f"min_inclusion_probability must lie in (0, 1], got {mp}")
    return {"rule": rule, "top_fraction": tf, "min_inclusion_probability": mp, "resamples": int(sel.get("resamples", 10000))}


def table_from_intervals(rows: Sequence[Mapping[str, Any]], primary: float, tasks: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """One row per model from the rows of ``tools.phase1_intervals.analyse`` (per-task rows carry geom_at_10, rank, in_top, top_share).

    Columns: model, revision, per task ``geom_<task>`` / ``rank_<task>`` / ``p_top_<task>`` / ``observed_top_<task>``, ``observed_top`` (any task),
    ``max_p_top``, ``selected_top_fraction`` (published rule), ``selected_p<thr>`` for 0.4 / 0.5 / 0.6 and ``selected_primary`` (threshold ``primary``).
    A threshold selects a model if its probability is >= thr on at least one task OR it is observed-top (the observed top is never dropped)."""
    task_rows = [r for r in rows if r.get("task") != "union"]
    tasks = list(tasks) if tasks else sorted({r["task"] for r in task_rows})
    by: Dict[tuple, Dict[str, Any]] = {}
    for r in task_rows:
        m = by.setdefault((r["model"], r.get("revision")), {"model": r["model"], "revision": r.get("revision")})
        t = r["task"]
        m[f"geom_{t}"], m[f"rank_{t}"], m[f"p_top_{t}"], m[f"observed_top_{t}"] = r["geom_at_10"], r["rank"], r["top_share"], bool(r["in_top"])
    out = []
    thresholds = sorted({*REPORT_THRESHOLDS, primary})
    for m in by.values():
        ps = [m.get(f"p_top_{t}", 0.0) for t in tasks]
        obs = any(m.get(f"observed_top_{t}", False) for t in tasks)
        m["observed_top"], m["max_p_top"] = obs, max(ps) if ps else 0.0
        m["selected_top_fraction"] = obs
        for thr in thresholds:
            m[f"selected_p{thr:g}"] = obs or any(p >= thr for p in ps)
        m["selected_primary"] = m[f"selected_p{primary:g}"]
        out.append(m)
    return sorted(out, key=lambda r: (-r["max_p_top"], r["model"]))


def select(table: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The models of the primary rule (rows of ``table_from_intervals``)."""
    chosen = [dict(r) for r in table if r["selected_primary"]]
    missing = [r["model"] for r in table if r["observed_top"] and not r["selected_primary"]]
    if missing:                                                      # cannot happen by construction; guards future edits
        raise ShortlistError(f"the observed top models must always be selected, but {missing} are not")
    return chosen


def counts(table: Sequence[Mapping[str, Any]], primary: float) -> Dict[str, int]:
    keys = ["selected_top_fraction"] + [f"selected_p{t:g}" for t in sorted({*REPORT_THRESHOLDS, primary})]
    return {k: sum(bool(r[k]) for r in table) for k in keys}

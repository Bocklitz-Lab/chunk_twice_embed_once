"""The evaluation split of every Phase 1 task, from ONE config key: ``params.phase1.eval_split`` (task -> split).

MTEB computes every split that a task has (ChemHotpotQARetrieval: train 180 queries, dev 8, test 18; ChemNQRetrieval: test 27) and writes them all into
``<Task>.json``. Stage 2 (the shortlist), the external table of ``make report``, ``tools/phase_transfer.py``, ``stages/run_bm25_mteb.py`` and the
efficiency table all read the split from here. There is no fallback order: a task without a configured split, or a result file without the configured
split, raises ``Phase1SplitError``.

``params.phase1.n_queries`` ({task: {split: number of queries}}, counted in the dataset files of BASF-AI/ChemHotpotQARetrieval and ChemNQRetrieval) is written
next to the split (columns ``split`` and ``n_queries``) in every Phase 1 output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pipeline_lib import config_params


class Phase1SplitError(ValueError):
    """The configured Phase 1 split is missing (in the config or in a result file)."""


def eval_splits(main_config: Optional[str | Path] = None) -> Dict[str, str]:
    cfg = (config_params.load_params(main_config).get("phase1") or {}).get("eval_split")
    if not cfg:
        raise Phase1SplitError("params.phase1.eval_split is not set in the main config (task -> split); there is no default order of splits")
    return dict(cfg)


def n_queries_table(main_config: Optional[str | Path] = None) -> Dict[str, Dict[str, int]]:
    return {t: dict(v) for t, v in ((config_params.load_params(main_config).get("phase1") or {}).get("n_queries") or {}).items()}


def split_of(task: str, eval_split: Mapping[str, str]) -> str:
    if task not in eval_split:
        raise Phase1SplitError(f"no evaluation split configured for task {task!r} (params.phase1.eval_split has {sorted(eval_split)})")
    return str(eval_split[task])


def n_queries_of(task: str, split: str, table: Optional[Mapping[str, Mapping[str, int]]]) -> Optional[int]:
    """Number of queries of (task, split) from ``params.phase1.n_queries``; None if the table has no entry (never guessed)."""
    return (table or {}).get(task, {}).get(split)


def pick_split(scores: Mapping[str, Any], task: str, eval_split: Mapping[str, str], where: str = "") -> Dict[str, Any]:
    """The score dict of the configured split of ``task`` from ``<Task>.json["scores"]``; raises if that split is not in the file."""
    split = split_of(task, eval_split)
    if not scores.get(split):
        raise Phase1SplitError(f"{where or task}: the configured split {split!r} is not in the result file (splits present: {sorted(scores)}); "
                               "fix params.phase1.eval_split or rerun the task")
    return scores[split][0]

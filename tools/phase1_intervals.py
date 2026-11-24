"""Phase 1 table with query-level intervals and shortlist stability (``make report``; Reviewer 2: the Phase 1 split and how fragile the top-8 cut is).

Input: the per-query files that stage 1 writes (``per_query_<Task>.csv`` next to the model's results; ``pipeline_lib/phase1_per_query.py``) for the split of
``params.phase1.eval_split``. Phase 1 queries have no paper grouping (18 queries on ChemHotpotQARetrieval test, 27 on ChemNQRetrieval), so everything is
QUERY-level (column ``level = query``): the queries are resampled with the draws of ``tools/stats.py`` (``params.report.resamples``, ``params.seed``), the same
resampled queries for every model of a task.

``table_phase1_ci.csv``, one row per model and task:
  geom_at_10, ndcg_at_10, recall_at_10          on the configured split (Geom@10 = sqrt(mean nDCG@10 x mean Recall@10), as everywhere)
  ci_low, ci_high                               95% percentile interval of Geom@10
  rank, in_top                                  rank among the models of the task (ties share the best rank) and whether stage 2 keeps it (rank / n_models <= top_percent)
  top_share                                     share of the resamples in which the model is in the top ``top_percent`` of its task (the stage 2 rule, applied inside the resample)
and, per model, a row with ``task = union``: ``top_share`` = share of the resamples in which the model is in the top of at least one task (it would be on
the shortlist), ``in_top`` = on the observed shortlist; ``union_size_mean`` / ``union_size_low`` / ``union_size_high`` (on every union row) give the size of the union
over the resamples (mean and 95% interval). BM25 and other baselines are not ranked.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import phase1_split  # noqa: E402
from tools import metrics as M, stats  # noqa: E402


def discover(roots: Sequence[Any]) -> Dict[str, Dict[tuple, Dict[str, Any]]]:
    """{task: {(model, revision): {"dir", "split", "n_queries"}}} from every ``per_query_<Task>.json`` below the Phase 1 roots (the first root wins)."""
    out: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
    for root in roots:
        if not Path(root).is_dir():
            continue
        for p in sorted(Path(root).glob("*/per_query_*.json")):
            info = json.loads(p.read_text(encoding="utf-8"))
            metas = sorted(p.parent.glob("**/model_meta.json"))
            if len(metas) != 1:
                continue
            meta = json.loads(metas[0].read_text(encoding="utf-8"))
            if str(meta.get("name") or "").lower().startswith("bm25"):
                continue
            key = (meta["name"], meta.get("revision"))
            out.setdefault(info["task"], {}).setdefault(key, {"dir": p.parent, "split": info["split"], "n_queries": info["n_queries"]})
    return out


def analyse(roots: Sequence[Any], eval_split: Dict[str, str], *, top_percent: float, resamples: int, seed: int) -> List[Dict[str, Any]]:
    found = discover(roots)
    rows: List[Dict[str, Any]] = []
    in_top_resample: Dict[tuple, np.ndarray] = {}
    observed_top: Dict[tuple, bool] = {}
    for task in sorted(found):
        want = phase1_split.split_of(task, eval_split)
        models = sorted(found[task])
        bad = [m for m in models if found[task][m]["split"] != want]
        if bad:
            raise phase1_split.Phase1SplitError(f"{task}: per-query files of {bad[:3]} are for another split than the configured {want!r}")
        nd, rc, qids = [], [], None
        for m in models:
            pq = M.load_per_query_csv(found[task][m]["dir"] / f"per_query_{task}.csv")
            ids = sorted(pq)
            if qids is None:
                qids = ids
            elif ids != qids:
                raise ValueError(f"{task}: {m} has other queries than the first model")
            nd.append([pq[q]["ndcg_at_10"] for q in qids]); rc.append([pq[q]["recall_at_10"] for q in qids])
        nd, rc = np.array(nd), np.array(rc)
        n_models, n_q = len(models), len(qids)
        idx = stats.resample_indices(n_q, resamples, seed)                               # (B, Q), the same queries for every model
        gn = nd[:, idx].mean(axis=2).T                                                   # (B, M)
        gr = rc[:, idx].mean(axis=2).T
        geom_b = np.sqrt(gn * gr)
        point = np.sqrt(nd.mean(axis=1) * rc.mean(axis=1))
        rank_b = 1 + (geom_b[:, None, :] > geom_b[:, :, None]).sum(axis=2)               # (B, M): ties share the best rank
        top_b = rank_b / n_models <= top_percent
        rank = 1 + (point[None, :] > point[:, None]).sum(axis=1)
        lo, hi = np.quantile(geom_b, [0.025, 0.975], axis=0)
        for i, m in enumerate(models):
            info = found[task][m]
            rows.append({"model": m[0], "revision": m[1], "task": task, "split": info["split"], "n_queries": n_q, "level": "query", "n_models": n_models,
                         "geom_at_10": point[i], "ndcg_at_10": nd[i].mean(), "recall_at_10": rc[i].mean(), "ci_low": lo[i], "ci_high": hi[i],
                         "rank": int(rank[i]), "in_top": bool(rank[i] / n_models <= top_percent), "top_share": top_b[:, i].mean(), "resamples": resamples, "seed": seed})
            in_top_resample[(m, task)] = top_b[:, i]
            observed_top[(m, task)] = bool(rank[i] / n_models <= top_percent)
    models_all = sorted({m for t in found for m in found[t]})
    if models_all and len(found) > 1:
        tasks = sorted(found)
        B = len(next(iter(in_top_resample.values())))
        union = np.array([np.any([in_top_resample.get((m, t), np.zeros(B, bool)) for t in tasks], axis=0) for m in models_all])      # (M, B)
        size = union.sum(axis=0)
        ulo, uhi = np.quantile(size, [0.025, 0.975])
        for i, m in enumerate(models_all):
            rows.append({"model": m[0], "revision": m[1], "task": "union", "split": "; ".join(f"{t}={phase1_split.split_of(t, eval_split)}" for t in tasks),
                         "n_queries": "; ".join(f"{t}={next(iter(found[t].values()))['n_queries']}" for t in tasks), "level": "query", "n_models": len(models_all),
                         "in_top": any(observed_top.get((m, t), False) for t in tasks), "top_share": union[i].mean(), "resamples": resamples, "seed": seed,
                         "union_size_mean": size.mean(), "union_size_low": ulo, "union_size_high": uhi})
    return rows

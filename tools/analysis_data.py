"""Shared loader and resampling for the analysis tools that run on the outputs of the full rerun (``make analysis``).

  RunSet            a matrix of runs x queries for every per-query metric, the query -> paper index and the run metadata
  load_runs         reads <root>/tests/<run>/results/*/per_query_metrics*.csv for one or several artifacts roots (main + transfer)
  PaperBootstrap    paper-level resampling of ALL runs at once: the same resampled papers for every run, so differences between runs
                    and rankings of runs are paired. Uses the draws of ``tools/stats.py`` (same seed, same papers) so the intervals of
                    the analysis tools agree with ``stats_ci.csv`` of ``make report``.

The 952 questions come from 149 papers (1-10 questions each); a paper is the resampling unit. Query ids are the row indices of ``qa.jsonl``.
Question type and subfield come only from ``load_question_metadata`` (query id = row index of qa.jsonl).
``kind`` of a run: dense / bm25 / hybrid / rerank (from ``run_info.json``). When the same (model, configuration) exists in several roots, the
first root wins. Run metadata is joined from ``runs_overview.csv`` when given (columns such as n_chunks, evaluation_time_s).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params, question_metadata as QM  # noqa: E402
from pipeline_lib.dataset_pin import QUESTION_METADATA_FILE  # noqa: E402
from pipeline_lib.span_labels import norm_doc_id  # noqa: E402
from tools import metrics as M  # noqa: E402
from tools import stats  # noqa: E402

BM25 = "bm25"
DEFAULT_METRICS = ("ndcg_at_10", "recall_at_10", "geom_at_10")
LABEL_FILES = {"span": "per_query_metrics.csv", "lexical": "per_query_metrics__qrels_lexical.csv"}


def run_kind(info: Dict[str, Any]) -> str:
    if info.get("hybrid"):
        return "hybrid"
    if info.get("rerank"):
        return "rerank"
    return "bm25" if str(info.get("model", "")).startswith(BM25) else "dense"


def run_key(model: str, chunker: str, size: Any, overlap: Any) -> str:
    return f"{model}|{chunker}|{int(size)}|{int(overlap)}"


# ------------------------------------------------------------------------------------------------------------------
# configuration defaults (nothing is hard-coded in the tools)
# ------------------------------------------------------------------------------------------------------------------
def defaults(main_config: Optional[str] = None) -> Dict[str, Any]:
    """Resamples, seed, cover level and report folder from ``params`` of the main config."""
    p = config_params.load_params(main_config)
    rep = p.get("report") or {}
    return {"resamples": int(rep.get("resamples", 10000)), "seed": int(p.get("seed", stats.DEFAULT_SEED)),
            "cover_level": (p.get("labeling") or {}).get("cover_level"), "report_dir": rep.get("out_dir", "results/v2-revision")}


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Options every analysis tool shares."""
    ap.add_argument("--roots", nargs="+", required=True, help="artifacts roots with a tests/ folder (main run first, then e.g. the transfer root)")
    ap.add_argument("--qa", required=True, help="qa.jsonl of the pinned release (query id = row index; corpus_id = paper)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--main-config", default=str(config_params.DEFAULT_MAIN_CONFIG))
    ap.add_argument("--resamples", type=int, help="paper-level bootstrap resamples (default: params.report.resamples)")
    ap.add_argument("--seed", type=int, help="default: params.seed")
    ap.add_argument("--label-set", choices=list(LABEL_FILES), default="span", help="which per-query metric files to read (default: span, the primary labels)")


def resolve(args: argparse.Namespace) -> Tuple[int, int]:
    d = defaults(args.main_config)
    return (args.resamples if args.resamples is not None else d["resamples"]), (args.seed if args.seed is not None else d["seed"])


# ------------------------------------------------------------------------------------------------------------------
# runs
# ------------------------------------------------------------------------------------------------------------------
@dataclass
class RunSet:
    """Runs x queries. ``metrics[name]`` is an (R, Q) array; ``meta[i]`` describes run i; ``groups[j]`` is the paper of query j."""
    meta: List[Dict[str, Any]]
    qids: List[str]
    metrics: Dict[str, np.ndarray]
    groups: np.ndarray
    index: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {m["key"]: i for i, m in enumerate(self.meta)}

    def __len__(self) -> int:
        return len(self.meta)

    def select(self, **where: Any) -> List[int]:
        """Indices of the runs whose metadata equal (or, for a list/set/tuple, are in) the given values, e.g. ``select(kind="dense", size=512)``."""
        out = []
        for i, m in enumerate(self.meta):
            if all((m.get(k) in v) if isinstance(v, (list, set, tuple, frozenset)) else m.get(k) == v for k, v in where.items()):
                out.append(i)
        return out

    def geom(self, rows: Optional[Sequence[int]] = None) -> np.ndarray:
        """Geom@10 of each run: sqrt(mean nDCG@10 x mean Recall@10) over all queries (the definition of the paper)."""
        sel = slice(None) if rows is None else list(rows)
        return np.sqrt(self.metrics["ndcg_at_10"][sel].mean(axis=1) * self.metrics["recall_at_10"][sel].mean(axis=1))

    def cells(self) -> List[Tuple[str, int, int]]:
        return sorted({(m["chunker"], m["size"], m["overlap"]) for m in self.meta})


def discover(roots: Iterable[Any]) -> List[Dict[str, Any]]:
    """Run folders below ``<root>/tests`` (a folder with run_info.json and one results/<model>/per_query_metrics.csv); the first root wins."""
    seen: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        tests = Path(root) / "tests"
        if not tests.is_dir():
            continue
        for d in sorted(p for p in tests.iterdir() if p.is_dir() and (p / "run_info.json").is_file()):
            info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
            outs = [o for o in (d / "results").glob("*") if o.is_dir() and (o / "per_query_metrics.csv").is_file()]
            if len(outs) != 1:
                continue
            key = run_key(info["model"], info["chunker"], info["size"], info["overlap"])
            if key in seen:
                continue
            kind = run_kind(info)
            src = (info.get("hybrid") or info.get("rerank") or {}).get("dense")
            seen[key] = {"key": key, "model": info["model"], "kind": kind, "chunker": info["chunker"], "size": int(info["size"]),
                         "overlap": int(info["overlap"]), "source_dense": src, "run_dir": str(d), "out_dir": str(outs[0]), "root": str(root),
                         "rerank_top_n": (info.get("rerank") or {}).get("top_n"), "prepared_dir": info.get("prepared_dir")}
    return sorted(seen.values(), key=lambda m: (m["chunker"], m["size"], m["overlap"], m["kind"], m["model"]))


def paper_index(qa_jsonl: Any, qids: Sequence[str]) -> np.ndarray:
    """Normalised paper id (``norm_doc_id``: ``098`` = ``98``) of each query id (= row index of qa.jsonl)."""
    docs = [norm_doc_id(json.loads(l)["corpus_id"]) for l in open(qa_jsonl, encoding="utf-8") if l.strip()]
    return np.array([docs[int(q)] for q in qids])


def load_question_metadata(qa_jsonl: Any, path: Optional[Any] = None) -> QM.QuestionMetadata:
    """The official question type and subfields per question (``question_metadata.jsonl`` next to qa.jsonl unless ``path`` is given), checked against
    ``qa_jsonl`` row by row (``pipeline_lib/question_metadata.py``). Raises ``QuestionMetadataError`` (with the download command) if the file is
    missing or does not match: the analysis tools have no fallback and never guess a question type from the text."""
    return QM.load(path or Path(qa_jsonl).with_name(QUESTION_METADATA_FILE), qa_jsonl)


def read_overview(path: Optional[Any]) -> Dict[str, Dict[str, str]]:
    """``runs_overview.csv`` rows by run key (empty if no file)."""
    if not path or not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {run_key(r["model"], r["chunker"], r["size"], r["overlap"]): r for r in csv.DictReader(f)}


def load_runs(roots: Iterable[Any], qa_jsonl: Any, *, metrics: Sequence[str] = DEFAULT_METRICS, label_set: str = "span",
              kinds: Optional[Sequence[str]] = None, overview: Optional[Any] = None) -> RunSet:
    """The per-query metrics of every run in ``roots`` (optionally only some ``kinds``). All runs must cover the same queries."""
    runs = [m for m in discover(roots) if kinds is None or m["kind"] in kinds]
    if not runs:
        raise FileNotFoundError(f"no run with per-query metrics below {[str(r) for r in roots]}")
    fname = LABEL_FILES[label_set]
    qids: Optional[List[str]] = None
    rows: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    keep: List[Dict[str, Any]] = []
    ov = read_overview(overview)
    for run in runs:
        path = Path(run["out_dir"]) / fname
        if not path.is_file():
            continue
        pq = M.load_per_query_csv(path)
        ids = sorted(pq, key=lambda x: (len(x), x))
        if qids is None:
            qids = ids
        elif ids != qids:
            raise ValueError(f"{path}: not the same queries as the other runs")
        for m in metrics:
            rows[m].append(np.array([pq[q][m] for q in qids], dtype=float))
        keep.append({**ov.get(run["key"], {}), **run})
    if qids is None:
        raise FileNotFoundError(f"no {fname} found")
    return RunSet(meta=keep, qids=qids, metrics={m: np.vstack(v) for m, v in rows.items()}, groups=paper_index(qa_jsonl, qids))


# ------------------------------------------------------------------------------------------------------------------
# paper-level resampling of many runs at once
# ------------------------------------------------------------------------------------------------------------------
class PaperBootstrap:
    """Paper-level bootstrap of every run at once.

    ``idx`` (B x G) are the resampled papers, drawn as in ``tools/stats.py`` (``default_rng(seed)``, ``integers(0, G, (B, G))``, papers in sorted
    order), so for the same seed the draws are the ones behind ``stats_ci.csv``. A resample keeps all questions of a drawn paper. The mean of a
    run in a resample is (sum over drawn papers of the paper's metric sum) / (number of questions of the drawn papers), which equals
    ``stats.sample_stats``. ``counts`` (B x G) says how often each paper was drawn."""

    def __init__(self, groups: np.ndarray, resamples: int, seed: int):
        self.uniq, self.inv = np.unique(groups, return_inverse=True)
        self.G, self.B, self.seed = len(self.uniq), int(resamples), int(seed)
        self.idx = stats._resample_indices(self.G, self.B, np.random.default_rng(self.seed))
        flat = (np.arange(self.B)[:, None] * self.G + self.idx).ravel()
        self.counts = np.bincount(flat, minlength=self.B * self.G).reshape(self.B, self.G).astype(np.float64)
        self.n_g = np.bincount(self.inv, minlength=self.G).astype(np.float64)
        self.onehot = np.zeros((len(groups), self.G))
        self.onehot[np.arange(len(groups)), self.inv] = 1.0
        self.n_questions = self.counts @ self.n_g                               # questions in each resample

    def paper_sums(self, values: np.ndarray) -> np.ndarray:
        """(R, Q) per-query values -> (R, G) sums per paper."""
        return np.atleast_2d(values) @ self.onehot

    def resample_means(self, values: np.ndarray) -> np.ndarray:
        """(R, Q) per-query values -> (B, R): the mean over the questions of the resampled papers, per resample and run."""
        return (self.counts @ self.paper_sums(values).T) / self.n_questions[:, None]

    def blocks(self, n_runs: int, block: int = 256) -> Iterator[Tuple[int, int]]:
        for a in range(0, n_runs, block):
            yield a, min(a + block, n_runs)

    def geom_resamples(self, ndcg: np.ndarray, recall: np.ndarray) -> np.ndarray:
        """(R, Q) nDCG@10 and Recall@10 -> (B, R) Geom@10 of every run in every resample (sqrt of the product of the resampled means)."""
        out = np.empty((self.B, ndcg.shape[0]))
        for a, b in self.blocks(ndcg.shape[0]):
            out[:, a:b] = np.sqrt(self.resample_means(ndcg[a:b]) * self.resample_means(recall[a:b]))
        return out

    def percentile_ci(self, resampled: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Percentile interval over the resamples (axis 0) of a (B, ...) array."""
        return np.quantile(resampled, alpha / 2, axis=0), np.quantile(resampled, 1 - alpha / 2, axis=0)


# ------------------------------------------------------------------------------------------------------------------
# small output helper shared by the tools
# ------------------------------------------------------------------------------------------------------------------
def write_rows(path: Any, rows: List[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> Path:
    """CSV with a stable column order (union of the keys in first-seen order unless ``columns`` is given); floats with 6 decimals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns else []
    if not columns:
        for r in rows:
            cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    return path

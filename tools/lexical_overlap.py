"""Lexical-overlap diagnostic: does the question share its words with the evidence span, and is BM25's advantage over a
dense model larger where it does?

Per question (952, one table for all configurations because the evidence spans do not depend on the chunking):
  overlap_share   = |question content words that also occur in the evidence span| / |question content words|
                    content words = unique lower-case ``\\w+`` tokens that are not in scikit-learn's English stop-word list
                    (318 words); numbers count as content words; a question without content words has no share (NaN).
Per question and run (every configuration, every model):
  rank of the evidence chunk = rank of the best-ranked relevant chunk (span labels) in the saved top-100; 101 = not in the top 100.
Per configuration and (dense model, BM25) pair:
  advantage = per-question metric of BM25 minus that of the dense model (nDCG@10, Recall@10, reciprocal rank at 100),
  mean advantage with a paper-level bootstrap CI per overlap bin, Spearman rho(overlap_share, advantage), and the slope of the
  advantage on overlap_share (least squares, paper-level bootstrap CI): the change of the BM25 advantage when the whole
  question is in the span rather than none of it.

  python -m tools.lexical_overlap --tests-root artifacts/pilot_v2/tests --qa artifacts/pilot_v2/chemquests_hf_db/qa.jsonl \\
        --dense intfloat/e5-large-v2 BAAI/bge-m3 --sparse bm25 --out-dir artifacts/pilot_v2/overlap
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import metrics as M  # noqa: E402

SEED = 20261121
B = 1000
BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)          # [0, .25) [.25, .5) [.5, .75) [.75, 1.0]; 1.0 belongs to the last bin
NOT_FOUND = 101
_TOKEN = re.compile(r"\w+", re.UNICODE)


def stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return ENGLISH_STOP_WORDS


def content_words(text: str, stop=None) -> set:
    stop = stopwords() if stop is None else stop
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in stop}


def overlap_share(question: str, span: str, stop=None) -> Tuple[float, int, int]:
    """(share, n question content words, n of them in the span); share is NaN without content words."""
    q = content_words(question, stop)
    if not q:
        return float("nan"), 0, 0
    hit = len(q & content_words(span, stop))
    return hit / len(q), len(q), hit


def question_table(qa_jsonl) -> List[Dict[str, Any]]:
    stop = stopwords()
    rows = []
    with open(qa_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            r = json.loads(line)
            share, nq, hit = overlap_share(r["question"], r["content"], stop)
            rows.append({"query_id": str(i), "corpus_id": str(r["corpus_id"]), "n_question_content_words": nq,
                         "n_in_span": hit, "overlap_share": share, "span_chars": len(r["content"])})
    return rows


def evidence_rank(ranking: Dict[str, float], relevant: Iterable[str]) -> int:
    rel = set(relevant)
    for rank, (doc, _s) in enumerate(M.ranked(ranking), start=1):
        if doc in rel:
            return rank
    return NOT_FOUND


def distribution(shares: Sequence[float]) -> Dict[str, Any]:
    a = np.array([s for s in shares if not math.isnan(s)])
    out = {"n": len(shares), "n_with_content_words": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
           "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max()),
           "pct_zero_overlap": 100 * float((a == 0).mean()), "pct_full_overlap": 100 * float((a == 1).mean())}
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (a >= lo) & ((a < hi) if hi < 1.0 else (a <= hi))
        out[f"pct_bin_{lo:g}_{hi:g}"] = 100 * float(m.mean())
    return out


def bin_of(share: float) -> Optional[str]:
    if math.isnan(share):
        return None
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if share >= lo and (share < hi or (hi >= 1.0 and share <= hi)):
            return f"[{lo:g},{hi:g}{']' if hi >= 1.0 else ')'}"
    return None


def _cluster_boot(values: np.ndarray, groups: np.ndarray, stat, rng, B: int = B) -> Tuple[float, float]:
    """95% percentile CI of ``stat(index array)`` when the papers (groups) are resampled."""
    uniq = np.unique(groups)
    members = [np.where(groups == g)[0] for g in uniq]
    res = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        res[b] = stat(np.concatenate([members[i] for i in pick]))
    return float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def advantage_rows(table: List[Dict[str, Any]], bm25_pq: Dict[str, Dict[str, float]], dense_pq: Dict[str, Dict[str, float]],
                   seed: int = SEED, label: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    from scipy.stats import spearmanr
    rows = []
    keep = [r for r in table if not math.isnan(r["overlap_share"])]
    qids = [r["query_id"] for r in keep]
    share = np.array([r["overlap_share"] for r in keep])
    groups = np.array([r["corpus_id"] for r in keep])
    bins = np.array([bin_of(s) for s in share])
    for metric in ("ndcg_at_10", "recall_at_10", "mrr_at_100"):
        adv = np.array([bm25_pq[q][metric] - dense_pq[q][metric] for q in qids])
        rng = np.random.default_rng(seed)
        for b in [f"[{lo:g},{hi:g}{']' if hi >= 1.0 else ')'}" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])] + ["all"]:
            idx = np.arange(len(qids)) if b == "all" else np.where(bins == b)[0]
            if len(idx) == 0:
                continue
            lo, hi = _cluster_boot(adv[idx], groups[idx], lambda i, a=adv[idx]: a[i].mean(), rng) if len(np.unique(groups[idx])) > 1 else (float("nan"),) * 2
            rows.append({**(label or {}), "metric": metric, "bin": b, "n_questions": len(idx),
                         "mean_bm25": float(np.mean([bm25_pq[qids[i]][metric] for i in idx])),
                         "mean_dense": float(np.mean([dense_pq[qids[i]][metric] for i in idx])),
                         "mean_advantage_bm25": float(adv[idx].mean()), "ci_low": lo, "ci_high": hi})
        rho = spearmanr(share, adv)
        slope = float(np.polyfit(share, adv, 1)[0])
        lo, hi = _cluster_boot(adv, groups, lambda i: float(np.polyfit(share[i], adv[i], 1)[0]), np.random.default_rng(seed + 1))
        rows.append({**(label or {}), "metric": metric, "bin": "trend", "n_questions": len(qids), "spearman_rho": float(rho.statistic),
                     "spearman_p": float(rho.pvalue), "slope_per_unit_overlap": slope, "slope_ci_low": lo, "slope_ci_high": hi})
    return rows


def _model_out(run_dir: Path) -> Path:
    outs = [d for d in (run_dir / "results").glob("*") if d.is_dir() and (d / "run_top100.trec.gz").exists()]
    if len(outs) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one results/<model>/ with run_top100.trec.gz")
    return outs[0]


def analyse(tests_root: Path, qa_jsonl, dense: Sequence[str], sparse: str = "bm25", out_dir: Optional[Path] = None,
            seed: int = SEED) -> Dict[str, Any]:
    from tools import hybrid
    table = question_table(qa_jsonl)
    cells = hybrid.discover(Path(tests_root))
    rank_rows: List[Dict[str, Any]] = []
    adv_rows: List[Dict[str, Any]] = []
    by_q = {r["query_id"]: dict(r) for r in table}
    for (chunker, size, overlap), models in sorted(cells.items()):
        if sparse not in models:
            continue
        prepared = Path(json.loads((models[sparse] / "run_info.json").read_text())["prepared_dir"])
        qrels = M.load_qrels(prepared / "mteb_task" / "qrels_span_c090.jsonl")
        pqs: Dict[str, Dict[str, Dict[str, float]]] = {}
        for model in [sparse, *[m for m in dense if m in models]]:
            out = _model_out(models[model])
            run = M.load_run(out / "run_top100.trec.gz")
            pqs[model] = M.load_per_query_csv(out / "per_query_metrics.csv")
            for q in by_q:
                rank_rows.append({"chunker": chunker, "size": size, "overlap": overlap, "model": model, "query_id": q,
                                  "overlap_share": by_q[q]["overlap_share"], "evidence_rank": evidence_rank(run.get(q, {}), qrels.get(q, {}))})
        for model in dense:
            if model in pqs:
                adv_rows += advantage_rows(table, pqs[sparse], pqs[model], seed,
                                           {"chunker": chunker, "size": size, "overlap": overlap, "dense": model, "sparse": sparse})
    res = {"distribution": distribution([r["overlap_share"] for r in table]), "questions": table, "ranks": rank_rows, "advantage": adv_rows}
    res["rank_by_bin"] = rank_by_bin(rank_rows)
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / "overlap_questions.csv", table)
        _write(out_dir / "overlap_evidence_rank.csv", rank_rows)
        _write(out_dir / "overlap_advantage.csv", adv_rows)
        _write(out_dir / "overlap_rank_by_bin.csv", res["rank_by_bin"])
        (out_dir / "overlap_distribution.json").write_text(json.dumps(res["distribution"], indent=2), encoding="utf-8")
    return res


def rank_by_bin(rank_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    g: Dict[tuple, List[int]] = defaultdict(list)
    for r in rank_rows:
        b = bin_of(r["overlap_share"])
        if b:
            g[(r["chunker"], r["size"], r["overlap"], r["model"], b)].append(r["evidence_rank"])
    out = []
    for (ch, s, o, m, b), ranks in sorted(g.items()):
        a = np.array(ranks)
        out.append({"chunker": ch, "size": s, "overlap": o, "model": m, "bin": b, "n_questions": len(a),
                    "pct_rank1": 100 * float((a == 1).mean()), "pct_in_top10": 100 * float((a <= 10).mean()),
                    "pct_not_in_top100": 100 * float((a == NOT_FOUND).mean()), "median_rank": float(np.median(a)),
                    "mean_reciprocal_rank": float(np.mean(1.0 / np.where(a == NOT_FOUND, np.inf, a)))})
    return out


def _write(path, rows: List[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests-root", required=True)
    ap.add_argument("--qa", required=True, help="qa.jsonl of the pinned release (query id = row index)")
    ap.add_argument("--dense", nargs="+", required=True)
    ap.add_argument("--sparse", default="bm25")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    res = analyse(Path(args.tests_root), args.qa, args.dense, args.sparse, Path(args.out_dir), args.seed)
    print(json.dumps(res["distribution"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

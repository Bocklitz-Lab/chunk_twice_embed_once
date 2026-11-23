"""Agreement of two annotators on the validation sheets of ``tools/validation_sample.py`` (percent agreement, Cohen's kappa, weighted kappa).

Reads two filled sheets (CSV; same items, e.g. ``validation_sheet_annotator_1.csv`` and ``..._2.csv``) and, for each answer column
(``span_correct``, ``span_sufficient``, ``chunks_sufficient_<configuration>``), over the items both annotators answered:

  n                    items with a valid answer from both (valid: yes / partial / no, case and spaces ignored)
  percent_agreement    share of items with the same answer
  kappa                Cohen's kappa (unweighted): (po - pe) / (1 - pe)
  kappa_weighted       linearly weighted kappa for the ordered answers no < partial < yes: 1 - sum(w o) / sum(w e), w = |i - j|
  ci_low / ci_high     95% percentile interval of the weighted and of the unweighted kappa, resampling PAPERS (the sampled questions of one paper
                       are not independent), ``resamples`` draws (default: params.report.resamples)

A kappa is undefined (NaN) when both annotators always give one and the same answer (chance agreement = 1); the percent agreement is still reported.
Also written: a table of every disagreement (item, question, column, both answers, both comments) for the discussion between the annotators.

  python -m tools.kappa --sheets validation_sheet_a.csv validation_sheet_b.csv --out-dir results/validation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params  # noqa: E402

LEVELS = ("no", "partial", "yes")                       # the order of the answers (weights = distance in this order)
K = len(LEVELS)
ANSWER_PREFIXES = ("span_correct", "span_sufficient", "chunks_sufficient_")


def clean(value: Any) -> Optional[int]:
    """Level number (0 no, 1 partial, 2 yes) of an answer; None if empty or not one of the three words."""
    v = str(value or "").strip().lower()
    return LEVELS.index(v) if v in LEVELS else None


def answer_columns(header: Sequence[str]) -> List[str]:
    return [c for c in header if c == "span_correct" or c == "span_sufficient" or c.startswith("chunks_sufficient_")]


def _weights(kind: str) -> np.ndarray:
    i = np.arange(K)
    d = np.abs(i[:, None] - i[None, :]).astype(float)
    return d if kind == "linear" else (np.zeros((K, K)) if kind == "none" else d ** 2)


def kappa_from_counts(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Cohen's kappa for confusion tables ``counts`` of shape (..., K, K) (rows = annotator A, columns = annotator B).

    With ``weights`` of zeros it is the unweighted kappa (identity disagreement); otherwise 1 - sum(w o) / sum(w e). NaN if chance agreement is 1."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        o = counts / n[..., None, None]
        e = o.sum(axis=-1)[..., :, None] * o.sum(axis=-2)[..., None, :]
        w = weights if weights.any() else 1.0 - np.eye(K)
        return 1.0 - (w * o).sum(axis=(-2, -1)) / (w * e).sum(axis=(-2, -1))


def confusion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bincount(a * K + b, minlength=K * K).reshape(K, K)


def bootstrap_kappa(a: np.ndarray, b: np.ndarray, papers: np.ndarray, resamples: int, seed: int) -> Dict[str, Tuple[float, float]]:
    """Paper-level 95% intervals of the unweighted and of the weighted kappa (percentile; resamples with an undefined kappa are left out)."""
    uniq, inv = np.unique(papers, return_inverse=True)
    G = len(uniq)
    per_paper = np.zeros((G, K * K))
    np.add.at(per_paper, (inv, a * K + b), 1.0)
    idx = np.random.default_rng(seed).integers(0, G, size=(resamples, G))
    draws = np.bincount((np.arange(resamples)[:, None] * G + idx).ravel(), minlength=resamples * G).reshape(resamples, G)     # how often each paper is drawn
    cells = (draws @ per_paper).reshape(resamples, K, K)
    out = {}
    for name, w in (("kappa", _weights("none")), ("kappa_weighted", _weights("linear"))):
        k = kappa_from_counts(cells, w)
        k = k[np.isfinite(k)]
        out[name] = (float(np.percentile(k, 2.5)), float(np.percentile(k, 97.5))) if len(k) else (float("nan"), float("nan"))
    return out


def read_sheet(path: Any) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def compare(sheet_a: Any, sheet_b: Any, resamples: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """(one row per answer column, disagreements, invalid-answer messages)."""
    ha, ra = read_sheet(sheet_a)
    hb, rb = read_sheet(sheet_b)
    if [r["query_id"] for r in ra] != [r["query_id"] for r in rb]:
        raise ValueError("the two sheets do not hold the same items in the same order")
    cols = [c for c in answer_columns(ha) if c in hb]
    rows, dis, invalid = [], [], []
    for col in cols:
        qa_, qb_ = [clean(r[col]) for r in ra], [clean(r[col]) for r in rb]
        for who, sheet, vals, raw in (("A", sheet_a, qa_, ra), ("B", sheet_b, qb_, rb)):
            for i, (v, r) in enumerate(zip(vals, raw)):
                if v is None and str(r[col]).strip() != "":
                    invalid.append(f"{Path(sheet).name}: item {r['item']} {col}: {r[col]!r} is not yes / partial / no")
        keep = [i for i in range(len(ra)) if qa_[i] is not None and qb_[i] is not None]
        row: Dict[str, Any] = {"column": col, "n": len(keep), "n_items": len(ra)}
        if keep:
            a, b = np.array([qa_[i] for i in keep]), np.array([qb_[i] for i in keep])
            papers = np.array([ra[i]["corpus_id"] for i in keep])
            cm = confusion(a, b)
            ci = bootstrap_kappa(a, b, papers, resamples, seed)
            row.update({"percent_agreement": 100.0 * float((a == b).mean()), "kappa": float(kappa_from_counts(cm, _weights("none"))),
                        "kappa_weighted": float(kappa_from_counts(cm, _weights("linear"))), "kappa_ci_low": ci["kappa"][0], "kappa_ci_high": ci["kappa"][1],
                        "kappa_weighted_ci_low": ci["kappa_weighted"][0], "kappa_weighted_ci_high": ci["kappa_weighted"][1],
                        "n_papers": len(set(papers)), **{f"A_{LEVELS[x]}__B_{LEVELS[y]}": int(cm[x, y]) for x in range(K) for y in range(K)}})
            for i in keep:
                if qa_[i] != qb_[i]:
                    dis.append({"item": ra[i]["item"], "query_id": ra[i]["query_id"], "column": col, "question": ra[i]["question"],
                                "answer_A": LEVELS[qa_[i]], "answer_B": LEVELS[qb_[i]], "comment_A": ra[i].get("comment", ""), "comment_B": rb[i].get("comment", "")})
        rows.append(row)
    return rows, dis, invalid


def write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheets", nargs=2, required=True, metavar=("SHEET_A", "SHEET_B"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--main-config", default=str(config_params.DEFAULT_MAIN_CONFIG))
    ap.add_argument("--resamples", type=int, help="default: params.report.resamples")
    ap.add_argument("--seed", type=int, help="default: params.seed")
    args = ap.parse_args(argv)
    p = config_params.load_params(args.main_config)
    resamples = args.resamples or int((p.get("report") or {}).get("resamples", 10000))
    seed = args.seed if args.seed is not None else int(p.get("seed", 0))
    rows, dis, invalid = compare(args.sheets[0], args.sheets[1], resamples, seed)
    out = Path(args.out_dir)
    write(out / "kappa.csv", rows)
    write(out / "kappa_disagreements.csv", dis or [{"item": "", "query_id": "", "column": "", "question": "", "answer_A": "", "answer_B": "", "comment_A": "", "comment_B": ""}][:0])
    for msg in invalid:
        print(f"[WARN] {msg}")
    for r in rows:
        if r["n"]:
            print(f"{r['column']:44s} n={r['n']:3d} agreement {r['percent_agreement']:5.1f}%  kappa {r['kappa']:.3f}  weighted {r['kappa_weighted']:.3f}")
        else:
            print(f"{r['column']:44s} no item answered by both")
    print(f"-> {out / 'kappa.csv'}, {len(dis)} disagreements in {out / 'kappa_disagreements.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

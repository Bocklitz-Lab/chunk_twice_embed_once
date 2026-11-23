"""Rescore saved predictions with other labels (CPU only; no embedding, no re-chunking).

For each configuration the saved ranking (``<task>_predictions.json``, written by stage 6) is scored with
  * a qrels file (``--qrels``; or ``--qrels-file`` names inside a pipeline folder), or
  * the overlap table of stage 5 (``overlap_table.jsonl``) plus one or more cover levels ``--cover-level c ...``:
    the labels for each c are derived from the table with the same rule as stages/make_span_qrels.py.

Metrics (pytrec_eval, exactly as MTEB computes them): nDCG@10, Recall@10, both averaged over ALL queries (a question
without a relevant chunk scores 0 and stays in the mean), and Geom@10 = sqrt(nDCG@10 * Recall@10) of those two
means (``geom_at_10_query_mean`` is the mean of the per-query geometric means). Predictions are sorted by score
before use. One CSV row per (config, label set, c).

Single configuration:
  python -m tools.rescore --predictions P.json --overlap-table overlap_table.jsonl --cover-level 0.8 0.7 1.0 --output out.csv
  python -m tools.rescore --predictions P.json --qrels qrels_span_c080.jsonl --output out.csv
Several pipeline folders (artifacts/tests/<config>/ with results/ and mteb_task/):
  python -m tools.rescore --config-dir DIR [DIR ...] --cover-level 0.8 0.7 --qrels-file qrels_lexical.jsonl --output out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib.span_labels import cover_tag, label_from_rows, parse_cover_level  # noqa: E402

COLUMNS = [
    "config", "label_set", "cover_level", "n_queries", "n_empty_questions",
    "ndcg_at_10", "recall_at_10", "geom_at_10", "geom_at_10_query_mean",
    "labels_source", "predictions_file",
]
STORED_COLUMNS = ["stored_ndcg_at_10", "stored_recall_at_10", "abs_diff_ndcg_at_10", "abs_diff_recall_at_10"]

_SPAN_FILE = re.compile(r"^qrels_span_c(\d{3})(?:p(\d+))?\.jsonl$")


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def _read_jsonl(path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_predictions(path) -> Dict[str, Dict[str, float]]:
    """{query id: {chunk id: score}}, each query sorted by score (descending, ties by chunk id descending)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(q): dict(sorted(((str(d), float(s)) for d, s in docs.items()), key=lambda kv: (kv[1], kv[0]), reverse=True))
            for q, docs in raw.items()}


def load_qrels(path) -> Dict[str, Dict[str, int]]:
    """{query id: {chunk id: 0/1}} from a qrels jsonl (reads ``relevance`` only; labels must be 0 or 1)."""
    out: Dict[str, Dict[str, int]] = {}
    for r in _read_jsonl(path):
        rel = r.get("relevance")
        if rel is None or float(rel) not in (0.0, 1.0):
            raise ValueError(f"{path}: relevance must be 0 or 1 (column 'score' is never a label), got {rel!r}")
        out.setdefault(str(r["query_id"]), {})[str(r["doc_id"])] = int(float(rel))
    return out


def load_overlap_table(path) -> Dict[str, List[Dict[str, Any]]]:
    rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in _read_jsonl(path):
        rows[str(r["query_id"])].append(r)
    return dict(rows)


def qrels_from_overlap_table(table: Dict[str, List[Dict[str, Any]]], query_ids: Iterable[str], c: Any) -> Dict[str, Dict[str, int]]:
    """Labels for cover level c from the overlap table. A query without a relevant chunk gets no positive entry (empty dict)."""
    frac = parse_cover_level(c)
    out: Dict[str, Dict[str, int]] = {}
    for q in query_ids:
        res = label_from_rows(table.get(q, []), frac)
        out[q] = {cid: 1 for cid in res["ids"]}
    return out


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def per_query_scores(predictions: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]):
    """({query: nDCG@10}, {query: Recall@10}, {query: positive labels}) for every query of ``qrels`` (0 for a query without a positive label)."""
    import pytrec_eval

    positive = {q: {d: 1 for d, r in rels.items() if r > 0} for q, rels in qrels.items()}
    positive = {q: rels for q, rels in positive.items() if rels}
    per_ndcg: Dict[str, float] = {q: 0.0 for q in qrels}
    per_recall: Dict[str, float] = {q: 0.0 for q in qrels}
    if positive:
        run = {q: dict(sorted(predictions[q].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)) for q in positive}
        res = pytrec_eval.RelevanceEvaluator(positive, {"ndcg_cut.10", "recall.10"}).evaluate(run)
        for q, m in res.items():
            per_ndcg[q], per_recall[q] = m["ndcg_cut_10"], m["recall_10"]
    return per_ndcg, per_recall, positive


def score(predictions: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """nDCG@10 and Recall@10 averaged over all queries of ``qrels``; a query without a positive label scores 0.

    ``qrels`` must contain every query (queries with only relevance-0 rows or an empty dict are 'empty questions').
    Predictions must cover the same queries; a mismatch is an error, never silently ignored.
    """
    if set(predictions) != set(qrels):
        only_p, only_q = sorted(set(predictions) - set(qrels))[:5], sorted(set(qrels) - set(predictions))[:5]
        raise ValueError(f"predictions and labels cover different queries (only in predictions: {only_p}..., only in labels: {only_q}...)")
    per_ndcg, per_recall, positive = per_query_scores(predictions, qrels)
    n = len(qrels)
    ndcg = sum(per_ndcg.values()) / n
    recall = sum(per_recall.values()) / n
    geom_q = sum(math.sqrt(per_ndcg[q] * per_recall[q]) for q in qrels) / n
    return {
        "n_queries": n,
        "n_empty_questions": n - len(positive),
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "geom_at_10": math.sqrt(ndcg * recall),
        "geom_at_10_query_mean": geom_q,
    }


def label_set_of(qrels_path) -> tuple:
    """(label set, cover level or '') from a qrels file name: qrels_span_c080.jsonl -> ('span', '0.8')."""
    name = Path(qrels_path).name
    m = _SPAN_FILE.match(name)
    if m:
        whole, frac = m.group(1), m.group(2) or ""
        percent = Fraction(Decimal(f"{int(whole)}.{frac}" if frac else str(int(whole))))
        return "span", _fmt_c(parse_cover_level(percent / 100))
    if name == "qrels_lexical.jsonl":
        return "lexical", ""
    return Path(name).stem, ""


def _fmt_c(frac) -> str:
    s = format(float(frac), "g")
    return s


def rescore_config(
    config: str,
    predictions_file,
    qrels_files: Sequence = (),
    overlap_table: Optional[str] = None,
    cover_levels: Sequence = (),
) -> List[Dict[str, Any]]:
    """All rows (one per label set / c) for one configuration."""
    preds = load_predictions(predictions_file)
    rows: List[Dict[str, Any]] = []
    for q in qrels_files:
        label_set, c = label_set_of(q)
        m = score(preds, load_qrels(q))
        rows.append({"config": config, "label_set": label_set, "cover_level": c, **m,
                     "labels_source": str(q), "predictions_file": str(predictions_file)})
    if cover_levels:
        if not overlap_table:
            raise ValueError("--cover-level needs the overlap table (--overlap-table or <config-dir>/mteb_task/overlap_table.jsonl)")
        table = load_overlap_table(overlap_table)
        seen = set()
        for c in cover_levels:
            frac = parse_cover_level(c)  # raises unless 0 < c <= 1
            if cover_tag(frac) in seen:
                continue
            seen.add(cover_tag(frac))
            m = score(preds, qrels_from_overlap_table(table, preds, frac))
            rows.append({"config": config, "label_set": "span", "cover_level": _fmt_c(frac), **m,
                         "labels_source": f"{overlap_table} @ c={_fmt_c(frac)}", "predictions_file": str(predictions_file)})
    return rows


def find_predictions(config_dir) -> Path:
    found = sorted(Path(config_dir, "results").rglob("*_predictions.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"{config_dir}: expected exactly one *_predictions.json under results/, found {len(found)}")
    return found[0]


def stored_metrics(config_dir) -> Optional[Dict[str, float]]:
    """nDCG@10 / Recall@10 that MTEB stored for the run (results/**/ChemQuest.json), if present."""
    for p in sorted(Path(config_dir, "results").rglob("ChemQuest.json")):
        scores = json.loads(p.read_text())["scores"]["test"][0]
        return {"ndcg_at_10": scores["ndcg_at_10"], "recall_at_10": scores["recall_at_10"]}
    return None


# --------------------------------------------------------------------------------------
# cover-level sensitivity of a whole report: every run (dense, BM25, hybrid, rerank), several c, paper-level intervals
# --------------------------------------------------------------------------------------
SENSITIVITY_COLUMNS = ["model", "kind", "chunker", "size", "overlap", "cover_level", "n_empty_questions", "ndcg_at_10", "ndcg_ci_low", "ndcg_ci_high",
                       "recall_at_10", "recall_ci_low", "recall_ci_high", "geom_at_10", "geom_ci_low", "geom_ci_high", "resamples"]


def sensitivity_rows(tests_root, qa_jsonl, cover_levels: Sequence, resamples: int, seed: int) -> List[Dict[str, Any]]:
    """One row per run and cover level: nDCG@10, Recall@10 and Geom@10 of the saved ranking under the labels that the span rule gives at that c (derived
    from the overlap table of the run's prepared folder: no re-chunking, no re-embedding), with paper-level bootstrap intervals (the draws are the
    ones of ``tools/stats.py``). Runs without a saved ranking or overlap table are skipped. Nothing is written into the run folders."""
    import numpy as np
    from tools import stats

    fracs = []
    for c in cover_levels:
        f = parse_cover_level(c)
        if cover_tag(f) not in {cover_tag(x) for x in fracs}:
            fracs.append(f)
    tests_root = Path(tests_root)
    tables: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    labels: Dict[tuple, Dict[str, Dict[str, int]]] = {}
    rows: List[Dict[str, Any]] = []
    groups = None
    for d in sorted(p for p in tests_root.iterdir() if p.is_dir() and (p / "run_info.json").is_file()):
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        preds = sorted((d / "results").glob("*/*_predictions.json"))
        table_path = Path(info["prepared_dir"]) / "mteb_task" / "overlap_table.jsonl"
        if len(preds) != 1 or not table_path.is_file():
            continue
        kind = "hybrid" if info.get("hybrid") else "rerank" if info.get("rerank") else "bm25" if str(info["model"]).startswith("bm25") else "dense"
        pred = load_predictions(preds[0])
        qids = sorted(pred, key=lambda x: (len(x), x))
        if groups is None:
            groups = stats.load_groups(qa_jsonl, qids)
        if str(table_path) not in tables:
            tables[str(table_path)] = load_overlap_table(table_path)
        for frac in fracs:
            key = (str(table_path), cover_tag(frac))
            if key not in labels:
                labels[key] = qrels_from_overlap_table(tables[str(table_path)], pred, frac)
            pn, pr, positive = per_query_scores(pred, labels[key])
            nd, rc = np.array([pn[q] for q in qids]), np.array([pr[q] for q in qids])
            ci = stats.bootstrap_ci(nd, rc, groups, resamples, seed)
            rows.append({"model": info["model"], "kind": kind, "chunker": info["chunker"], "size": info["size"], "overlap": info["overlap"],
                         "cover_level": _fmt_c(frac), "n_empty_questions": len(qids) - len(positive),
                         "ndcg_at_10": ci["ndcg_at_10"]["estimate"], "ndcg_ci_low": ci["ndcg_at_10"]["ci_low"], "ndcg_ci_high": ci["ndcg_at_10"]["ci_high"],
                         "recall_at_10": ci["recall_at_10"]["estimate"], "recall_ci_low": ci["recall_at_10"]["ci_low"], "recall_ci_high": ci["recall_at_10"]["ci_high"],
                         "geom_at_10": ci["geom_at_10"]["estimate"], "geom_ci_low": ci["geom_at_10"]["ci_low"], "geom_ci_high": ci["geom_at_10"]["ci_high"],
                         "resamples": resamples})
    return sorted(rows, key=lambda r: (r["chunker"], r["size"], r["overlap"], r["kind"], r["model"], float(r["cover_level"])))


def write_sensitivity(path, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SENSITIVITY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Rescore saved predictions with a qrels file or with the overlap table and cover levels.")
    ap.add_argument("--config-dir", nargs="+", help="pipeline folders (results/ with predictions, mteb_task/ with the labels)")
    ap.add_argument("--qrels-file", nargs="+", default=[], help="with --config-dir: qrels file names inside mteb_task/ (e.g. qrels_lexical.jsonl)")
    ap.add_argument("--predictions", help="one configuration: predictions file (<task>_predictions.json)")
    ap.add_argument("--qrels", nargs="+", default=[], help="one configuration: qrels file(s)")
    ap.add_argument("--overlap-table", help="one configuration: overlap_table.jsonl (with --cover-level)")
    ap.add_argument("--cover-level", nargs="+", default=[], metavar="C", help="cover levels in (0, 1]; labels are derived from the overlap table")
    ap.add_argument("--name", help="one configuration: name for the CSV (default: folder of the predictions)")
    ap.add_argument("--with-stored", action="store_true", help="with --config-dir: add the metrics MTEB stored for the run and the differences")
    ap.add_argument("--output", required=True, help="CSV file")
    ap.add_argument("--sensitivity", metavar="TESTS_ROOT", help="every run below <artifacts>/tests (all kinds): nDCG/Recall/Geom@10 at each --cover-level "
                    "(default params.analysis.cover_levels) with paper-level intervals; needs --qa; ignores the other selection options")
    ap.add_argument("--qa", help="with --sensitivity: qa.jsonl (paper of each question)")
    ap.add_argument("--resamples", type=int, help="with --sensitivity (default params.report.resamples)")
    ap.add_argument("--seed", type=int, help="with --sensitivity (default params.seed)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sensitivity:
        from pipeline_lib import config_params
        from tools import stats
        if not args.qa:
            raise SystemExit("ERROR: --sensitivity needs --qa")
        levels = args.cover_level or config_params.section("analysis").get("cover_levels") or []
        rows = sensitivity_rows(args.sensitivity, args.qa, levels, args.resamples or stats.DEFAULT_RESAMPLES,
                                args.seed if args.seed is not None else stats.DEFAULT_SEED)
        write_sensitivity(args.output, rows)
        print(f"wrote {len(rows)} rows ({len(levels)} cover levels) -> {args.output}")
        return 0
    rows: List[Dict[str, Any]] = []
    try:
        if args.config_dir:
            if not (args.qrels_file or args.cover_level):
                raise ValueError("give --qrels-file and/or --cover-level")
            for d in args.config_dir:
                d = Path(d)
                got = rescore_config(
                    d.name, find_predictions(d),
                    qrels_files=[d / "mteb_task" / n for n in args.qrels_file],
                    overlap_table=str(d / "mteb_task" / "overlap_table.jsonl") if args.cover_level else None,
                    cover_levels=args.cover_level,
                )
                if args.with_stored:
                    st = stored_metrics(d)
                    for r in got:
                        r["stored_ndcg_at_10"] = st["ndcg_at_10"] if st else ""
                        r["stored_recall_at_10"] = st["recall_at_10"] if st else ""
                        r["abs_diff_ndcg_at_10"] = abs(r["ndcg_at_10"] - st["ndcg_at_10"]) if st else ""
                        r["abs_diff_recall_at_10"] = abs(r["recall_at_10"] - st["recall_at_10"]) if st else ""
                rows.extend(got)
        else:
            if not args.predictions or not (args.qrels or args.cover_level):
                raise ValueError("give --predictions and --qrels, or --predictions, --overlap-table and --cover-level (or --config-dir)")
            name = args.name or Path(args.predictions).resolve().parents[2].name
            rows = rescore_config(name, args.predictions, qrels_files=args.qrels,
                                  overlap_table=args.overlap_table, cover_levels=args.cover_level)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"ERROR: {e}")
    columns = COLUMNS + (STORED_COLUMNS if args.with_stored else [])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

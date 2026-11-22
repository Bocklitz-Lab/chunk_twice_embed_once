"""All retrieval metrics from a saved ranking, for ANY qrels file (no embedding needed).

Per query: nDCG, MAP, Recall, Precision, MRR at k = 1, 10, 100 (configurable) and Geom@10 = sqrt(nDCG@10 * Recall@10).
Aggregates are means over ALL queries of the label set (a question without a relevant chunk scores 0 and stays in).
``Geom@10`` of the aggregate is sqrt(mean nDCG@10 * mean Recall@10), the definition of the paper (Phase 1 screening);
``geom_at_10_query_mean`` (mean of the per-query values) is reported next to it.

nDCG, MAP, Recall and Precision come from pytrec_eval exactly as MTEB calls it; MRR@k is MTEB's definition
(reciprocal rank of the first relevant chunk in the top k). ``cross_check`` compares the result with the aggregate
metrics MTEB stored in ``ChemQuest.json`` (MTEB rounds to 5 decimals).

Rankings are read from a TREC run (``qid Q0 docid rank score tag``, optionally .gz), a JSONL run
(``{"query_id", "doc_id", "rank", "score"}``) or MTEB's ``*_predictions.json`` ({qid: {docid: score}}).
Ties are broken like trec_eval: score descending, then document id descending.

  python -m tools.metrics --run run_top100.trec.gz --qrels mteb_task/qrels_span_c090.jsonl --out per_query_metrics.csv
  python -m tools.metrics --run ... --qrels A.jsonl B.jsonl --queries mteb_task/queries.jsonl --out-dir out/
  python -m tools.metrics --run ... --qrels ... --cross-check results/.../ChemQuest.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_KS = (1, 10, 100)
METRICS = ("ndcg", "map", "recall", "precision", "mrr")
Run = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]


# --------------------------------------------------------------------------------------
# reading and writing rankings
# --------------------------------------------------------------------------------------
def _open(path, mode="rt"):
    p = str(path)
    if p.endswith(".gz") and "w" in mode:
        # mtime=0 and no file name in the header: the same ranking gives the same bytes (repeat-run check)
        return io.TextIOWrapper(gzip.GzipFile(filename="", mode="wb", fileobj=open(p, "wb"), mtime=0), encoding="utf-8")
    return gzip.open(p, mode, encoding="utf-8") if p.endswith(".gz") else open(p, mode, encoding="utf-8")


def load_run(path) -> Run:
    """{query id: {chunk id: score}} from a TREC run, JSONL run or MTEB predictions JSON."""
    name = str(path)
    base = name[:-3] if name.endswith(".gz") else name
    run: Run = {}
    if base.endswith(".json"):
        with _open(path) as f:
            raw = json.load(f)
        return {str(q): {str(d): float(s) for d, s in docs.items()} for q, docs in raw.items()}
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if base.endswith(".jsonl"):
                r = json.loads(line)
                run.setdefault(str(r["query_id"]), {})[str(r["doc_id"])] = float(r["score"])
            else:
                qid, _q0, did, _rank, score, *_ = line.split()
                run.setdefault(qid, {})[did] = float(score)
    return run


def ranked(docs: Dict[str, float], depth: Optional[int] = None) -> List[tuple]:
    """[(doc id, score)] best first; ties like trec_eval (document id descending)."""
    out = sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return out[:depth] if depth else out


def write_trec_run(run: Run, path, tag: str = "run", depth: int = 100) -> int:
    """Gzip-able TREC run: ``qid Q0 docid rank score tag``, top ``depth`` per query. Returns the number of lines."""
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "wt") as f:
        for qid in sorted(run, key=lambda x: (len(x), x)):
            for rank, (did, score) in enumerate(ranked(run[qid], depth), start=1):
                f.write(f"{qid} Q0 {did} {rank} {score:.8f} {tag}\n")
                n += 1
    return n


def load_qrels(path) -> Qrels:
    """{query id: {chunk id: relevance}} from a qrels jsonl (``relevance`` column only; 0 or 1)."""
    out: Qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rel = r.get("relevance")
            if rel is None or float(rel) not in (0.0, 1.0):
                raise ValueError(f"{path}: relevance must be 0 or 1 (the 'score' column is never a label), got {rel!r}")
            out.setdefault(str(r["query_id"]), {})[str(r["doc_id"])] = int(float(rel))
    return out


def load_query_ids(path) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.append(str(r.get("id", r.get("_id", r.get("query_id")))))
    return ids


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def per_query_metrics(run: Run, qrels: Qrels, ks: Sequence[int] = DEFAULT_KS,
                      query_ids: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
    """{query id: {metric_at_k: value, ..., 'geom_at_10': value}} for every query of ``query_ids``
    (default: the queries of ``qrels``). A query without a positive label scores 0 on everything;
    a query that is missing from the run is an error."""
    import pytrec_eval

    qids = [str(q) for q in (query_ids if query_ids is not None else qrels)]
    missing = [q for q in qids if q not in run]
    if missing:
        raise ValueError(f"the ranking has no entry for {len(missing)} queries (first: {missing[:3]})")
    ks = sorted(set(int(k) for k in ks))
    out = {q: {f"{m}_at_{k}": 0.0 for m in METRICS for k in ks} for q in qids}
    positive = {q: {d: 1 for d, r in qrels.get(q, {}).items() if r > 0} for q in qids}
    positive = {q: rels for q, rels in positive.items() if rels}
    if positive:
        measures = {f"ndcg_cut.{','.join(map(str, ks))}", f"map_cut.{','.join(map(str, ks))}",
                    f"recall.{','.join(map(str, ks))}", f"P.{','.join(map(str, ks))}"}
        res = pytrec_eval.RelevanceEvaluator(positive, measures).evaluate({q: run[q] for q in positive})
        for q, m in res.items():
            for k in ks:
                out[q][f"ndcg_at_{k}"] = m[f"ndcg_cut_{k}"]
                out[q][f"map_at_{k}"] = m[f"map_cut_{k}"]
                out[q][f"recall_at_{k}"] = m[f"recall_{k}"]
                out[q][f"precision_at_{k}"] = m[f"P_{k}"]
        kmax = max(ks)
        for q, rels in positive.items():
            top = [d for d, _ in ranked(run[q], kmax)]
            for k in ks:
                rr = 0.0
                for rank, d in enumerate(top[:k], start=1):
                    if d in rels:
                        rr = 1.0 / rank
                        break
                out[q][f"mrr_at_{k}"] = rr
    for q in qids:
        out[q]["geom_at_10"] = math.sqrt(out[q].get("ndcg_at_10", 0.0) * out[q].get("recall_at_10", 0.0)) if 10 in ks else float("nan")
    out_n_pos = {q: len(positive.get(q, {})) for q in qids}
    for q in qids:
        out[q]["n_relevant"] = out_n_pos[q]
    return out


def aggregate(pq: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    n = len(pq)
    cols = [c for c in next(iter(pq.values())) if c not in ("n_relevant",)]
    agg: Dict[str, Any] = {c: sum(v[c] for v in pq.values()) / n for c in cols if c != "geom_at_10"}
    if "ndcg_at_10" in agg and "recall_at_10" in agg:
        agg["geom_at_10"] = math.sqrt(agg["ndcg_at_10"] * agg["recall_at_10"])
        agg["geom_at_10_query_mean"] = sum(v["geom_at_10"] for v in pq.values()) / n
    agg["n_queries"] = n
    agg["n_empty_questions"] = sum(1 for v in pq.values() if v["n_relevant"] == 0)
    return agg


def write_per_query_csv(path, pq: Dict[str, Dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = list(next(iter(pq.values())))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + cols)
        for q in sorted(pq, key=lambda x: (len(x), x)):
            w.writerow([q] + [f"{pq[q][c]:.10f}" if isinstance(pq[q][c], float) else pq[q][c] for c in cols])


def load_per_query_csv(path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.pop("query_id")
            out[q] = {k: float(v) for k, v in row.items()}
    return out


# --------------------------------------------------------------------------------------
# cross-check against MTEB
# --------------------------------------------------------------------------------------
def mteb_stored_scores(chemquest_json) -> Dict[str, float]:
    d = json.loads(Path(chemquest_json).read_text())
    return d["scores"]["test"][0]


def cross_check(run: Run, qrels: Qrels, chemquest_json, query_ids: Optional[Iterable[str]] = None,
                max_k: int = 100, tol: float = 6e-6) -> Dict[str, Any]:
    """Compare our aggregates with the metrics MTEB stored (nDCG, MAP, Recall, Precision, MRR at k <= max_k).

    MTEB rounds to 5 decimals, so differences up to 5e-6 are rounding; ``tol`` allows a little more.
    ``ok`` is False if any metric differs more than ``tol``."""
    stored = mteb_stored_scores(chemquest_json)
    ks = sorted({int(key.rsplit("_at_", 1)[1]) for key in stored if key.startswith("ndcg_at_")
                 and int(key.rsplit("_at_", 1)[1]) <= max_k})
    agg = aggregate(per_query_metrics(run, qrels, ks, query_ids))
    diffs = {}
    for m in METRICS:
        for k in ks:
            key = f"{m}_at_{k}"
            if key in stored and key in agg:
                diffs[key] = abs(agg[key] - float(stored[key]))
    worst = max(diffs, key=diffs.get) if diffs else None
    return {"ok": bool(diffs) and max(diffs.values()) <= tol, "n_metrics": len(diffs), "ks": ks,
            "max_abs_diff": diffs[worst] if worst else None, "worst_metric": worst, "tolerance": tol}


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="All metrics from a saved ranking and any qrels file.")
    ap.add_argument("--run", required=True, help="TREC run (.gz ok), JSONL run or MTEB *_predictions.json")
    ap.add_argument("--qrels", nargs="+", required=True, help="one or more qrels jsonl files")
    ap.add_argument("--queries", help="queries.jsonl: the universe of queries (default: the queries of each qrels file)")
    ap.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    ap.add_argument("--out", help="per-query CSV (single qrels file)")
    ap.add_argument("--out-dir", help="folder for per_query_metrics__<qrels name>.csv and summary.json (several qrels files)")
    ap.add_argument("--cross-check", help="MTEB ChemQuest.json of the same run: compare the aggregates (first qrels file)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run = load_run(args.run)
    qids = load_query_ids(args.queries) if args.queries else None
    summary: Dict[str, Any] = {}
    for qf in args.qrels:
        qrels = load_qrels(qf)
        pq = per_query_metrics(run, qrels, args.ks, qids)
        agg = aggregate(pq)
        summary[Path(qf).name] = agg
        if args.out and len(args.qrels) == 1:
            write_per_query_csv(args.out, pq)
        if args.out_dir:
            write_per_query_csv(Path(args.out_dir) / f"per_query_metrics__{Path(qf).stem}.csv", pq)
        print(f"{Path(qf).name}: n={agg['n_queries']} empty={agg['n_empty_questions']} "
              f"nDCG@10={agg.get('ndcg_at_10', float('nan')):.5f} Recall@10={agg.get('recall_at_10', float('nan')):.5f} "
              f"Geom@10={agg.get('geom_at_10', float('nan')):.5f}")
    if args.cross_check:
        cc = cross_check(run, load_qrels(args.qrels[0]), args.cross_check, qids)
        summary["cross_check"] = cc
        print("cross-check vs MTEB:", cc)
        if not cc["ok"]:
            return 1
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

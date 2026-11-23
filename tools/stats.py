"""Statistics over the per-query files (per_query_metrics*.csv) and the label files.

  ci        bootstrap 95% CI of Geom@10, nDCG@10 and Recall@10 (1000 resamples): over QUERIES and over PAPERS.
  compare   paired comparison of two runs (same queries): paired bootstrap of the difference (CI and p-value) and the
            Wilcoxon signed-rank test, for Geom@10, nDCG@10 and Recall@10. With several runs: all pairs or each run
            against ``--baseline``, Holm-corrected within each (family, level, metric) and test.
  labels    lexical vs span qrels of one prepared configuration: % of questions with the same top relevant chunk and
            the Jaccard overlap of the relevant sets.
  rankings  Kendall's tau (tau-b) between the configuration rankings under the lexical and the span labels, from the
            run table of tools/collect_runs.py: over all runs, within each model, and within each configuration.

Geom@10 of a sample is sqrt(mean nDCG@10 * mean Recall@10) (the definition of the paper), recomputed in every resample.
The 952 questions come from 149 papers (1-10 per paper): questions of one paper are not independent, so the paper-level
bootstrap (resample papers, keep all their questions) is the more careful interval; both are reported.
Bootstrap p-value (two-sided): 2 * (1 + min(#{d* <= 0}, #{d* >= 0})) / (B + 1), capped at 1. All randomness uses --seed.

  python -m tools.stats ci --per-query per_query_metrics.csv --qa qa.jsonl --out ci.csv
  python -m tools.stats compare --runs A=pqA.csv B=pqB.csv C=pqC.csv --baseline A --qa qa.jsonl --out cmp.csv
  python -m tools.stats labels --prepared artifacts/prepared/fixed_token_c128_o0__abcd1234
  python -m tools.stats rankings --runs-csv artifacts/runs_overview.csv
"""
from __future__ import annotations

import argparse
import csv
import functools
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params  # noqa: E402
from tools import metrics as M  # noqa: E402

# defaults come from configs/main_config.yaml (params.seed, params.report.resamples): nothing is hard-coded here
_PARAMS = config_params.load_params()
DEFAULT_SEED = int(_PARAMS.get("seed", 20261121))
DEFAULT_RESAMPLES = int((_PARAMS.get("report") or {}).get("resamples", 10000))
METRICS = ("geom_at_10", "ndcg_at_10", "recall_at_10")


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------
def _stat(name: str, ndcg: np.ndarray, recall: np.ndarray, per_q_geom: Optional[np.ndarray] = None):
    """The statistic of ``name`` for sample means ndcg / recall (arrays: one value per resample, or scalars)."""
    if name == "ndcg_at_10":
        return ndcg
    if name == "recall_at_10":
        return recall
    return np.sqrt(ndcg * recall)


def load_columns(per_query_csv, query_order: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    pq = M.load_per_query_csv(per_query_csv)
    qids = list(query_order) if query_order is not None else sorted(pq, key=lambda x: (len(x), x))
    return {"qids": np.array(qids), "ndcg_at_10": np.array([pq[q]["ndcg_at_10"] for q in qids]),
            "recall_at_10": np.array([pq[q]["recall_at_10"] for q in qids])}


def load_groups(qa_jsonl, qids: Sequence[str]) -> np.ndarray:
    """Paper (corpus_id) of each query id; query id = row index of qa.jsonl."""
    docs = [str(json.loads(l)["corpus_id"]) for l in open(qa_jsonl, encoding="utf-8") if l.strip()]
    return np.array([docs[int(q)] for q in qids])


def _resample_indices(n_units: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_units, size=(B, n_units))


@functools.lru_cache(maxsize=4)
def resample_indices(n_units: int, B: int, seed: int) -> np.ndarray:
    """The (B, n_units) resampled unit numbers of ``default_rng(seed)``. Memoised: every run of a configuration is resampled with the SAME
    draws (they depend only on n_units, B and seed), so a comparison of two runs is paired and the draws are made once, not once per run.
    The array must not be modified."""
    return _resample_indices(n_units, B, np.random.default_rng(seed))


def sample_stats(ndcg: np.ndarray, recall: np.ndarray, groups: Optional[np.ndarray], idx: np.ndarray):
    """Means of nDCG@10 / Recall@10 in every resample. ``idx``: resampled queries (B, n) or, with ``groups``, resampled
    group numbers (B, n_groups)."""
    if groups is None:
        return ndcg[idx].mean(axis=1), recall[idx].mean(axis=1)
    _, inv = np.unique(groups, return_inverse=True)
    ng = inv.max() + 1
    cnt = np.bincount(inv, minlength=ng).astype(float)
    sn = np.bincount(inv, weights=ndcg, minlength=ng)
    sr = np.bincount(inv, weights=recall, minlength=ng)
    c = cnt[idx].sum(axis=1)
    return sn[idx].sum(axis=1) / c, sr[idx].sum(axis=1) / c


def _n_units(ndcg, groups):
    return len(ndcg) if groups is None else len(np.unique(groups))


def resampled_means(ndcg: np.ndarray, recall: np.ndarray, groups: Optional[np.ndarray], B: int, seed: int):
    """(mean nDCG@10, mean Recall@10) of one run in each of the B resamples (queries, or papers when ``groups`` is given)."""
    return sample_stats(ndcg, recall, groups, resample_indices(_n_units(ndcg, groups), B, seed))


def ci_from_means(ndcg_mean: float, recall_mean: float, bn: np.ndarray, br: np.ndarray, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """{metric: {estimate, ci_low, ci_high}} from the point means and the resampled means (percentile interval)."""
    out = {}
    for m in METRICS:
        est = float(_stat(m, ndcg_mean, recall_mean))
        dist = _stat(m, bn, br)
        out[m] = {"estimate": est, "ci_low": float(np.quantile(dist, alpha / 2)), "ci_high": float(np.quantile(dist, 1 - alpha / 2))}
    return out


def paired_from_means(a_means, b_means, an, ar, bn, br, B: int, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Paired bootstrap of A - B from the point means ``(ndcg, recall)`` and the resampled means of the two runs (same resamples)."""
    out = {}
    for m in METRICS:
        d = _stat(m, an, ar) - _stat(m, bn, br)
        est = float(_stat(m, *a_means) - _stat(m, *b_means))
        le, ge = int((d <= 0).sum()), int((d >= 0).sum())
        out[m] = {"diff": est, "ci_low": float(np.quantile(d, alpha / 2)), "ci_high": float(np.quantile(d, 1 - alpha / 2)),
                  "p_bootstrap": min(1.0, 2 * (1 + min(le, ge)) / (B + 1))}
    return out


def bootstrap_ci(ndcg: np.ndarray, recall: np.ndarray, groups: Optional[np.ndarray] = None,
                 B: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """{metric: {estimate, ci_low, ci_high}} (percentile interval). ``groups``: resample these clusters (papers)."""
    bn, br = resampled_means(ndcg, recall, groups, B, seed)
    return ci_from_means(ndcg.mean(), recall.mean(), bn, br, alpha)


def paired_bootstrap(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], groups: Optional[np.ndarray] = None,
                     B: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Paired bootstrap of (A - B): the SAME resampled queries (or papers) are used for both runs."""
    an, ar = resampled_means(a["ndcg_at_10"], a["recall_at_10"], groups, B, seed)
    bn, br = resampled_means(b["ndcg_at_10"], b["recall_at_10"], groups, B, seed)
    return paired_from_means((a["ndcg_at_10"].mean(), a["recall_at_10"].mean()), (b["ndcg_at_10"].mean(), b["recall_at_10"].mean()), an, ar, bn, br, B, alpha)


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = x - y
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# --------------------------------------------------------------------------------------
# labels: lexical vs span
# --------------------------------------------------------------------------------------
def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def label_agreement(task_dir, span_qrels: str = "qrels_span_c090.jsonl", lexical_qrels: str = "qrels_lexical.jsonl") -> Dict[str, Any]:
    """Agreement of the lexical and the span labels of one configuration.

    top relevant chunk: lexical = the relevant chunk with the highest lexical ``score`` (ties: lower chunk index);
    span = the relevant chunk with the largest overlap with the evidence span (ties: lower chunk index).
    ``jaccard`` is averaged over the questions where at least one label set has a relevant chunk; ``same_top`` over the
    questions where both have one. Questions with no relevant chunk in either set are counted separately."""
    task_dir = Path(task_dir)
    qids = M.load_query_ids(task_dir / "queries.jsonl")
    chunk_index = {}
    for r in _read_jsonl(task_dir / "corpus.jsonl"):
        chunk_index[str(r["id"])] = int(str(r["id"]).rsplit(":", 1)[1])
    lex_rel: Dict[str, Dict[str, float]] = {}
    for r in _read_jsonl(task_dir / lexical_qrels):
        if float(r["relevance"]) > 0:
            lex_rel.setdefault(str(r["query_id"]), {})[str(r["doc_id"])] = float(r.get("score", 0.0))
    span_rel = {q: {d for d, v in rels.items() if v > 0} for q, rels in M.load_qrels(task_dir / span_qrels).items()}
    overlap: Dict[str, Dict[str, int]] = {}
    for r in _read_jsonl(task_dir / "overlap_table.jsonl"):
        overlap.setdefault(str(r["query_id"]), {})[str(r["chunk_id"])] = int(r["overlap_chars"])
    same = both = jac_n = 0
    jac_sum = 0.0
    only_lex = only_span = neither = 0
    for q in qids:
        L, S = set(lex_rel.get(q, {})), span_rel.get(q, set())
        if not L and not S:
            neither += 1
            continue
        jac_sum += len(L & S) / len(L | S)
        jac_n += 1
        if L and S:
            both += 1
            top_l = min(L, key=lambda d: (-lex_rel[q][d], chunk_index[d]))
            top_s = min(S, key=lambda d: (-overlap.get(q, {}).get(d, 0), chunk_index[d]))
            same += top_l == top_s
        elif L:
            only_lex += 1
        else:
            only_span += 1
    return {"n_questions": len(qids), "n_both_have_relevant": both, "pct_same_top_chunk": 100.0 * same / both if both else float("nan"),
            "mean_jaccard": jac_sum / jac_n if jac_n else float("nan"), "n_only_lexical_has_relevant": only_lex,
            "n_only_span_has_relevant": only_span, "n_neither": neither,
            "mean_relevant_lexical": sum(len(v) for v in lex_rel.values()) / len(qids),
            "mean_relevant_span": sum(len(v) for v in span_rel.values()) / len(qids)}


def kendall(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import kendalltau
    if len(x) < 3:
        return {"tau": float("nan"), "p": float("nan"), "n": len(x)}
    t = kendalltau(x, y, variant="b")
    return {"tau": float(t.statistic), "p": float(t.pvalue), "n": len(x)}


def ranking_agreement(runs_csv, metric: str = "geom_at_10") -> List[Dict[str, Any]]:
    """Kendall's tau-b between the rankings by the lexical and by the span label value of ``metric`` (run table of
    collect_runs). Rows: all runs; within each model (configurations ranked); within each configuration (models ranked)."""
    with open(runs_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get(f"span_{metric}") not in (None, "") and r.get(f"lexical_{metric}") not in (None, "")]
    out = []

    def add(scope, name, sub):
        k = kendall([float(r[f"span_{metric}"]) for r in sub], [float(r[f"lexical_{metric}"]) for r in sub])
        out.append({"scope": scope, "name": name, "metric": metric, **k})

    add("all_runs", "all", rows)
    for m in sorted({r["model"] for r in rows}):
        add("within_model", m, [r for r in rows if r["model"] == m])
    for cfg in sorted({(r["chunker"], r["size"], r["overlap"]) for r in rows}):
        add("within_configuration", "_".join(map(str, cfg)), [r for r in rows if (r["chunker"], r["size"], r["overlap"]) == cfg])
    return out


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def _write(path, rows):
    if not rows:
        return
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def _aligned(paths: Dict[str, str]):
    cols = {n: load_columns(p) for n, p in paths.items()}
    ref = next(iter(cols.values()))["qids"]
    for n, c in cols.items():
        if set(c["qids"]) != set(ref):
            raise SystemExit(f"ERROR: {n} does not cover the same queries as the other runs")
    order = list(ref)
    return {n: load_columns(paths[n], order) for n in paths}, order


def _cached_means(cache: Optional[Dict[tuple, tuple]], name: str, level: str, ndcg, recall, groups, B: int, seed: int):
    """Resampled means of run ``name`` at ``level`` from ``cache`` (filled on first use); the draws are the same for every run, so the
    values may be shared between the CI step and the paired tests of one configuration."""
    key = (name, level, B, seed)
    if cache is not None and key in cache:
        return cache[key]
    val = resampled_means(ndcg, recall, groups, B, seed)
    if cache is not None:
        cache[key] = val
    return val


def ci_rows(paths: Dict[str, str], qa: Optional[str] = None, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
            cache: Optional[Dict[tuple, tuple]] = None) -> List[Dict[str, Any]]:
    """Bootstrap CI rows (query level, and paper level when ``qa`` is given) for runs {name: per_query_metrics.csv}.
    ``cache``: dict shared with ``compare_rows`` so that each run is resampled once per level."""
    rows = []
    for name, path in paths.items():
        c = load_columns(path)
        levels = [("query", None)] + ([("paper", load_groups(qa, c["qids"]))] if qa else [])
        for level, groups in levels:
            bn, br = _cached_means(cache, name, level, c["ndcg_at_10"], c["recall_at_10"], groups, resamples, seed)
            for m, v in ci_from_means(c["ndcg_at_10"].mean(), c["recall_at_10"].mean(), bn, br).items():
                rows.append({"run": name, "level": level, "units": _n_units(c["ndcg_at_10"], groups), "metric": m, **v})
    return rows


def cmd_ci(args) -> int:
    paths = {}
    for spec in args.per_query:
        name, _, path = spec.rpartition("=")
        paths[name or Path(path).parent.name] = path
    rows = ci_rows(paths, args.qa, args.resamples, args.seed)
    if args.out:
        _write(args.out, rows)
    for r in rows:
        print(f"{r['run']:40s} {r['level']:5s} {r['metric']:13s} {r['estimate']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    return 0


def compare_rows(paths: Dict[str, str], pairs: Optional[Sequence[tuple]] = None, qa: Optional[str] = None,
                 resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED, families: Optional[Dict[tuple, str]] = None,
                 cache: Optional[Dict[tuple, tuple]] = None) -> List[Dict[str, Any]]:
    """Paired comparison rows for runs {name: per_query_metrics.csv}. ``pairs``: [(A, B)] (default: all pairs).
    Holm correction within (family, level, metric): ``families`` maps a pair to its family name (default: one family "all", i.e. one family
    per metric and level over the pairs given, as before). Every run is resampled once per level (``cache``), not once per pair."""
    cols, order = _aligned(paths)
    pairs = list(pairs) if pairs is not None else list(itertools.combinations(list(paths), 2))
    rows = []
    levels = [("query", None)] + ([("paper", load_groups(qa, order))] if qa else [])
    geom = {n: np.sqrt(c["ndcg_at_10"] * c["recall_at_10"]) for n, c in cols.items()}
    for level, groups in levels:
        means = {n: _cached_means(cache, n, level, c["ndcg_at_10"], c["recall_at_10"], groups, resamples, seed) for n, c in cols.items()
                 if any(n in p for p in pairs)}
        for a, b in pairs:
            pb = paired_from_means((cols[a]["ndcg_at_10"].mean(), cols[a]["recall_at_10"].mean()), (cols[b]["ndcg_at_10"].mean(), cols[b]["recall_at_10"].mean()),
                                   *means[a], *means[b], resamples)
            wil = {"ndcg_at_10": wilcoxon_p(cols[a]["ndcg_at_10"], cols[b]["ndcg_at_10"]),
                   "recall_at_10": wilcoxon_p(cols[a]["recall_at_10"], cols[b]["recall_at_10"]),
                   "geom_at_10": wilcoxon_p(geom[a], geom[b])}
            for m in METRICS:
                rows.append({"family": (families or {}).get((a, b), "all"), "level": level, "A": a, "B": b, "metric": m, **pb[m],
                             "p_wilcoxon_per_query": wil[m]})
    for fam in {r["family"] for r in rows}:
        for level in {r["level"] for r in rows}:
            for m in METRICS:
                sel = [r for r in rows if r["family"] == fam and r["level"] == level and r["metric"] == m]
                for key in ("p_bootstrap", "p_wilcoxon_per_query"):
                    for r, adj in zip(sel, holm([r[key] for r in sel])):
                        r[key + "_holm"] = adj
    return rows


def cmd_compare(args) -> int:
    paths = dict(s.split("=", 1) for s in args.runs)
    names = list(paths)
    pairs = [(args.baseline, n) for n in names if n != args.baseline] if args.baseline else None
    rows = compare_rows(paths, pairs, args.qa, args.resamples, args.seed)
    if args.out:
        _write(args.out, rows)
    for r in rows:
        print(f"{r['level']:5s} {r['A']} - {r['B']} {r['metric']:13s} diff {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"p_boot {r['p_bootstrap']:.4f} (Holm {r['p_bootstrap_holm']:.4f}) p_wilcoxon {r['p_wilcoxon_per_query']:.4g} (Holm {r['p_wilcoxon_per_query_holm']:.4g})")
    return 0


def cmd_labels(args) -> int:
    rows = []
    for d in args.prepared:
        man = json.loads((Path(d) / "manifest.json").read_text()) if (Path(d) / "manifest.json").exists() else {}
        rows.append({"config": Path(d).name, **label_agreement(Path(d) / "mteb_task", man.get("span_qrels_file", "qrels_span_c090.jsonl"))})
    if args.out:
        _write(args.out, rows)
    for r in rows:
        print(f"{r['config']}: same top chunk {r['pct_same_top_chunk']:.1f}% (n={r['n_both_have_relevant']}), Jaccard {r['mean_jaccard']:.3f}, "
              f"only lexical {r['n_only_lexical_has_relevant']}, only span {r['n_only_span_has_relevant']}, neither {r['n_neither']}")
    return 0


def cmd_rankings(args) -> int:
    rows = ranking_agreement(args.runs_csv, args.metric)
    if args.out:
        _write(args.out, rows)
    for r in rows:
        print(f"{r['scope']:22s} {r['name']:44s} tau-b {r['tau']:.3f} (p {r['p']:.3g}, n={r['n']})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ci", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--qa", help="qa.jsonl: adds the paper-level (cluster) bootstrap")
        p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--out")
        if name == "ci":
            p.add_argument("--per-query", nargs="+", required=True, help="per_query_metrics.csv or NAME=path ...")
        else:
            p.add_argument("--runs", nargs="+", required=True, help="NAME=per_query_metrics.csv (two or more)")
            p.add_argument("--baseline", help="compare every other run with this one (default: all pairs)")
    p = sub.add_parser("labels")
    p.add_argument("--prepared", nargs="+", required=True)
    p.add_argument("--out")
    p = sub.add_parser("rankings")
    p.add_argument("--runs-csv", required=True)
    p.add_argument("--metric", default="geom_at_10")
    p.add_argument("--out")
    args = ap.parse_args(argv)
    return {"ci": cmd_ci, "compare": cmd_compare, "labels": cmd_labels, "rankings": cmd_rankings}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

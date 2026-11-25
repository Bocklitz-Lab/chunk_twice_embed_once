"""Results by question type, by automatic question feature and by chemistry subfield (``make analysis``; Reviewer 2 items 23 and 24).

LABELS. Question type and chemistry subfields are the OFFICIAL per-question labels of the dataset (``question_metadata.jsonl`` next to ``qa.jsonl``, pinned
separately, loaded and checked by ``pipeline_lib/question_metadata.py``). Nothing is guessed from the question text and nothing comes from the paper-level
``fields_of_study``. If the file is missing the tool stops and says which command downloads it.

GROUPS OF QUESTIONS
  question_type:<type>        exclusive: every question is in exactly one of the four types (Conceptual, Mechanistic, Applied, Synthetic and Experimental).
  subfield:<name>             17 subfields; a question carries one or two. Two label rules, both written (column ``label_rule``):
                                multi_label  a question counts fully in every subfield it carries (groups overlap, sizes add up to more than 952);
                                fractional   a question has weight 1/len(subfields) in each of its subfields (weighted means and weighted bootstrap),
                                             so the weights of the groups add up to the number of questions (952) and every question counts once.
  auto_span_has_number:<flag>, auto_span_has_formula:<flag>   the same rules applied to the EVIDENCE SPAN ``full_text[start_index:end_index]`` (same tables, same tests)
  auto_has_number:<flag>, auto_has_formula:<flag>      the AUTOMATIC, rule-based features of ``pipeline_lib/question_features.py`` (not annotations, not
                              question types; ``feature_source = rule_based``), flagged / not_flagged, overall and inside each question type.

TABLES (one row per run, group and metric: nDCG@10, Recall@10, Geom@10; the 95% intervals are paper-level percentile intervals)
  breakdown_question_type.csv, breakdown_question_features.csv, breakdown_subfield.csv
        value and interval; the papers OF THE GROUP are resampled (``params.report.resamples`` resamples, ``params.seed``, the draws of ``tools/stats.py``,
        the same draws for every run). ``ci_width`` = ci_high - ci_low. Rows with ``row_kind = contrast`` hold the difference of two groups for one run
        (question type: all 6 pairs; feature: flagged - not_flagged, overall and inside each type) with a paired paper-level bootstrap over the papers of
        both groups (the same resampled papers for both), the bootstrap p-value and ``p_holm``.
  breakdown_*_paired.csv
        one row per comparison of two runs of the same configuration (dense - BM25, hybrid - dense, hybrid - BM25, rerank - dense, rerank - BM25,
        rerank - hybrid) and group: difference, interval, p_bootstrap (two-sided, 2(1+min(#<=0, #>=0))/(B+1)) and ``p_holm``. Contrast rows
        (``row_kind = contrast``) hold the INTERACTION: the difference between the two groups of that difference (does dense - BM25 differ between types?).
        Holm families (within one comparison, run, configuration and metric): question type: the 4 types, and the 6 type pairs; feature: all groups of
        that feature (flagged, not flagged, and the cells inside the types), and the contrasts of that feature; subfield: the 17 subfields of one label rule.
  A group (or cell) with fewer than ``params.analysis.breakdown.min_questions_for_test`` questions (30) gets its numbers and intervals, but no p-value, and
  no contrast is made with it.
  breakdown_groups.csv      per group: n_questions, n_papers, weight_sum, max_paper_share (largest share of the group's weight that one paper holds),
                            largest_paper, median_ci_width_geom (median over runs of the Geom@10 interval width)
  question_features.csv     per question: id, text, question_type (metadata), auto_has_number, auto_has_formula, the tokens that triggered each flag, feature_source

Small groups have wide intervals; do not read a difference between two groups from two overlapping intervals without a test.

  python -m tools.subfield_breakdown --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params, question_features as QF  # noqa: E402
from pipeline_lib.question_metadata import QUESTION_TYPES  # noqa: E402
from tools import analysis_data as AD  # noqa: E402
from tools import stats  # noqa: E402

FEATURES = ("auto_has_number", "auto_has_formula", "auto_span_has_number", "auto_span_has_formula")      # of the question text / of the evidence span
FEATURE_SOURCE = "rule_based"
MULTI, FRACTIONAL = "multi_label", "fractional"
METRIC_LIST = ("ndcg_at_10", "recall_at_10", "geom_at_10")


# ------------------------------------------------------------------------------------------------------------------
# groups
# ------------------------------------------------------------------------------------------------------------------
@dataclass
class Group:
    """A set of questions with a weight in [0, 1] each (0 = not in the group). ``holm`` names the family of its p-values."""
    name: str
    weights: np.ndarray
    holm: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_questions(self) -> int:
        return int((self.weights > 0).sum())


@dataclass
class Contrast:
    """Difference between two groups (a - b); ``holm`` names the family of its p-values."""
    name: str
    a: str
    b: str
    holm: str
    attrs: Dict[str, Any] = field(default_factory=dict)


def build_groups(types: Sequence[str], subfields: Sequence[Sequence[str]], features: Sequence[QF.QuestionFeatures], subfield_names: Sequence[str],
                 min_questions: int = 30, question_types: Sequence[str] = QUESTION_TYPES) -> Dict[str, Tuple[List[Group], List[Contrast]]]:
    """{"question_type": ..., "question_features": ..., "subfield": ...} -> (groups, contrasts). All inputs are aligned (one entry per question)."""
    n = len(types)
    t = np.asarray(types)
    # ---- question type: exclusive
    tgroups = [Group(f"question_type:{ty}", (t == ty).astype(float), "question_type", {"question_type": ty}) for ty in question_types]
    tcontrasts = [Contrast(f"question_type:{a} - question_type:{b}", f"question_type:{a}", f"question_type:{b}", "question_type:contrast",
                           {"question_type": f"{a} - {b}"}) for a, b in itertools.combinations(question_types, 2)]
    # ---- automatic features, overall and inside each type
    fgroups: List[Group] = []
    fcontrasts: List[Contrast] = []
    size = {ty: int((t == ty).sum()) for ty in question_types}
    for feat in [f for f in FEATURES if all(hasattr(x, f) for x in features)]:          # the span flags only when the caller gives them (QF.all_features)
        flag = np.array([getattr(f, feat) for f in features], dtype=bool)
        base = {"feature": feat, "feature_source": FEATURE_SOURCE}
        for label, mask in (("flagged", flag), ("not_flagged", ~flag)):
            fgroups.append(Group(f"{feat}:{label}", mask.astype(float), f"{feat}:group", {**base, "flag": label}))
        fcontrasts.append(Contrast(f"{feat}:flagged - {feat}:not_flagged", f"{feat}:flagged", f"{feat}:not_flagged", f"{feat}:contrast", {**base, "flag": "flagged - not_flagged"}))
        for ty in question_types:
            cell = {}
            for label, mask in (("flagged", flag), ("not_flagged", ~flag)):
                w = ((t == ty) & mask).astype(float)
                cell[label] = Group(f"{feat}:{label}|question_type:{ty}", w, f"{feat}:group", {**base, "flag": label, "question_type": ty})
                fgroups.append(cell[label])
            if cell["flagged"].n_questions >= min_questions and cell["not_flagged"].n_questions >= min_questions:
                fcontrasts.append(Contrast(f"{feat}:flagged|question_type:{ty} - {feat}:not_flagged|question_type:{ty}", cell["flagged"].name, cell["not_flagged"].name,
                                           f"{feat}:contrast", {**base, "flag": "flagged - not_flagged", "question_type": ty}))
    # ---- subfields: multi-label and fractional
    sgroups: List[Group] = []
    for rule in (MULTI, FRACTIONAL):
        for name in subfield_names:
            w = np.zeros(n)
            for i, subs in enumerate(subfields):
                if name in subs:
                    w[i] = 1.0 if rule == MULTI else 1.0 / len(subs)
            sgroups.append(Group(f"subfield:{name}", w, f"subfield:{rule}", {"label_rule": rule}))
    return {"question_type": (tgroups, tcontrasts), "question_features": (fgroups, fcontrasts), "subfield": (sgroups, [])}


# ------------------------------------------------------------------------------------------------------------------
# resampling
# ------------------------------------------------------------------------------------------------------------------
class Resampler:
    """Paper-level bootstrap of the weighted means of one or two groups of questions.

    The papers that have a question of positive weight in any of the groups are resampled (the same resampled papers for all groups and all runs;
    with one group these are the papers of the group). The weighted mean of a run in a group, in a resample, is
    sum_drawn-papers (weighted metric sum) / sum_drawn-papers (weight): with weights 0 / 1 the mean over the questions of the group in the drawn papers."""

    def __init__(self, weights: Dict[str, np.ndarray], papers: np.ndarray, resamples: int, seed: int):
        any_w = np.any([w > 0 for w in weights.values()], axis=0)
        self.q = np.where(any_w)[0]
        uniq, inv = np.unique(papers[self.q], return_inverse=True)
        self.G, self.B = len(uniq), int(resamples)
        self.n_papers = self.G
        self.onehot = np.zeros((len(self.q), self.G))
        self.onehot[np.arange(len(self.q)), inv] = 1.0
        self.w = {k: np.asarray(v, dtype=float)[self.q] for k, v in weights.items()}
        self.wg = {k: v @ self.onehot for k, v in self.w.items()}
        if self.G >= 2:
            idx = stats._resample_indices(self.G, self.B, np.random.default_rng(seed))
            self.counts = np.bincount((np.arange(self.B)[:, None] * self.G + idx).ravel(), minlength=self.B * self.G).reshape(self.B, self.G).astype(float)
            self.den = {k: self.counts @ v for k, v in self.wg.items()}

    def point(self, name: str, values: np.ndarray) -> np.ndarray:
        """(R, Q) -> (R,) weighted means over the questions of the group."""
        w = self.w[name]
        return (values[:, self.q] * w).sum(axis=1) / w.sum()

    def resampled(self, name: str, values: np.ndarray) -> np.ndarray:
        """(R, Q) -> (B, R) resampled weighted means (nan where a resample holds no question of the group)."""
        sums = (values[:, self.q] * self.w[name]) @ self.onehot
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.counts @ sums.T) / np.where(self.den[name] > 0, self.den[name], np.nan)[:, None]


def metric_values(rs: Resampler, name: str, ndcg: np.ndarray, recall: np.ndarray, metric: str, resampled: bool = True, block: int = 256) -> np.ndarray:
    """(B, R) resampled (or (R,) point) value of ``metric`` of every run in group ``name``. Geom@10 = sqrt(mean nDCG@10 x mean Recall@10)."""
    f = rs.resampled if resampled else rs.point
    R = ndcg.shape[0]
    out = np.empty((rs.B, R)) if resampled else np.empty(R)
    for a in range(0, R, block):
        b = min(a + block, R)
        if metric == "ndcg_at_10":
            v = f(name, ndcg[a:b])
        elif metric == "recall_at_10":
            v = f(name, recall[a:b])
        else:
            v = np.sqrt(f(name, ndcg[a:b]) * f(name, recall[a:b]))
        if resampled:
            out[:, a:b] = v
        else:
            out[a:b] = v
    return out


def boot_p(d: np.ndarray, B: int) -> np.ndarray:
    """Two-sided bootstrap p-value 2(1+min(#<=0, #>=0))/(B+1) of resampled differences (B, k)."""
    le, ge = (d <= 0).sum(axis=0), (d >= 0).sum(axis=0)
    return np.minimum(1.0, 2 * (1 + np.minimum(le, ge)) / (B + 1))


def run_pairs(runs: AD.RunSet) -> List[Tuple[int, int, str]]:
    """(run a, run b, comparison) for every comparison of two runs of the same configuration that ``make report`` makes between families (a - b):
    dense - bm25, hybrid - dense, hybrid - bm25, rerank - dense, rerank - bm25, rerank - hybrid. Dense - dense is not made."""
    pairs: List[Tuple[int, int, str]] = []
    cells: Dict[Tuple[str, int, int], List[int]] = {}
    for i, m in enumerate(runs.meta):
        cells.setdefault((m["chunker"], m["size"], m["overlap"]), []).append(i)
    for idx in cells.values():
        bm = [i for i in idx if runs.meta[i]["kind"] == "bm25"]
        bm25 = next((i for i in bm if runs.meta[i]["model"] == AD.BM25), bm[0] if bm else None)
        dense = {runs.meta[i]["model"]: i for i in idx if runs.meta[i]["kind"] == "dense"}
        hybrid = {runs.meta[i]["source_dense"]: i for i in idx if runs.meta[i]["kind"] == "hybrid"}
        for i in idx:
            m = runs.meta[i]
            if m["kind"] == "dense" and bm25 is not None:
                pairs.append((i, bm25, "dense_minus_bm25"))
            elif m["kind"] == "hybrid":
                if m["source_dense"] in dense:
                    pairs.append((i, dense[m["source_dense"]], "hybrid_minus_dense"))
                if bm25 is not None:
                    pairs.append((i, bm25, "hybrid_minus_bm25"))
            elif m["kind"] == "rerank":
                if m["source_dense"] in dense:
                    pairs.append((i, dense[m["source_dense"]], "rerank_minus_dense"))
                if bm25 is not None:
                    pairs.append((i, bm25, "rerank_minus_bm25"))
                if m["source_dense"] in hybrid:
                    pairs.append((i, hybrid[m["source_dense"]], "rerank_minus_hybrid"))
                if str(m["source_dense"]).startswith(AD.BM25):                     # rerank of the BM25 top-N against the rerank of each dense first stage
                    pairs += [(i, j, "rerank_bm25_minus_rerank_dense") for j in idx if runs.meta[j]["kind"] == "rerank"
                              and not str(runs.meta[j]["source_dense"]).startswith(AD.BM25) and runs.meta[j]["rerank_top_n"] == m["rerank_top_n"]]
    return pairs


def _meta(runs: AD.RunSet, i: int) -> Dict[str, Any]:
    m = runs.meta[i]
    return {"model": m["model"], "kind": m["kind"], "chunker": m["chunker"], "size": m["size"], "overlap": m["overlap"]}


def _weight_stats(g: Group, papers: np.ndarray) -> Dict[str, Any]:
    w = g.weights
    per_paper: Dict[Any, float] = {}
    for p, x in zip(papers[w > 0], w[w > 0]):
        per_paper[p] = per_paper.get(p, 0.0) + float(x)
    total = float(w.sum())
    top = max(per_paper.items(), key=lambda kv: (kv[1], str(kv[0]))) if per_paper else (None, 0.0)
    return {"n_questions": g.n_questions, "n_papers": len(per_paper), "weight_sum": total, "max_paper_share": (top[1] / total) if total else float("nan"),
            "largest_paper": top[0]}


def analyse(runs: AD.RunSet, groups: Sequence[Group], contrasts: Sequence[Contrast] = (), *, resamples: int, seed: int, min_questions: int = 30,
            metrics: Sequence[str] = METRIC_LIST) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(rows per run, group and metric; paired rows; group sizes). See the module docstring for the contents and the Holm families."""
    nd, rc = runs.metrics["ndcg_at_10"], runs.metrics["recall_at_10"]
    pairs = run_pairs(runs)
    pa = np.array([p[0] for p in pairs], dtype=int)
    pb = np.array([p[1] for p in pairs], dtype=int)
    by_name = {g.name: g for g in groups}
    rows: List[Dict[str, Any]] = []
    prow: List[Dict[str, Any]] = []
    sizes: List[Dict[str, Any]] = []
    nan = float("nan")

    def paired_rows(label: Dict[str, Any], diff_pt: np.ndarray, diff_res: np.ndarray, metric: str, testable: bool, holm: str, extra: Dict[str, Any]) -> None:
        if not pairs:
            return
        lo, hi = np.quantile(diff_res, [0.025, 0.975], axis=0)
        p = boot_p(diff_res, diff_res.shape[0]) if testable else np.full(len(pairs), nan)
        for k, (i, j, comp) in enumerate(pairs):
            prow.append({**label, **_meta(runs, i), "comparison": comp, "reference": runs.meta[j]["model"], "metric": metric, "difference": float(diff_pt[k]),
                         "ci_low": lo[k], "ci_high": hi[k], "p_bootstrap": p[k], "p_holm": nan, "n_questions": extra["n_questions"], "n_papers": extra["n_papers"],
                         "row_kind": extra["row_kind"], "_holm": (holm, comp, runs.meta[i]["key"], metric)})

    for g in groups:
        ws = _weight_stats(g, runs.groups)
        label = {"family": "", "group": g.name, **g.attrs}
        if g.n_questions == 0:                                           # e.g. no question of a type carries the flag: nothing to estimate
            sizes.append({**label, **ws, "median_ci_width_geom": nan, "row_kind": "group"})
            continue
        rs = Resampler({g.name: g.weights}, runs.groups, resamples, seed)
        usable = rs.G >= 2
        testable = usable and g.n_questions >= min_questions
        geom_widths = []
        for metric in metrics:
            pt = metric_values(rs, g.name, nd, rc, metric, resampled=False)
            res = metric_values(rs, g.name, nd, rc, metric) if usable else None
            lo, hi = np.quantile(res, [0.025, 0.975], axis=0) if usable else (np.full(len(runs), nan), np.full(len(runs), nan))
            if metric == "geom_at_10":
                geom_widths = hi - lo
            for i in range(len(runs)):
                rows.append({**label, **_meta(runs, i), "metric": metric, "value": pt[i], "ci_low": lo[i], "ci_high": hi[i], "ci_width": hi[i] - lo[i],
                             "n_questions": ws["n_questions"], "n_papers": ws["n_papers"], "weight_sum": ws["weight_sum"], "row_kind": "group"})
            if usable and pairs:
                paired_rows(label, pt[pa] - pt[pb], res[:, pa] - res[:, pb], metric, testable, g.holm, {**ws, "row_kind": "group"})
        sizes.append({**label, **ws, "median_ci_width_geom": float(np.nanmedian(geom_widths)) if np.isfinite(geom_widths).any() else nan, "row_kind": "group"})

    for c in contrasts:
        ga, gb = by_name[c.a], by_name[c.b]
        rs = Resampler({c.a: ga.weights, c.b: gb.weights}, runs.groups, resamples, seed)
        ok = rs.G >= 2 and ga.n_questions >= min_questions and gb.n_questions >= min_questions
        if not ok:
            continue
        info = {"n_questions": ga.n_questions + gb.n_questions, "n_papers": rs.n_papers, "row_kind": "contrast"}
        label = {"family": "", "group": c.name, **c.attrs}
        for metric in metrics:
            pa_pt, pb_pt = (metric_values(rs, x, nd, rc, metric, resampled=False) for x in (c.a, c.b))
            ra, rb = (metric_values(rs, x, nd, rc, metric) for x in (c.a, c.b))
            d_pt, d_res = pa_pt - pb_pt, ra - rb
            lo, hi = np.quantile(d_res, [0.025, 0.975], axis=0)
            p = boot_p(d_res, rs.B)
            for i in range(len(runs)):
                rows.append({**label, **_meta(runs, i), "metric": metric, "value": d_pt[i], "ci_low": lo[i], "ci_high": hi[i], "ci_width": hi[i] - lo[i],
                             "p_bootstrap": p[i], "p_holm": nan, "n_questions": info["n_questions"], "n_papers": info["n_papers"], "weight_sum": nan,
                             "row_kind": "contrast", "_holm": (c.holm, "", runs.meta[i]["key"], metric)})
            if pairs:
                paired_rows(label, (pa_pt[pa] - pa_pt[pb]) - (pb_pt[pa] - pb_pt[pb]), (ra[:, pa] - ra[:, pb]) - (rb[:, pa] - rb[:, pb]), metric, True, c.holm, info)
        sizes.append({**label, "n_questions": info["n_questions"], "n_papers": rs.n_papers, "weight_sum": nan, "max_paper_share": nan, "largest_paper": "",
                      "median_ci_width_geom": nan, "row_kind": "contrast"})

    # Holm inside each family (one comparison, run, configuration and metric; the groups of the family)
    for table in (rows, prow):
        fam: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in table:
            if "_holm" in r and not np.isnan(r["p_bootstrap"]):
                fam.setdefault(r["_holm"], []).append(r)
        for rs_ in fam.values():
            for r, adj in zip(rs_, stats.holm([x["p_bootstrap"] for x in rs_])):
                r["p_holm"] = adj
        for r in table:
            r.pop("_holm", None)
    return rows, prow, sizes


# ------------------------------------------------------------------------------------------------------------------
# per-question table of the automatic features
# ------------------------------------------------------------------------------------------------------------------
def feature_table(qa_rows: Sequence[Dict[str, Any]], types: Sequence[str], spans: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """One row per question: ``question_features.csv`` (the tokens let a person check each flag); with ``spans`` also the flags of the evidence span."""
    out = []
    for i, (r, ty) in enumerate(zip(qa_rows, types)):
        f = QF.question_features(r["question"])
        row = {"query_id": i, "question": r["question"], "question_type": ty, "auto_has_number": int(f.auto_has_number), "auto_has_formula": int(f.auto_has_formula),
               "number_tokens": " | ".join(f.number_tokens), "formula_tokens": " | ".join(f.formula_tokens), "feature_source": FEATURE_SOURCE}
        if spans is not None:
            sf = QF.span_features(spans[i])
            row.update({"auto_span_has_number": int(sf.auto_span_has_number), "auto_span_has_formula": int(sf.auto_span_has_formula),
                        "span_number_tokens": " | ".join(sf.number_tokens), "span_formula_tokens": " | ".join(sf.formula_tokens)})
        out.append(row)
    return out


GROUP_COLS = ["family", "group", "label_rule", "feature", "feature_source", "flag", "question_type"]
RUN_COLS = ["model", "kind", "chunker", "size", "overlap"]
ROW_COLS = GROUP_COLS + RUN_COLS + ["metric", "value", "ci_low", "ci_high", "ci_width", "p_bootstrap", "p_holm", "n_questions", "n_papers", "weight_sum", "row_kind"]
PAIRED_COLS = GROUP_COLS + RUN_COLS + ["comparison", "reference", "metric", "difference", "ci_low", "ci_high", "p_bootstrap", "p_holm", "n_questions", "n_papers", "row_kind"]
SIZE_COLS = GROUP_COLS + ["n_questions", "n_papers", "weight_sum", "max_paper_share", "largest_paper", "median_ci_width_geom", "row_kind"]
OUTPUT_NAMES = {"question_type": "breakdown_question_type", "question_features": "breakdown_question_features", "subfield": "breakdown_subfield"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    ap.add_argument("--metadata", help="question_metadata.jsonl (default: next to qa.jsonl)")
    ap.add_argument("--kinds", nargs="+", help="only these kinds (default: all; BM25 is needed for the paired tables)")
    args = ap.parse_args(argv)
    resamples, seed = AD.resolve(args)
    min_q = int((config_params.section("analysis", args.main_config).get("breakdown") or {}).get("min_questions_for_test", 30))
    meta = AD.load_question_metadata(args.qa, args.metadata)                 # raises, with the download command, if missing or not matching qa.jsonl
    runs = AD.load_runs(args.roots, args.qa, kinds=args.kinds, label_set=args.label_set)
    qa_rows = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    idx = [int(q) for q in runs.qids]
    types = [meta.question_type[i] for i in idx]
    spans = QF.span_texts(args.qa, qa_rows)
    feats = [QF.all_features(qa_rows[i]["question"], spans[i]) for i in idx]
    families = build_groups(types, [meta.subfields[i] for i in idx], feats, meta.subfield_names, min_q)
    out = Path(args.out_dir)
    AD.write_rows(out / "question_features.csv", feature_table(qa_rows, list(meta.question_type), spans))
    all_sizes: List[Dict[str, Any]] = []
    for fam, (groups, contrasts) in families.items():
        rows, paired, sizes = analyse(runs, groups, contrasts, resamples=resamples, seed=seed, min_questions=min_q)
        for r in rows + paired + sizes:
            r["family"] = fam
        AD.write_rows(out / f"{OUTPUT_NAMES[fam]}.csv", rows, ROW_COLS)
        if paired:
            AD.write_rows(out / f"{OUTPUT_NAMES[fam]}_paired.csv", paired, PAIRED_COLS)
        all_sizes += sizes
    AD.write_rows(out / "breakdown_groups.csv", all_sizes, SIZE_COLS)
    print(f"{len(runs)} runs, {len(all_sizes)} groups and contrasts -> {out} ({resamples} paper-level resamples; labels: {meta.path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

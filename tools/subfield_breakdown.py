"""Results by chemistry subfield and by question type (``make analysis``; Reviewer 2 items 23 and 24).

GROUPS OF QUESTIONS (a question can belong to several groups)

  subfield:<name>   every subfield listed in ``fields_of_study`` of the question's paper (``metadata.jsonl``; the join uses
                    ``span_labels.norm_doc_id`` because ``qa.jsonl`` ids are zero-padded). A paper in several subfields counts in each of them; a paper
                    without a label is in ``subfield:unlabelled``. The groups therefore overlap and their sizes add up to more than 952.
  tag:<name>        rule-based tags of the QUESTION TEXT (regular expressions below; they are heuristics, tested on examples, not annotations):
      qword:what / which / how / why / other   first word of the question (exactly one of the five; "how many" is "how"; when / where / who / ... are "other")
      yes_no                                    the question starts with an auxiliary verb (is, are, was, were, do, does, did, can, could, will, would,
                                                should, has, have, had, may, might): a yes/no question
      number_or_unit                            contains a digit, or a unit symbol of UNIT_RE (nm, mol, eV, ppm, °C, %, kJ, MPa, ...)
      chemical_formula                          contains a token that parses completely into element symbols and is a formula: at least two
                                                element symbols (NaCl, CO2, TiO2, LiFePO4) or a symbol with a count (O2, H2): "NMR", "DNA", "PEG",
                                                "In", "As" are not (NMR and DNA contain letters that are no element, "In" is one symbol without a count)
    Every boolean tag also has a complement group ``tag:no_<name>``.

OUTPUT (one row per run and group)
  breakdown_subfield.csv / breakdown_tag.csv    mean nDCG@10, Recall@10, Geom@10 of the questions of the group, with a paper-level 95% bootstrap
                                                interval (the papers OF THE GROUP are resampled; all runs share the resamples), n_questions, n_papers
  breakdown_*_paired.csv                        dense run - BM25 run of the same configuration, per group: difference, its interval and bootstrap p
                                                (two-sided, 2(1+min(#<=0, #>=0))/(B+1)); ``p_holm`` corrects over the groups of one configuration and model
  breakdown_groups.csv                          questions and papers per group
Small groups have wide intervals (e.g. 25 questions in Earth, Space, and Environmental Chemistry); do not read a difference between two groups
from two overlapping intervals without a test.

  python -m tools.subfield_breakdown --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib.span_labels import norm_doc_id  # noqa: E402
from tools import analysis_data as AD  # noqa: E402
from tools import stats  # noqa: E402

UNLABELLED = "unlabelled"
ELEMENTS = frozenset("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
                     "In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu "
                     "Am Cm Bk Cf Es Fm Md No Lr".split())
# abbreviations that parse into element symbols but are not formulas in a question: UV (U, V), IR (I, R is no element, listed anyway), CV, PVC, PS, PP, PU, ICP
NOT_FORMULAS = frozenset({"UV", "IR", "CV", "PVC", "PS", "PP", "PU", "NIR", "ICP", "PCB", "BCS"})
YES_NO_WORDS = frozenset("is are was were do does did can could will would should has have had may might".split())
QWORDS = ("what", "which", "how", "why")
UNIT_RE = re.compile(r"(?:(?<![A-Za-z])(?:nm|µm|μm|mm|cm|mg|kg|mol|mmol|kJ|kcal|eV|mV|kHz|MHz|GHz|THz|MPa|GPa|kPa|bar|atm|ppm|ppb|mL|µL|μL|rpm|wt)(?![A-Za-z])|°C|°|%)")
_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])((?:[A-Z][a-z]?\d*)+)(?![A-Za-z0-9])")
_PART_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def is_formula(token: str) -> bool:
    """True if ``token`` is made of element symbols only and has at least two of them or one with a count (NaCl, CO2, O2; not NMR, In)."""
    if token in NOT_FORMULAS:
        return False
    parts = _PART_RE.findall(token)
    if "".join(a + b for a, b in parts) != token or any(a not in ELEMENTS for a, _ in parts):
        return False
    return len(parts) >= 2 or bool(parts[0][1])


def question_tags(question: str) -> Dict[str, Any]:
    """The tags of one question text (see the module docstring)."""
    q = question.strip()
    words = re.findall(r"[A-Za-z]+", q)
    first = words[0].lower() if words else ""
    return {"qword": first if first in QWORDS else "other", "yes_no": first in YES_NO_WORDS,
            "number_or_unit": bool(re.search(r"\d", q) or UNIT_RE.search(q)),
            "chemical_formula": any(is_formula(m.group(1)) for m in _TOKEN_RE.finditer(q))}


def question_groups(qa_rows: Sequence[Dict[str, Any]], metadata: Dict[str, List[str]]) -> Dict[str, Dict[str, np.ndarray]]:
    """{"subfield": {name: mask}, "tag": {name: mask}} over the questions (row = query id)."""
    n = len(qa_rows)
    sub: Dict[str, np.ndarray] = {}
    tag: Dict[str, np.ndarray] = {}

    def mark(d, name, i):
        d.setdefault(name, np.zeros(n, dtype=bool))[i] = True

    for i, r in enumerate(qa_rows):
        for f in metadata.get(norm_doc_id(r["corpus_id"])) or [UNLABELLED]:
            mark(sub, f"subfield:{f}", i)
        t = question_tags(r["question"])
        mark(tag, f"tag:qword:{t['qword']}", i)
        for b in ("yes_no", "number_or_unit", "chemical_formula"):
            mark(tag, f"tag:{b}" if t[b] else f"tag:no_{b}", i)
    return {"subfield": dict(sorted(sub.items())), "tag": dict(sorted(tag.items()))}


def read_fields(metadata_jsonl: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for line in open(metadata_jsonl, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[norm_doc_id(r["corpus_id"])] = list(r.get("fields_of_study") or [])
    return out


# ------------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------------
class GroupBootstrap:
    """Paper-level bootstrap inside one group of questions: the papers that have questions in the group are resampled (the same draws for every run)."""

    def __init__(self, mask: np.ndarray, papers: np.ndarray, resamples: int, seed: int):
        self.q = np.where(mask)[0]
        uniq, inv = np.unique(papers[self.q], return_inverse=True)
        self.G, self.B = len(uniq), int(resamples)
        self.n_questions, self.n_papers = len(self.q), self.G
        self.onehot = np.zeros((len(self.q), self.G))
        self.onehot[np.arange(len(self.q)), inv] = 1.0
        self.n_g = self.onehot.sum(axis=0)
        if self.G >= 2:
            idx = stats._resample_indices(self.G, self.B, np.random.default_rng(seed))
            self.counts = np.bincount((np.arange(self.B)[:, None] * self.G + idx).ravel(), minlength=self.B * self.G).reshape(self.B, self.G).astype(float)
            self.den = self.counts @ self.n_g

    def point(self, values: np.ndarray) -> np.ndarray:
        return values[:, self.q].mean(axis=1)

    def resampled(self, values: np.ndarray) -> np.ndarray:
        """(R, Q) -> (B, R) resampled group means."""
        return (self.counts @ (values[:, self.q] @ self.onehot).T) / self.den[:, None]


def metric_resamples(gb: GroupBootstrap, ndcg: np.ndarray, recall: np.ndarray, metric: str, block: int = 256) -> np.ndarray:
    """(B, R) resampled group value of ``metric`` for every run."""
    R = ndcg.shape[0]
    out = np.empty((gb.B, R))
    for a in range(0, R, block):
        b = min(a + block, R)
        if metric == "ndcg_at_10":
            out[:, a:b] = gb.resampled(ndcg[a:b])
        elif metric == "recall_at_10":
            out[:, a:b] = gb.resampled(recall[a:b])
        else:
            out[:, a:b] = np.sqrt(gb.resampled(ndcg[a:b]) * gb.resampled(recall[a:b]))
    return out


def point_value(gb: GroupBootstrap, ndcg: np.ndarray, recall: np.ndarray, metric: str) -> np.ndarray:
    n, r = gb.point(ndcg), gb.point(recall)
    return n if metric == "ndcg_at_10" else r if metric == "recall_at_10" else np.sqrt(n * r)


def bonferroni_holm(p: Sequence[float]) -> List[float]:
    return stats.holm(list(p))


def analyse(runs: AD.RunSet, groups: Dict[str, np.ndarray], *, resamples: int, seed: int, metrics: Sequence[str] = stats.METRICS
            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(rows per run and group, paired dense - BM25 rows, group sizes)."""
    nd, rc = runs.metrics["ndcg_at_10"], runs.metrics["recall_at_10"]
    pairs: List[Tuple[int, int]] = []
    bm_of = {(m["chunker"], m["size"], m["overlap"]): i for i, m in enumerate(runs.meta) if m["kind"] == "bm25"}
    for i, m in enumerate(runs.meta):
        j = bm_of.get((m["chunker"], m["size"], m["overlap"]))
        if m["kind"] == "dense" and j is not None:
            pairs.append((i, j))
    rows, prow, sizes = [], [], []
    for gname, mask in groups.items():
        gb = GroupBootstrap(mask, runs.groups, resamples, seed)
        sizes.append({"group": gname, "n_questions": gb.n_questions, "n_papers": gb.n_papers})
        for metric in metrics:
            pt = point_value(gb, nd, rc, metric)
            lo, hi = (np.quantile(metric_resamples(gb, nd, rc, metric), [0.025, 0.975], axis=0) if gb.G >= 2 else (np.full(len(runs), np.nan),) * 2)
            res = metric_resamples(gb, nd, rc, metric) if (pairs and gb.G >= 2) else None
            for i, m in enumerate(runs.meta):
                rows.append({"group": gname, "model": m["model"], "kind": m["kind"], "chunker": m["chunker"], "size": m["size"], "overlap": m["overlap"],
                             "metric": metric, "value": pt[i], "ci_low": lo[i], "ci_high": hi[i], "n_questions": gb.n_questions, "n_papers": gb.n_papers})
            if res is not None:
                di, bj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
                d = res[:, di] - res[:, bj]
                dlo, dhi = np.quantile(d, [0.025, 0.975], axis=0)
                le, ge = (d <= 0).sum(axis=0), (d >= 0).sum(axis=0)
                p_boot = np.minimum(1.0, 2 * (1 + np.minimum(le, ge)) / (gb.B + 1))
                for k, (i, j) in enumerate(pairs):
                    m = runs.meta[i]
                    prow.append({"group": gname, "model": m["model"], "chunker": m["chunker"], "size": m["size"], "overlap": m["overlap"], "metric": metric,
                                 "dense_minus_bm25": float(pt[i] - pt[j]), "ci_low": dlo[k], "ci_high": dhi[k], "p_bootstrap": p_boot[k],
                                 "n_questions": gb.n_questions, "n_papers": gb.n_papers})
    # Holm over the groups of one (configuration, model, metric)
    fam: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in prow:
        fam.setdefault((r["model"], r["chunker"], r["size"], r["overlap"], r["metric"]), []).append(r)
    for rs in fam.values():
        for r, adj in zip(rs, bonferroni_holm([x["p_bootstrap"] for x in rs])):
            r["p_holm"] = adj
    return rows, prow, sizes


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    ap.add_argument("--metadata", help="metadata.jsonl (default: next to qa.jsonl)")
    ap.add_argument("--kinds", nargs="+", help="only these kinds (default: all; BM25 is needed for the paired tables)")
    args = ap.parse_args(argv)
    resamples, seed = AD.resolve(args)
    runs = AD.load_runs(args.roots, args.qa, kinds=args.kinds, label_set=args.label_set)
    qa_rows = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    meta = read_fields(args.metadata or Path(args.qa).with_name("metadata.jsonl"))
    groups = question_groups([qa_rows[int(q)] for q in runs.qids], meta)
    out = Path(args.out_dir)
    all_sizes: List[Dict[str, Any]] = []
    for kind in ("subfield", "tag"):
        rows, paired, sizes = analyse(runs, groups[kind], resamples=resamples, seed=seed)
        AD.write_rows(out / f"breakdown_{kind}.csv", rows)
        if paired:
            AD.write_rows(out / f"breakdown_{kind}_paired.csv", paired)
        all_sizes += sizes
    AD.write_rows(out / "breakdown_groups.csv", all_sizes)
    print(f"{len(runs)} runs, {len(all_sizes)} groups -> {out} ({resamples} paper-level resamples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

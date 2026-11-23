"""Success and failure examples for the discussion (``make analysis``; Reviewer 2 item 25): concrete questions behind the averages.

NAMED CASE RULES (``params.analysis.examples``; a case is one question, a model and one or two configurations; thresholds in the config)
  fails_small_works_large    same model and strategy: at the small size no relevant chunk in the top 10 (Recall@10 = 0), at the large size the question works
                             (Recall@10 = 1 and nDCG@10 >= works_ndcg). Shows what a small chunk costs.
  dense_wins_vs_bm25         same configuration: nDCG@10 of the dense model - nDCG@10 of BM25 >= margin
  bm25_wins_vs_dense         the reverse (BM25 - dense >= margin)
  all_fail                   same configuration: no retriever (every dense model, BM25, hybrid and rerank run) has a relevant chunk in its top 10
  rerank_rescues_dense_miss  the dense run has Recall@10 = 0, the rerank run of that dense model has Recall@10 = 1 and nDCG@10 >= works_ndcg

For each rule up to ``n_per_rule`` cases are drawn at random (seeded; at most one case per question). Every case is exported with the question, the paper, the
evidence span (at most ``excerpt_chars`` characters), and for every retriever involved its first relevant rank and its top ``top_chunks`` chunks with rank,
label (relevant under the span labels, c = labeling.cover_level) and text. ``examples.csv`` (one row per case and retriever) and ``examples.md`` (readable).

  python -m tools.examples --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params  # noqa: E402
from pipeline_lib.span_labels import qrels_filename  # noqa: E402
from tools import analysis_data as AD  # noqa: E402
from tools import metrics as M  # noqa: E402

RULES = ("fails_small_works_large", "dense_wins_vs_bm25", "bm25_wins_vs_dense", "all_fail", "rerank_rescues_dense_miss")


def settings(main_config: Optional[str] = None) -> Dict[str, Any]:
    s = config_params.section("analysis", main_config).get("examples")
    if not s:
        raise ValueError("params.analysis.examples is missing in the main config")
    return s


# ------------------------------------------------------------------------------------------------------------------
# candidate cases: (rule, query index, [run indices involved])
# ------------------------------------------------------------------------------------------------------------------
def find_cases(runs: AD.RunSet, st: Dict[str, Any]) -> Dict[str, List[Tuple[int, List[int]]]]:
    """{rule: [(query index, [run index, ...])]} for every question that meets the rule (before sampling)."""
    nd, rc = runs.metrics["ndcg_at_10"], runs.metrics["recall_at_10"]
    works = float(st["works_ndcg"])
    margin = float(st["margin"])
    cases: Dict[str, List[Tuple[int, List[int]]]] = {r: [] for r in RULES}
    key = {(m["model"], m["chunker"], m["size"], m["overlap"]): i for i, m in enumerate(runs.meta)}
    cell = {}
    for i, m in enumerate(runs.meta):
        cell.setdefault((m["chunker"], m["size"], m["overlap"]), []).append(i)

    def ok(i: int) -> np.ndarray:
        return (rc[i] >= 1.0) & (nd[i] >= works)

    for small, large in st["size_pairs"]:
        for m in runs.meta:
            if m["kind"] != "dense" or m["size"] != small:
                continue
            j = key.get((m["model"], m["chunker"], large, m["overlap"]))
            if j is None:
                continue
            i = key[(m["model"], m["chunker"], small, m["overlap"])]
            for q in np.where((rc[i] == 0) & ok(j))[0]:
                cases["fails_small_works_large"].append((int(q), [i, j]))
    for c, idx in cell.items():
        bm = [i for i in idx if runs.meta[i]["kind"] == "bm25"]
        if bm:
            b = bm[0]
            for i in (i for i in idx if runs.meta[i]["kind"] == "dense"):
                for q in np.where(nd[i] - nd[b] >= margin)[0]:
                    cases["dense_wins_vs_bm25"].append((int(q), [i, b]))
                for q in np.where(nd[b] - nd[i] >= margin)[0]:
                    cases["bm25_wins_vs_dense"].append((int(q), [b, i]))
        if len(idx) > 1:
            for q in np.where((rc[idx] == 0).all(axis=0))[0]:
                cases["all_fail"].append((int(q), list(idx)))
        for r in (i for i in idx if runs.meta[i]["kind"] == "rerank"):
            d = key.get((runs.meta[r]["source_dense"], c[0], c[1], c[2]))
            if d is not None:
                for q in np.where((rc[d] == 0) & ok(r))[0]:
                    cases["rerank_rescues_dense_miss"].append((int(q), [d, r]))
    return cases


def sample_cases(cases: Dict[str, List[Tuple[int, List[int]]]], n: int, seed: int) -> Dict[str, List[Tuple[int, List[int]]]]:
    """Up to n cases per rule, at most one per question, in a seeded random order (the same seed gives the same cases)."""
    rng = random.Random(seed)
    out: Dict[str, List[Tuple[int, List[int]]]] = {}
    for rule in RULES:
        pool = sorted(cases.get(rule, []), key=lambda c: (c[0], c[1]))
        rng.shuffle(pool)
        seen, chosen = set(), []
        for q, runs_ in pool:
            if q not in seen:
                seen.add(q)
                chosen.append((q, runs_))
            if len(chosen) == n:
                break
        out[rule] = chosen
    return out


# ------------------------------------------------------------------------------------------------------------------
# export
# ------------------------------------------------------------------------------------------------------------------
class Material:
    """Rankings, labels and chunk texts of the runs, loaded on demand and kept."""

    def __init__(self, runs: AD.RunSet, cover_level: Any):
        self.runs, self.cover = runs, cover_level
        self._run: Dict[int, M.Run] = {}
        self._prep: Dict[str, Tuple[Dict[str, set], Dict[str, str]]] = {}

    def ranking(self, i: int) -> M.Run:
        if i not in self._run:
            self._run[i] = M.load_run(Path(self.runs.meta[i]["out_dir"]) / "run_top100.trec.gz")
        return self._run[i]

    def prepared(self, i: int) -> Tuple[Dict[str, set], Dict[str, str]]:
        d = str(self.runs.meta[i]["prepared_dir"])
        if d not in self._prep:
            rel: Dict[str, set] = {}
            for line in open(Path(d) / "mteb_task" / qrels_filename("span", self.cover), encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if float(r["relevance"]) > 0:
                        rel.setdefault(str(r["query_id"]), set()).add(str(r["doc_id"]))
            text = {}
            for line in open(Path(d) / "chunks" / "chunks.jsonl", encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    text[str(r["chunk_id"])] = r["text"]
            self._prep[d] = (rel, text)
        return self._prep[d]


def excerpt(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def case_rows(rule: str, case_no: int, q: int, run_ids: Sequence[int], mat: Material, qa_rows: Sequence[Dict[str, Any]], st: Dict[str, Any]) -> List[Dict[str, Any]]:
    qid = str(q)
    qa = qa_rows[q]
    rows = []
    for i in run_ids:
        m = mat.runs.meta[i]
        rel, text = mat.prepared(i)
        ranked = [d for d, _ in M.ranked(mat.ranking(i).get(qid, {}))]
        first = next((r for r, d in enumerate(ranked, start=1) if d in rel.get(qid, set())), None)
        row = {"rule": rule, "case": f"{rule}-{case_no}", "query_id": qid, "paper": qa["corpus_id"], "question": qa["question"],
               "span_excerpt": excerpt(qa["content"], int(st["excerpt_chars"])), "retriever": m["model"], "kind": m["kind"],
               "configuration": f"{m['chunker']} {m['size']}/{m['overlap']}", "first_relevant_rank": first if first is not None else "not in top 100",
               "ndcg_at_10": float(mat.runs.metrics["ndcg_at_10"][i][q]), "recall_at_10": float(mat.runs.metrics["recall_at_10"][i][q])}
        for k, d in enumerate(ranked[: int(st["top_chunks"])], start=1):
            row[f"top{k}"] = f"[{'RELEVANT' if d in rel.get(qid, set()) else 'not relevant'}] {d}: {excerpt(text.get(d, ''), int(st['chunk_chars']))}"
        rows.append(row)
    return rows


def to_markdown(rows: List[Dict[str, Any]], top_chunks: int) -> str:
    lines = ["# Success and failure examples", ""]
    seen = set()
    for r in rows:
        if r["case"] not in seen:
            seen.add(r["case"])
            lines += [f"## {r['case']}  (query {r['query_id']}, paper {r['paper']})", "", f"**Question:** {r['question']}", "", f"**Evidence span:** {r['span_excerpt']}", ""]
        lines += [f"**{r['retriever']}**, {r['configuration']}: first relevant rank {r['first_relevant_rank']}, nDCG@10 {r['ndcg_at_10']:.2f}, Recall@10 {r['recall_at_10']:.2f}", ""]
        lines += [f"{k}. {r[f'top{k}']}" for k in range(1, top_chunks + 1) if f"top{k}" in r] + [""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    ap.add_argument("--n-per-rule", type=int, help="default: params.analysis.examples.n_per_rule")
    args = ap.parse_args(argv)
    st = settings(args.main_config)
    d = AD.defaults(args.main_config)
    seed = args.seed if args.seed is not None else int(st.get("seed", d["seed"]))
    runs = AD.load_runs(args.roots, args.qa, label_set=args.label_set)
    qa_rows = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    chosen = sample_cases(find_cases(runs, st), args.n_per_rule or int(st["n_per_rule"]), seed)
    mat = Material(runs, d["cover_level"])
    rows: List[Dict[str, Any]] = []
    for rule, cs in chosen.items():
        for k, (q, ids) in enumerate(cs, start=1):
            rows += case_rows(rule, k, q, ids, mat, qa_rows, st)
    out = Path(args.out_dir)
    AD.write_rows(out / "examples.csv", rows)
    (out / "examples.md").write_text(to_markdown(rows, int(st["top_chunks"])), encoding="utf-8")
    print("cases per rule: " + ", ".join(f"{r} {len(c)}" for r, c in chosen.items()) + f" -> {out / 'examples.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

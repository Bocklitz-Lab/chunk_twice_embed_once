"""Fixed retrieved-token budget (``make analysis``, Reviewer 2 item 7): how much of the evidence span does a reader see when the retriever may
hand over only B tokens, whatever the chunk size?

Rank-based metrics at k = 10 compare 10 chunks of 128 tokens with 10 chunks of 512 tokens. Here the unit is the amount of text instead.
For every run (dense, BM25, hybrid, rerank) and every budget B (``params.analysis.token_budgets``, default 512 / 1024 / 2048 tokens of the
REFERENCE tokenizer ``params.chunking_tokenizer``, no special tokens), per question:

  1. walk the saved ranking (``run_top100.trec.gz``) from the top and hand over the chunks in order;
  2. count only text that was not handed over before: overlapping chunks (overlap > 0) of the same document are merged as the UNION of their
     character intervals, so an overlap region costs tokens once. Tokens of a piece = reference-tokenizer tokens of the document whose start
     lies in the piece (document-level tokenization with offsets: the same text always costs the same);
  3. the chunk (piece) that crosses the budget is CUT at its last token that still fits (the end offset of the k-th remaining token); nothing
     after that is handed over;
  4. span coverage = share of the evidence span's characters inside the handed-over text (exact integer arithmetic);
     hit = coverage >= c (``params.labeling.cover_level``, 0.90; the label rule's cover level, so a hit is "the span is as covered as the span labels
     demand of a relevant chunk"; ``--cover-level`` may give several).

If the saved ranking (top 100) is exhausted before B tokens, everything is handed over and ``budget_reached`` is false for that question.
Output ``token_budget.csv``: one row per run and budget with the mean span coverage and the hit rate, each with a paper-level bootstrap
95% interval (``tools/analysis_data.PaperBootstrap``: papers are resampled, all runs share the resamples). Per-question values with ``--per-query``.

  python -m tools.token_budget --roots artifacts --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params  # noqa: E402
from pipeline_lib.span_labels import norm_doc_id, parse_cover_level  # noqa: E402
from tools import analysis_data as AD  # noqa: E402
from tools import metrics as M  # noqa: E402

DEFAULT_BUDGETS = (512, 1024, 2048)
TokenOffsets = Tuple[List[int], List[int]]                 # (token start chars, token end chars) of one document


def budgets_from_config(main_config: Optional[str] = None) -> List[int]:
    return [int(b) for b in (config_params.section("analysis", main_config).get("token_budgets") or DEFAULT_BUDGETS)]


# ------------------------------------------------------------------------------------------------------------------
# tokens and intervals
# ------------------------------------------------------------------------------------------------------------------
def hf_token_offsets(text: str, tokenizer: Any) -> TokenOffsets:
    """Start and end character of every token of ``text`` (fast tokenizer, no special tokens, no truncation)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
    pairs = [(a, b) for a, b in enc["offset_mapping"] if b > a]
    return [a for a, _ in pairs], [b for _, b in pairs]


def load_documents(full_text_jsonl: Any, tokenize: Callable[[str], TokenOffsets]) -> Dict[str, TokenOffsets]:
    """{normalised document id: token offsets} for every document of full_text.jsonl."""
    out: Dict[str, TokenOffsets] = {}
    with open(full_text_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[norm_doc_id(r["corpus_id"])] = tokenize(r["text"])
    return out


def load_chunk_positions(chunks_jsonl: Any) -> Dict[str, Tuple[str, int, int]]:
    """{chunk id: (document, start, end)} of a prepared configuration."""
    pos: Dict[str, Tuple[str, int, int]] = {}
    with open(chunks_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                start = r["start_char"] if "start_char" in r else r["start"]
                end = r["end_char"] if "end_char" in r else r["end"]
                pos[str(r["chunk_id"])] = (norm_doc_id(r["source_doc_id"]), int(start), int(end))
    return pos


def new_pieces(retrieved: List[Tuple[int, int]], a: int, b: int) -> List[Tuple[int, int]]:
    """The parts of [a, b) not inside the sorted, disjoint intervals ``retrieved``."""
    pieces, cur = [], a
    for x, y in retrieved:
        if y <= cur:
            continue
        if x >= b:
            break
        if x > cur:
            pieces.append((cur, x))
        cur = max(cur, y)
        if cur >= b:
            break
    if cur < b:
        pieces.append((cur, b))
    return pieces


def add_interval(retrieved: List[Tuple[int, int]], a: int, b: int) -> None:
    """Insert [a, b) into the sorted disjoint interval list (merging touching or overlapping intervals)."""
    out: List[Tuple[int, int]] = []
    placed = False
    for x, y in retrieved:
        if y < a:
            out.append((x, y))
        elif x > b:
            if not placed:
                out.append((a, b)); placed = True
            out.append((x, y))
        else:
            a, b = min(a, x), max(b, y)
    if not placed:
        out.append((a, b))
    retrieved[:] = out


def walk_budgets(ranking: Sequence[str], chunk_pos: Dict[str, Tuple[str, int, int]], doc_tokens: Dict[str, TokenOffsets],
                 span: Tuple[str, int, int], budgets: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """For one question: {budget: {covered (span chars), span_len, tokens_used, n_chunks, reached}} (see the module docstring)."""
    doc_s, s, e = span
    span_len = e - s
    pending = sorted(budgets)
    out: Dict[int, Dict[str, Any]] = {}
    retrieved: Dict[str, List[Tuple[int, int]]] = {}
    used = covered = 0
    for n_chunk, cid in enumerate(ranking, start=1):
        if not pending:
            break
        doc, a, b = chunk_pos[cid]
        starts, ends = doc_tokens[doc]
        ivs = retrieved.setdefault(doc, [])
        for x, y in new_pieces(ivs, a, b):
            i0, i1 = bisect.bisect_left(starts, x), bisect.bisect_left(starts, y)
            n = i1 - i0
            while pending and used + n >= pending[0]:
                k = pending[0] - used                                       # tokens of this piece that still fit
                cut = min(ends[i0 + k - 1], y) if k > 0 else x
                cov = covered + (max(0, min(cut, e) - max(x, s)) if doc == doc_s else 0)
                out[pending[0]] = {"covered": cov, "span_len": span_len, "tokens_used": pending[0], "n_chunks": n_chunk, "reached": True}
                pending.pop(0)
            used += n
            if doc == doc_s:
                covered += max(0, min(y, e) - max(x, s))
            add_interval(ivs, x, y)
    for bud in pending:                                                     # ranking exhausted before the budget was filled
        out[bud] = {"covered": covered, "span_len": span_len, "tokens_used": used, "n_chunks": len(ranking), "reached": False}
    return out


# ------------------------------------------------------------------------------------------------------------------
# runs
# ------------------------------------------------------------------------------------------------------------------
def run_per_query(run_meta: Dict[str, Any], qids: Sequence[str], spans: Dict[str, Tuple[str, int, int]], doc_tokens: Dict[str, TokenOffsets],
                  budgets: Sequence[int], chunk_cache: Dict[str, Dict[str, Tuple[str, int, int]]]) -> Dict[int, List[Dict[str, Any]]]:
    """{budget: [per-question result in the order of ``qids``]} for one run."""
    prepared = Path(run_meta["prepared_dir"])
    key = str(prepared)
    if key not in chunk_cache:
        chunk_cache[key] = load_chunk_positions(prepared / "chunks" / "chunks.jsonl")
    pos = chunk_cache[key]
    run = M.load_run(Path(run_meta["out_dir"]) / "run_top100.trec.gz")
    res: Dict[int, List[Dict[str, Any]]] = {b: [] for b in budgets}
    for q in qids:
        ranking = [d for d, _ in M.ranked(run.get(q, {}))]
        r = walk_budgets(ranking, pos, doc_tokens, spans[q], budgets)
        for b in budgets:
            res[b].append(r[b])
    return res


def load_spans(qa_jsonl: Any, qids: Sequence[str]) -> Dict[str, Tuple[str, int, int]]:
    rows = [json.loads(l) for l in open(qa_jsonl, encoding="utf-8") if l.strip()]
    return {q: (norm_doc_id(rows[int(q)]["corpus_id"]), int(rows[int(q)]["start_index"]), int(rows[int(q)]["end_index"])) for q in qids}


def hit_array(results: List[Dict[str, Any]], cover: Fraction) -> np.ndarray:
    """1.0 where covered / span_len >= c, exact: covered * den >= num * span_len."""
    return np.array([1.0 if r["covered"] * cover.denominator >= cover.numerator * r["span_len"] else 0.0 for r in results])


def analyse(runs: AD.RunSet, qa_jsonl: Any, full_text_jsonl: Any, *, budgets: Sequence[int], cover_levels: Sequence[Any], resamples: int, seed: int,
            tokenize: Optional[Callable[[str], TokenOffsets]] = None, per_query_dir: Optional[Path] = None, verbose: bool = True) -> List[Dict[str, Any]]:
    """Rows of ``token_budget.csv``: one per run, budget and cover level."""
    if tokenize is None:
        from transformers import AutoTokenizer
        name, rev = config_params.reference_tokenizer()
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        tokenize = lambda text: hf_token_offsets(text, tok)           # noqa: E731
    doc_tokens = load_documents(full_text_jsonl, tokenize)
    spans = load_spans(qa_jsonl, runs.qids)
    covers = [parse_cover_level(c) for c in cover_levels]
    boot = AD.PaperBootstrap(runs.groups, resamples, seed)
    R, Q = len(runs), len(runs.qids)
    share = {b: np.zeros((R, Q)) for b in budgets}
    hits = {(b, c): np.zeros((R, Q)) for b in budgets for c in covers}
    tokens_used = {b: np.zeros(R) for b in budgets}
    reached = {b: np.zeros(R) for b in budgets}
    chunk_cache: Dict[str, Dict[str, Tuple[str, int, int]]] = {}
    for i, meta in enumerate(runs.meta):
        res = run_per_query(meta, runs.qids, spans, doc_tokens, budgets, chunk_cache)
        for b in budgets:
            share[b][i] = [r["covered"] / r["span_len"] if r["span_len"] > 0 else 0.0 for r in res[b]]
            for c in covers:
                hits[(b, c)][i] = hit_array(res[b], c)
            tokens_used[b][i] = np.mean([r["tokens_used"] for r in res[b]])
            reached[b][i] = np.mean([r["reached"] for r in res[b]])
        if per_query_dir is not None:
            _write_per_query(per_query_dir, meta, runs.qids, res)
        if verbose and (i + 1) % 100 == 0:
            print(f"  token budget: {i + 1}/{R} runs")
    rows: List[Dict[str, Any]] = []
    for b in budgets:
        s_res = boot.resample_means(share[b])
        s_lo, s_hi = boot.percentile_ci(s_res)
        for c in covers:
            h_res = boot.resample_means(hits[(b, c)])
            h_lo, h_hi = boot.percentile_ci(h_res)
            for i, meta in enumerate(runs.meta):
                rows.append({"model": meta["model"], "kind": meta["kind"], "chunker": meta["chunker"], "size": meta["size"], "overlap": meta["overlap"],
                             "budget_tokens": b, "cover_level": float(c), "mean_span_coverage": share[b][i].mean(), "span_coverage_ci_low": s_lo[i],
                             "span_coverage_ci_high": s_hi[i], "hit_rate": hits[(b, c)][i].mean(), "hit_ci_low": h_lo[i], "hit_ci_high": h_hi[i],
                             "mean_tokens_used": tokens_used[b][i], "share_budget_reached": reached[b][i], "resamples": resamples, "seed": seed})
    return rows


def _write_per_query(out_dir: Path, meta: Dict[str, Any], qids: Sequence[str], res: Dict[int, List[Dict[str, Any]]]) -> None:
    rows = []
    for b, lst in res.items():
        for q, r in zip(qids, lst):
            rows.append({"model": meta["model"], "chunker": meta["chunker"], "size": meta["size"], "overlap": meta["overlap"], "budget_tokens": b,
                         "query_id": q, "span_coverage": r["covered"] / r["span_len"] if r["span_len"] else 0.0, "tokens_used": r["tokens_used"],
                         "n_chunks": r["n_chunks"], "budget_reached": r["reached"]})
    AD.write_rows(out_dir / f"token_budget_per_query__{meta['key'].replace('/', '_').replace('|', '__')}.csv", rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    AD.add_common_args(ap)
    ap.add_argument("--full-text", help="full_text.jsonl (default: next to qa.jsonl)")
    ap.add_argument("--budgets", nargs="+", type=int, help="tokens (default: params.analysis.token_budgets)")
    ap.add_argument("--cover-level", nargs="+", help="hit threshold(s) (default: params.labeling.cover_level)")
    ap.add_argument("--kinds", nargs="+", help="only these kinds: dense bm25 hybrid rerank")
    ap.add_argument("--per-query", action="store_true", help="also write the per-question values of every run")
    args = ap.parse_args(argv)
    resamples, seed = AD.resolve(args)
    budgets = args.budgets or budgets_from_config(args.main_config)
    covers = args.cover_level or [AD.defaults(args.main_config)["cover_level"]]
    runs = AD.load_runs(args.roots, args.qa, kinds=args.kinds, label_set=args.label_set)
    full = args.full_text or str(Path(args.qa).with_name("full_text.jsonl"))
    out = Path(args.out_dir)
    rows = analyse(runs, args.qa, full, budgets=budgets, cover_levels=covers, resamples=resamples, seed=seed,
                   per_query_dir=out if args.per_query else None)
    AD.write_rows(out / "token_budget.csv", rows)
    print(f"{len(runs)} runs x {len(budgets)} budgets -> {out / 'token_budget.csv'} ({resamples} paper-level resamples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Coverage diagnostics (no embedding needed) and the chunking-independent span-coverage metric.

1. ``config``: one row per prepared configuration (chunks + qrels):
   n_chunks, mean chunk length in tokens (reference tokenizer) and characters, % of questions whose evidence span lies
   fully inside ONE chunk, mean number of relevant chunks per question for the lexical and the span qrels, and the
   number of questions with zero relevant chunks (for both). Also writes ``no_relevant_chunk.csv`` (the questions
   without a relevant chunk, per label set, with the reason).

2. ``span-coverage``: from a saved ranking, the fraction of the evidence-span characters covered by the UNION of the
   top-k retrieved chunks (k = 1, 5, 10). It does not use any relevance label, only the character positions of the
   evidence span and of the chunks (the overlap table of stage 5). Bigger chunks cover more by construction, so the
   mean number of retrieved characters and the share of retrieved characters that lie inside the span are reported
   next to it.

  python -m tools.coverage_diagnostics config --prepared artifacts/prepared/fixed_token_c128_o0__abcd1234 --out diag.csv
  python -m tools.coverage_diagnostics config --prepared-root artifacts/prepared --out coverage_diagnostics.csv
  python -m tools.coverage_diagnostics span-coverage --run run_top100.trec.gz --overlap-table overlap_table.jsonl \\
        --queries queries.jsonl --corpus corpus.jsonl --out span_coverage.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import metrics as M  # noqa: E402

COVERAGE_KS = (1, 5, 10)


def _read_jsonl(path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------------------
# 1. per-configuration diagnostics
# --------------------------------------------------------------------------------------
def _labels_summary(qrels_path, query_ids: Sequence[str]) -> Dict[str, Any]:
    qrels = M.load_qrels(qrels_path)
    n_rel = {q: sum(1 for r in qrels.get(q, {}).values() if r > 0) for q in query_ids}
    vals = list(n_rel.values())
    return {"mean_relevant": sum(vals) / len(vals), "max_relevant": max(vals),
            "n_zero_relevant": sum(1 for v in vals if v == 0), "zero_ids": [q for q, v in n_rel.items() if v == 0]}


def _token_lengths(texts: List[str], tokenizer_name: Optional[str], revision: Optional[str]) -> Optional[List[int]]:
    if not tokenizer_name:
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, revision=revision)
    out: List[int] = []
    for i in range(0, len(texts), 2000):
        out += [len(x) for x in tok(texts[i:i + 2000], add_special_tokens=False, truncation=False)["input_ids"]]
    return out


def diagnose_config(task_dir, chunks_jsonl, tokenizer_name: Optional[str] = None, tokenizer_revision: Optional[str] = None,
                    span_qrels: Optional[str] = None, lexical_qrels: str = "qrels_lexical.jsonl",
                    out_dir: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One diagnostics row for the configuration in ``task_dir`` (mteb_task folder). See the module docstring."""
    task_dir = Path(task_dir)
    qids = M.load_query_ids(task_dir / "queries.jsonl")
    chunks = list(_read_jsonl(chunks_jsonl))
    texts = [c["text"] for c in chunks]
    tok_len = _token_lengths(texts, tokenizer_name, tokenizer_revision)
    chars = [len(t) for t in texts]
    row: Dict[str, Any] = dict(extra or {})
    row.update({
        "n_questions": len(qids), "n_chunks": len(chunks), "n_docs": len({c["source_doc_id"] for c in chunks}),
        "mean_chunk_chars": sum(chars) / len(chars),
        "mean_chunk_tokens": (sum(tok_len) / len(tok_len)) if tok_len else "",
        "max_chunk_tokens": max(tok_len) if tok_len else "",
        "chunk_token_counter": f"{tokenizer_name}@{(tokenizer_revision or 'main')[:8]}" if tokenizer_name else "",
    })
    table = task_dir / "overlap_table.jsonl"
    inside = set()
    if table.exists():
        for r in _read_jsonl(table):
            if r["contains_span"]:
                inside.add(str(r["query_id"]))
        row["pct_span_inside_one_chunk"] = 100.0 * len(inside & set(qids)) / len(qids)
    else:
        row["pct_span_inside_one_chunk"] = ""
    reasons: List[Dict[str, str]] = []
    for label_set, fname in (("lexical", lexical_qrels), ("span", span_qrels)):
        if fname and (task_dir / fname).exists():
            s = _labels_summary(task_dir / fname, qids)
            row[f"mean_relevant_{label_set}"] = s["mean_relevant"]
            row[f"max_relevant_{label_set}"] = s["max_relevant"]
            row[f"n_zero_relevant_{label_set}"] = s["n_zero_relevant"]
            reasons += [{"label_set": label_set, "qrels_file": fname, "query_id": q} for q in s["zero_ids"]]
        else:
            row[f"mean_relevant_{label_set}"] = row[f"max_relevant_{label_set}"] = row[f"n_zero_relevant_{label_set}"] = ""
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "no_relevant_chunk.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label_set", "qrels_file", "query_id"])
            w.writeheader()
            w.writerows(reasons)
        write_rows(Path(out_dir) / "coverage_diagnostics.csv", [row])
    return row


def write_rows(path, rows: List[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


# --------------------------------------------------------------------------------------
# 2. span coverage from a ranking
# --------------------------------------------------------------------------------------
def load_overlaps(overlap_table):
    """{query id: {chunk id: (overlap_start, overlap_end)}} and the span length per query."""
    out: Dict[str, Dict[str, tuple]] = {}
    span_len: Dict[str, int] = {}
    for r in _read_jsonl(overlap_table):
        q = str(r["query_id"])
        out.setdefault(q, {})[str(r["chunk_id"])] = (int(r["overlap_start"]), int(r["overlap_end"]))
        span_len[q] = int(r["span_len"])
    return out, span_len


def _union_len(intervals: List[tuple]) -> int:
    total, cur_s, cur_e = 0, None, None
    for a, b in sorted(intervals):
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def span_coverage(run: M.Run, overlap_table, query_ids: Sequence[str], ks: Sequence[int] = COVERAGE_KS,
                  chunk_chars: Optional[Dict[str, int]] = None) -> Dict[str, Dict[str, float]]:
    """Per query: ``span_coverage_at_k`` = |span ∩ union(top-k chunks)| / |span| (0 for a question whose span no chunk
    touches, so the mean is over all questions). With ``chunk_chars``: ``retrieved_chars_at_k`` (sum of the lengths of
    the top-k chunks) and ``span_char_precision_at_k`` (covered span characters / retrieved characters)."""
    overlaps, span_len = load_overlaps(overlap_table)
    out: Dict[str, Dict[str, float]] = {}
    for q in query_ids:
        q = str(q)
        if q not in run:
            raise ValueError(f"the ranking has no entry for query {q!r}")
        top = [d for d, _ in M.ranked(run[q], max(ks))]
        rec: Dict[str, float] = {}
        for k in ks:
            ivs = [overlaps.get(q, {}).get(d) for d in top[:k]]
            covered = _union_len([iv for iv in ivs if iv])
            rec[f"span_coverage_at_{k}"] = covered / span_len[q] if q in span_len else 0.0
            if chunk_chars is not None:
                total = sum(chunk_chars.get(d, 0) for d in top[:k])
                rec[f"retrieved_chars_at_{k}"] = float(total)
                rec[f"span_char_precision_at_{k}"] = covered / total if total else 0.0
        out[q] = rec
    return out


def mean_of(rows: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    n = len(rows)
    cols = next(iter(rows.values())).keys()
    return {c: sum(r[c] for r in rows.values()) / n for c in cols}


def write_span_coverage_csv(path, rows: Dict[str, Dict[str, float]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = list(next(iter(rows.values())))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + cols)
        for q in sorted(rows, key=lambda x: (len(x), x)):
            w.writerow([q] + [f"{rows[q][c]:.8f}" for c in cols])


def corpus_chunk_chars(corpus_jsonl) -> Dict[str, int]:
    return {str(r["id"]): len(r["text"]) for r in _read_jsonl(corpus_jsonl)}


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config", help="diagnostics per prepared configuration")
    c.add_argument("--prepared", nargs="*", default=[], help="prepared folders (each with manifest.json)")
    c.add_argument("--prepared-root", help="folder holding prepared folders")
    c.add_argument("--out", required=True, help="CSV")
    s = sub.add_parser("span-coverage", help="span coverage@k from a saved ranking")
    s.add_argument("--run", required=True)
    s.add_argument("--overlap-table", required=True)
    s.add_argument("--queries", required=True)
    s.add_argument("--corpus", help="corpus.jsonl (adds retrieved characters and span-character precision)")
    s.add_argument("--ks", nargs="+", type=int, default=list(COVERAGE_KS))
    s.add_argument("--out", required=True, help="per-query CSV")
    args = ap.parse_args(argv)
    if args.cmd == "config":
        dirs = [Path(d) for d in args.prepared]
        if args.prepared_root:
            dirs += sorted(p.parent for p in Path(args.prepared_root).glob("*/manifest.json"))
        rows = []
        for d in dirs:
            m = json.loads((d / "manifest.json").read_text())
            rows.append(diagnose_config(d / "mteb_task", d / "chunks" / "chunks.jsonl", m.get("tokenizer"),
                                        m.get("tokenizer_revision"), m.get("span_qrels_file"),
                                        extra={"config": m.get("config_name", d.name)}))
        write_rows(args.out, rows)
        print(f"wrote {len(rows)} rows -> {args.out}")
        return 0
    run = M.load_run(args.run)
    qids = M.load_query_ids(args.queries)
    cc = corpus_chunk_chars(args.corpus) if args.corpus else None
    rows = span_coverage(run, args.overlap_table, qids, args.ks, cc)
    write_span_coverage_csv(args.out, rows)
    print({k: round(v, 4) for k, v in mean_of(rows).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 5, span labels: build the MTEB task (queries, corpus, qrels) from the evidence-span positions.

Replaces the lexical labels (BM25 + Jaccard + fuzzy, ``stages/make_mteb_task.py``) when
``labeling.method`` is ``span``. For every question the relevant chunks are chosen from the chunker's exact
positions and the span [start_index, end_index) of ``qa.jsonl`` by the rule in ``pipeline_lib/span_labels.py``
(a chunk holding >= c of the span; otherwise greedy by new characters; otherwise every overlapping chunk).

Inputs (config keys or command-line options):
  qa_jsonl_file, chunks_jsonl_file, full_text_jsonl_file (default: full_text.jsonl next to qa.jsonl), output_folder,
  labeling.cover_level (c; ``--cover-level`` takes one or more values and wins over the config).

Outputs in ``output_folder``:
  queries.jsonl                 id = row index in qa.jsonl, text = the question only
  corpus.jsonl                  same format as the lexical stage 5
  qrels_span_<c-tag>.jsonl      binary labels, one file per c (plus a relevance-0 row for a question without a relevant chunk)
  label_stats_<c-tag>.json      steps, labels per question, empty questions, c
  overlap_table.jsonl           one row per (question, chunk) pair that overlaps the span; independent of c

Fails (non-zero exit) if a chunk has no exact position, if ``full_text[start:end] != text`` for a chunk, or if
``full_text[start_index:end_index] != content`` for a question.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from fractions import Fraction
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib.dataset_pin import CHEMQUESTS_QA_SHA256, verify_qa_sha256  # noqa: E402
from pipeline_lib.span_labels import (  # noqa: E402
    cover_tag,
    label_from_rows,
    label_stats_filename,
    norm_doc_id,
    overlap_rows,
    parse_cover_level,
    qrels_filename,
)


class SpanQrelsError(ValueError):
    """The inputs do not allow exact span labels (missing or wrong chunk positions, wrong span text, ...)."""


# --------------------------------------------------------------------------------------
# loading and validation
# --------------------------------------------------------------------------------------
def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_full_text(path: str) -> Dict[str, str]:
    full_text: Dict[str, str] = {}
    for row in _read_jsonl(path):
        doc = norm_doc_id(row["corpus_id"])
        if doc in full_text:
            raise SpanQrelsError(f"{path}: document {doc!r} appears twice")
        full_text[doc] = row["text"]
    return full_text


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def chunk_title(doc: str, idx: int, embed_chunk_title: bool) -> str:
    """MTEB embeds ``title + " " + text``. False (default): empty title, the model sees the chunk text only."""
    return f"{doc} - Chunk {idx}" if embed_chunk_title else ""


def load_chunks(path: str, full_text: Dict[str, str], embed_chunk_title: bool = False):
    """Return (chunks_by_doc, corpus_rows). Every chunk must carry exact positions: full_text[start:end] == text."""
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    corpus: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in _read_jsonl(path):
        doc = norm_doc_id(row["source_doc_id"])
        cid = str(row["chunk_id"])
        idx = int(row["chunk_index"])
        text = row.get("text", "")
        start = row["start_char"] if "start_char" in row else row.get("start")
        end = row["end_char"] if "end_char" in row else row.get("end")
        if not _is_int(start) or not _is_int(end):
            raise SpanQrelsError(
                f"chunk {cid!r} (document {doc!r}) has no exact position (start={start!r}, end={end!r}); "
                "span labels need the chunker's native positions. Re-run stage 4."
            )
        if doc not in full_text:
            raise SpanQrelsError(f"chunk {cid!r}: document {doc!r} is not in the full text file")
        doc_text = full_text[doc]
        if not (0 <= start <= end <= len(doc_text)) or doc_text[start:end] != text:
            raise SpanQrelsError(
                f"chunk {cid!r} (document {doc!r}): full_text[{start}:{end}] does not equal the chunk text "
                f"(positions are wrong or the chunk is not a verbatim part of the document)."
            )
        if cid in seen_ids:
            raise SpanQrelsError(f"duplicate chunk_id {cid!r}")
        seen_ids.add(cid)
        by_doc.setdefault(doc, []).append(
            {"chunk_id": cid, "chunk_index": idx, "start": start, "end": end, "text": text}
        )
        corpus.append({"id": cid, "title": chunk_title(doc, idx, embed_chunk_title), "text": text})
    for doc, chunks in by_doc.items():
        chunks.sort(key=lambda x: x["chunk_index"])
        indices = [c["chunk_index"] for c in chunks]
        if len(set(indices)) != len(indices):
            raise SpanQrelsError(f"document {doc!r}: chunk_index is not unique")
    return by_doc, corpus


def load_qa(path: str, full_text: Dict[str, str]) -> List[Dict[str, Any]]:
    """One dict per record of qa.jsonl (id = row index). Every span must reproduce its content in the full text."""
    qa: List[Dict[str, Any]] = []
    for qid, row in enumerate(_read_jsonl(path)):
        doc = norm_doc_id(row["corpus_id"])
        question = (row.get("question") or "").strip()
        content = row.get("content", "")
        s, e = row.get("start_index"), row.get("end_index")
        if not _is_int(s) or not _is_int(e):
            raise SpanQrelsError(f"question {qid}: start_index/end_index must be integers, got {s!r}/{e!r}")
        if doc not in full_text:
            raise SpanQrelsError(f"question {qid}: document {doc!r} is not in the full text file")
        text = full_text[doc]
        if not (0 <= s <= e <= len(text)) or text[s:e] != content:
            raise SpanQrelsError(
                f"question {qid} (document {doc!r}): full_text[{s}:{e}] does not equal the span text 'content'. "
                "qa.jsonl does not belong to this full text (wrong dataset revision?)."
            )
        if not question:
            raise SpanQrelsError(f"question {qid}: the question text is empty")
        if content.strip() and content.strip() in question:
            raise SpanQrelsError(f"question {qid}: the query text contains the evidence span text")
        qa.append({"qid": qid, "doc": doc, "question": question, "start": s, "end": e})
    return qa


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------
def _write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _zero_row(qid: int, by_doc, doc: str, corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunks = by_doc.get(doc)
    cid = chunks[0]["chunk_id"] if chunks else corpus[0]["id"]
    return {"query_id": str(qid), "doc_id": cid, "relevance": 0}


def build_span_qrels(
    qa_jsonl_file: str,
    chunks_jsonl_file: str,
    output_folder: str,
    cover_levels: List[Any],
    full_text_jsonl_file: Optional[str] = None,
    qa_sha256: Optional[str] = None,
    extra_qrels_file: Optional[str] = None,
    verbose: bool = True,
    embed_chunk_title: bool = False,
) -> Dict[str, Any]:
    """Write queries, corpus, overlap table and, for each c, qrels + label stats. Returns {tag: stats}."""
    verify_qa_sha256(qa_jsonl_file, qa_sha256 or CHEMQUESTS_QA_SHA256)
    if not cover_levels:
        raise ValueError("no cover level given (config labeling.cover_level or --cover-level)")
    levels: Dict[str, Fraction] = {}
    for c in cover_levels:
        frac = parse_cover_level(c)  # raises unless 0 < c <= 1
        levels.setdefault(cover_tag(frac), frac)

    full_text_path = full_text_jsonl_file or os.path.join(os.path.dirname(os.path.abspath(qa_jsonl_file)), "full_text.jsonl")
    full_text = load_full_text(full_text_path)
    by_doc, corpus = load_chunks(chunks_jsonl_file, full_text, embed_chunk_title)
    if not corpus:
        raise SpanQrelsError("the chunk file is empty")
    qa = load_qa(qa_jsonl_file, full_text)
    os.makedirs(output_folder, exist_ok=True)

    queries_out = [{"id": str(q["qid"]), "title": f"Query {q['qid']}", "text": q["question"]} for q in qa]
    assert [q["id"] for q in queries_out] == [str(i) for i in range(len(qa))], "query ids != qa.jsonl rows"

    table: List[Dict[str, Any]] = []
    rows_by_q: Dict[int, List[Dict[str, Any]]] = {}
    for q in qa:
        rows = overlap_rows(q["qid"], q["doc"], q["start"], q["end"], by_doc.get(q["doc"], []))
        rows_by_q[q["qid"]] = rows
        table.extend(rows)

    _write_jsonl(os.path.join(output_folder, "queries.jsonl"), queries_out)
    _write_jsonl(os.path.join(output_folder, "corpus.jsonl"), corpus)
    _write_jsonl(os.path.join(output_folder, "overlap_table.jsonl"), table)

    all_stats: Dict[str, Any] = {}
    for tag, frac in levels.items():
        qrels: List[Dict[str, Any]] = []
        steps: Counter = Counter()
        n_labels: List[int] = []
        empty_ids: List[int] = []
        step3_ids: List[int] = []
        for q in qa:
            res = label_from_rows(rows_by_q[q["qid"]], frac, q["start"], q["end"])
            steps[res["step"]] += 1
            n_labels.append(len(res["ids"]))
            if res["ids"]:
                qrels.extend({"query_id": str(q["qid"]), "doc_id": cid, "relevance": 1} for cid in res["ids"])
            else:
                qrels.append(_zero_row(q["qid"], by_doc, q["doc"], corpus))
                empty_ids.append(q["qid"])
                if verbose:
                    print(f"⚠️ [{tag}] question {q['qid']} (document {q['doc']}): no chunk overlaps the span "
                          f"({res['reason']}) -> kept with a relevance-0 row, it scores 0.")
            if res["step"] == 3:
                step3_ids.append(q["qid"])
                if verbose:
                    print(f"⚠️ [{tag}] question {q['qid']}: cover level not reachable "
                          f"({res['covered']}/{res['span_len']} characters) -> every overlapping chunk is relevant (step 3).")
        n = len(qa)
        stats = {
            "cover_level": float(frac),
            "cover_level_exact": f"{frac.numerator}/{frac.denominator}",
            "cover_tag": tag,
            "qrels_file": qrels_filename("span", frac),
            "n_questions": n,
            "n_chunks": len(corpus),
            "questions_per_step": {
                "step1": steps[1],
                "step2": steps[2],
                "step3": steps[3],
                "step0_no_overlapping_chunk": steps[0],
            },
            "labels_per_question": {
                "mean": (sum(n_labels) / n) if n else 0.0,
                "max": max(n_labels) if n_labels else 0,
                "total": sum(n_labels),
            },
            "n_empty_questions": len(empty_ids),
            "empty_question_ids": empty_ids,
            "step3_question_ids": step3_ids,
        }
        _write_jsonl(os.path.join(output_folder, qrels_filename("span", frac)), qrels)
        with open(os.path.join(output_folder, label_stats_filename(frac)), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        all_stats[tag] = stats
        if verbose:
            print(f"✅ [{tag}] c={frac.numerator}/{frac.denominator}: steps 1/2/3 = {steps[1]}/{steps[2]}/{steps[3]}, "
                  f"labels per question mean {stats['labels_per_question']['mean']:.3f} max {stats['labels_per_question']['max']}, "
                  f"empty questions {len(empty_ids)} -> {qrels_filename('span', frac)}")

    # The evaluation reads the file named in the config (labeling.qrels_file). When that name is not the
    # derived one, write the same labels under it (only possible for a single c).
    if extra_qrels_file and extra_qrels_file not in {s["qrels_file"] for s in all_stats.values()}:
        if len(levels) == 1:
            tag = next(iter(levels))
            src = os.path.join(output_folder, all_stats[tag]["qrels_file"])
            with open(src, "rb") as fi, open(os.path.join(output_folder, extra_qrels_file), "wb") as fo:
                fo.write(fi.read())
            if verbose:
                print(f"   also written as {extra_qrels_file} (labeling.qrels_file)")
        elif verbose:
            print(f"⚠️ labeling.qrels_file={extra_qrels_file!r} ignored: several cover levels were requested.")

    if verbose:
        print(f"   Queries: {len(queries_out)} | Corpus: {len(corpus)} | overlap rows: {len(table)} | "
              f"cover levels: {', '.join(levels)}")
    return all_stats


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml  # PyYAML

            return yaml.safe_load(f) or {}
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Span-based qrels from the chunker's positions and the evidence spans.")
    ap.add_argument("--config", help="stage 5 config (YAML/JSON); the options below override its values")
    ap.add_argument("--qa-jsonl", help="qa.jsonl (pinned release)")
    ap.add_argument("--chunks-jsonl", help="chunks.jsonl written by stage 4")
    ap.add_argument("--full-text-jsonl", help="full_text.jsonl (default: next to qa.jsonl)")
    ap.add_argument("--output-folder", help="folder for queries, corpus, qrels, overlap table, label stats")
    ap.add_argument("--qa-sha256", help="expected SHA256 of qa.jsonl (default: the pinned release)")
    ap.add_argument("--cover-level", nargs="+", metavar="C",
                    help="one or more cover levels in (0, 1], e.g. 0.7 0.8 1.0 (one qrels file each); overrides labeling.cover_level")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config) if args.config else {}
    labeling = cfg.get("labeling") or {}

    def pick(cli, key):
        return cli if cli is not None else cfg.get(key)

    qa_path = pick(args.qa_jsonl, "qa_jsonl_file")
    chunks_path = pick(args.chunks_jsonl, "chunks_jsonl_file")
    out = pick(args.output_folder, "output_folder")
    missing = [n for n, v in (("qa_jsonl_file", qa_path), ("chunks_jsonl_file", chunks_path), ("output_folder", out)) if not v]
    if missing:
        raise SystemExit(f"ERROR: missing {', '.join(missing)} (give --config or the command-line options)")

    if args.cover_level is not None:
        levels: List[Any] = list(args.cover_level)
        explicit_name = None  # sensitivity run: file names come from the c values
    else:
        if labeling.get("cover_level") is None:
            raise SystemExit("ERROR: no cover level: set labeling.cover_level in the config or pass --cover-level")
        levels = [labeling["cover_level"]]
        explicit_name = labeling.get("qrels_file")

    try:
        build_span_qrels(
            qa_jsonl_file=qa_path,
            chunks_jsonl_file=chunks_path,
            output_folder=out,
            cover_levels=levels,
            full_text_jsonl_file=pick(args.full_text_jsonl, "full_text_jsonl_file"),
            qa_sha256=pick(args.qa_sha256, "qa_sha256"),
            extra_qrels_file=explicit_name,
            embed_chunk_title=bool(cfg.get("embed_chunk_title", False)),
        )
    except (SpanQrelsError, ValueError) as e:
        raise SystemExit(f"ERROR: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A small synthetic artifacts root for the analysis tools (not a test module): prepared folders, runs with rankings and per-query metrics.

12 papers x 3 questions; every paper has a text of 300 words ``w<paper>_<i>``; the span of question j is a window of the text. Chunks are cut by characters
(``small`` / ``large`` step), so the span of a question is split over two small chunks for some questions and not for others. Rankings are random scores
plus a quality bonus for the chunks that the span rule labels relevant: a better ``quality`` puts the relevant chunk higher. Everything is made with the
repository's own functions (span rule, metrics, TREC writer), so the files have the layout of a real run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from pipeline_lib import span_labels as SL
from tools import coverage_diagnostics as CD, metrics as M

N_PAPERS, PER_PAPER = 12, 3
FIELDS = [["Organic Chemistry", "Catalysis"], ["Physical Chemistry"], ["Materials Science", "Energy"], []]       # paper p has FIELDS[p % 4]
DEFAULT_CONFIGS = [("fixed_token", 128, 0), ("fixed_token", 512, 0), ("hierarchical_section", 128, 0), ("hierarchical_section", 512, 0)]
CHAR_STEP = {128: 160, 256: 320, 512: 640}
QUESTIONS = ["What is the yield of {x}?", "Which catalyst gives CO2 at 25 nm?", "How does NaCl change the rate?", "Is the sample stable at 300 °C?",
             "Why does the 12 mg film crack?", "When was it measured?"]


def paper_text(p: int) -> str:
    return " ".join(f"w{p}_{i}" for i in range(300))


def build_corpus(root: Path) -> Dict[str, Any]:
    d = root / "chemquests_hf_db"
    d.mkdir(parents=True, exist_ok=True)
    qa, meta, full = [], [], []
    for p in range(1, N_PAPERS + 1):
        t = paper_text(p)
        full.append({"corpus_id": f"{p:03d}", "text": t})
        meta.append({"corpus_id": str(p), "title": f"Paper {p}", "fields_of_study": FIELDS[p % 4]})
        for j in range(PER_PAPER):
            s = 300 + 400 * j + 7 * p
            qa.append({"question": QUESTIONS[(p + j) % len(QUESTIONS)].format(x=f"w{p}_{j}"), "corpus_id": f"{p:03d}", "content": t[s:s + 70],
                       "start_index": s, "end_index": s + 70})
    for name, rows in (("qa.jsonl", qa), ("metadata.jsonl", meta), ("full_text.jsonl", full)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return {"dir": d, "qa": qa}


def build_prepared(root: Path, qa: List[Dict[str, Any]], chunker: str, size: int, overlap: int) -> Path:
    step = CHAR_STEP[size]
    folder = root / "prepared" / f"{chunker}_c{size}_o{overlap}__abc123"
    (folder / "chunks").mkdir(parents=True)
    (folder / "mteb_task").mkdir()
    chunks, by_doc = [], {}
    for p in range(1, N_PAPERS + 1):
        t = paper_text(p)
        for k, a in enumerate(range(0, len(t), step)):
            ch = {"chunk_id": f"{p}:{k}", "chunk_index": k, "source_doc_id": str(p), "start_char": a, "end_char": min(len(t), a + step), "text": t[a:a + step]}
            chunks.append(ch)
            by_doc.setdefault(str(p), []).append({"chunk_id": ch["chunk_id"], "chunk_index": k, "start": ch["start_char"], "end": ch["end_char"]})
    (folder / "chunks" / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks))
    table, qrels, steps = [], [], {1: 0, 2: 0, 3: 0, 0: 0}
    for qid, r in enumerate(qa):
        doc = str(int(r["corpus_id"]))
        rows = SL.overlap_rows(qid, doc, r["start_index"], r["end_index"], by_doc[doc])
        table += rows
        res = SL.label_from_rows(rows, 0.9, r["start_index"], r["end_index"])
        steps[res["step"]] += 1
        qrels += [{"query_id": str(qid), "doc_id": c, "relevance": 1} for c in res["ids"]]
    mt = folder / "mteb_task"
    (mt / "queries.jsonl").write_text("".join(json.dumps({"id": str(i), "title": "", "text": r["question"]}) + "\n" for i, r in enumerate(qa)))
    (mt / "corpus.jsonl").write_text("".join(json.dumps({"id": c["chunk_id"], "title": "", "text": c["text"]}) + "\n" for c in chunks))
    (mt / "overlap_table.jsonl").write_text("".join(json.dumps(r) + "\n" for r in table))
    (mt / "qrels_span_c090.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qrels))
    (mt / "label_stats_c090.json").write_text(json.dumps({"cover_level": 0.9, "n_questions": len(qa), "n_chunks": len(chunks), "questions_per_step": {
        "step1": steps[1], "step2": steps[2], "step3": steps[3], "step0_no_overlapping_chunk": steps[0]},
        "labels_per_question": {"mean": len(qrels) / len(qa), "max": 3, "total": len(qrels)}, "n_empty_questions": 0}))
    (folder / "manifest.json").write_text(json.dumps({"key": "abc123", "chunker": chunker, "size": size, "overlap": overlap}))
    inside = 100.0 * sum(1 for r in table if r["contains_span"]) / len(qa)
    (folder / "coverage_diagnostics.csv").write_text(f"n_chunks,mean_chunk_tokens,max_chunk_tokens,pct_span_inside_one_chunk\n{len(chunks)},{step / 6:.2f},{step // 5},{inside:.4f}\n")
    return folder


def rankings(prepared: Path, qa: List[Dict[str, Any]], quality: float, seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    chunks = [json.loads(l)["chunk_id"] for l in open(prepared / "chunks" / "chunks.jsonl")]
    rel: Dict[str, set] = {}
    for l in open(prepared / "mteb_task" / "qrels_span_c090.jsonl"):
        r = json.loads(l)
        rel.setdefault(r["query_id"], set()).add(r["doc_id"])
    run = {}
    for qid in range(len(qa)):
        sc = {c: float(rng.normal()) + (quality if c in rel.get(str(qid), set()) else 0.0) for c in chunks}
        run[str(qid)] = dict(sorted(sc.items(), key=lambda kv: -kv[1])[:100])
    return run


def write_run(root: Path, name: str, model: str, chunker: str, size: int, overlap: int, run: Dict[str, Dict[str, float]], prepared: Path,
              qa: List[Dict[str, Any]], extra_info: Dict[str, Any] | None = None) -> Path:
    d = root / "tests" / f"{name}_{chunker}_c{size}_o{overlap}"
    out = d / "results" / name
    out.mkdir(parents=True)
    qids = [str(i) for i in range(len(qa))]
    qrels = M.load_qrels(prepared / "mteb_task" / "qrels_span_c090.jsonl")
    M.write_per_query_csv(out / "per_query_metrics.csv", M.per_query_metrics(run, qrels, M.DEFAULT_KS, qids))
    M.write_trec_run(run, out / "run_top100.trec.gz", tag=name, depth=100)
    (out / "ChemQuest_default_predictions.json").write_text(json.dumps(run))
    cov = CD.span_coverage(run, prepared / "mteb_task" / "overlap_table.jsonl", qids, CD.COVERAGE_KS, CD.corpus_chunk_chars(prepared / "mteb_task" / "corpus.jsonl"))
    CD.write_span_coverage_csv(out / "span_coverage.csv", cov)
    (d / "results" / "summary.json").write_text("[]")
    (d / "run_info.json").write_text(json.dumps({"model": model, "chunker": chunker, "size": size, "overlap": overlap, "prepared_dir": str(prepared),
                                                  "prepared_key": "abc123", **(extra_info or {})}))
    return d


MODELS = {"org/m_a": 2.2, "org/m_b": 1.2, "org/m_c": 0.4, "bm25": 1.0}


def build_world(root: Path, configs: Sequence[Tuple[str, int, int]] = DEFAULT_CONFIGS, models: Dict[str, float] | None = None, seed: int = 0,
                with_rerank: bool = True) -> Dict[str, Any]:
    """Artifacts root with the corpus, one prepared folder per configuration and one run per model (+ a rerank run of org/m_c)."""
    corpus = build_corpus(root)
    qa = corpus["qa"]
    models = models or MODELS
    prepared = {}
    for k, (ch, s, o) in enumerate(configs):
        prepared[(ch, s, o)] = build_prepared(root, qa, ch, s, o)
        for m, (name, q) in enumerate(models.items()):
            bonus = q + (0.5 if s >= 512 else 0.0) + (0.2 if ch == "hierarchical_section" else 0.0)       # larger chunks and the other strategy help a bit
            write_run(root, name.replace("/", "_"), name, ch, s, o, rankings(prepared[(ch, s, o)], qa, bonus, seed + 100 * k + m), prepared[(ch, s, o)], qa)
        if with_rerank:
            write_run(root, "rerank_m_c", "rerank_m_c", ch, s, o, rankings(prepared[(ch, s, o)], qa, 3.0, seed + 7), prepared[(ch, s, o)], qa,
                      {"rerank": {"model": "ce", "revision": "r", "top_n": 50, "dense": "org/m_c", "depth": 100}})
    return {"root": root, "qa_path": corpus["dir"] / "qa.jsonl", "qa": qa, "prepared": prepared, "dir": corpus["dir"]}

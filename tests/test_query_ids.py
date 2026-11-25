"""B12: query id == row index in qa.jsonl, also when questions have nothing to match."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, qa_lines, chunk_rows):
    qa = tmp_path / "qa.jsonl"
    qa.write_text("".join(qa_lines))
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("".join(json.dumps(r) + "\n" for r in chunk_rows))
    out = tmp_path / "task"
    cfg = tmp_path / "s5.yaml"
    cfg.write_text(yaml.safe_dump({
        "qa_jsonl_file": str(qa), "chunks_jsonl_file": str(chunks), "output_folder": str(out),
        "qa_sha256": hashlib.sha256(qa.read_bytes()).hexdigest(),
        "min_score_threshold": 0.95, "max_results_per_query": 5,
    }))
    r = subprocess.run([sys.executable, "-m", "stages.make_mteb_task", "--config", str(cfg)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [json.loads(l) for l in (out / "queries.jsonl").read_text().splitlines()], \
           [json.loads(l) for l in (out / "qrels_lexical.jsonl").read_text().splitlines()]


def _qa(question, content):
    return json.dumps({"question": question, "corpus_id": "1", "content": content,
                       "start_index": 0, "end_index": len(content)}) + "\n"


def test_query_id_is_row_index_even_if_questions_are_empty(tmp_path):
    chunks = [
        {"source_doc_id": "1", "chunk_index": 0, "chunk_id": "1:0", "text": "alpha beta gamma"},
        {"source_doc_id": "1", "chunk_index": 1, "chunk_id": "1:1", "text": "delta epsilon zeta"},
        {"source_doc_id": "1", "chunk_index": 2, "chunk_id": "1:2", "text": "eta theta iota"},
    ]
    qa_lines = [
        _qa("", ""),                              # row 0: empty (used to be skipped, shifting every later id)
        _qa("about alpha?", "alpha beta"),        # row 1
        "\n",                                     # blank line: not a record
        _qa("", ""),                              # row 2: empty again
        _qa("about theta?", "theta iota"),        # row 3
        _qa("about delta?", "delta epsilon"),     # row 4
    ]
    queries, qrels = _run(tmp_path, qa_lines, chunks)
    assert [q["id"] for q in queries] == ["0", "1", "2", "3", "4"]
    assert [q["text"] for q in queries] == ["", "about alpha?", "", "about theta?", "about delta?"]
    pos = {r["query_id"]: r["doc_id"] for r in qrels if r["relevance"] == 1}
    assert pos == {"1": "1:0", "3": "1:2", "4": "1:1"}   # ids stay glued to their own question
    assert {r["query_id"] for r in qrels if r["relevance"] == 0} == {"0", "2"}

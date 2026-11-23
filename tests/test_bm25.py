"""BM25 baseline: ranking, tie-breaking, and the full run on a small task with the same outputs as run_eval."""
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("rank_bm25")
pytest.importorskip("mteb")

from pipeline_lib.bm25_retriever import BM25Retriever, tokenize
from stages import run_bm25

CORPUS = {
    "0:0": {"title": "0 - Chunk 0", "text": "Sodium hydroxide was added to the flask."},
    "0:1": {"title": "0 - Chunk 1", "text": "The mixture was stirred under argon for two hours."},
    "1:0": {"title": "1 - Chunk 0", "text": "Palladium catalysed cross coupling gave the biaryl product."},
    "1:1": {"title": "1 - Chunk 1", "text": "The mixture was stirred."},
    "2:0": {"title": "2 - Chunk 0", "text": "unrelated text about gardening and tomatoes"},
}
QUERIES = {"0": "Which catalyst was used for cross coupling?", "1": "How long was it stirred under argon?", "2": "tomatoes"}
QRELS = {"0": {"1:0": 1}, "1": {"0:1": 1}, "2": {"0:0": 0}}      # query 2 has no relevant chunk (relevance-0 row)


def test_tokenizer_lowercases_and_keeps_unicode_words():
    assert tokenize("Cross-Coupling of α-pinene (2.5 mL)") == ["cross", "coupling", "of", "α", "pinene", "2", "5", "ml"]


def test_ranking_and_depth():
    r = BM25Retriever()
    r.index(CORPUS)
    run = r.search(QUERIES, top_k=3)
    assert list(run["0"])[0] == "1:0" and list(run["1"])[0] == "0:1" and list(run["2"])[0] == "2:0"
    assert all(len(v) == 3 for v in run.values())


def test_ties_are_broken_by_descending_chunk_id():
    r = BM25Retriever()
    r.index({"a:0": {"text": "same words"}, "b:0": {"text": "same words"}, "c:0": {"text": "same words"}})
    assert list(r.search({"q": "same"}, top_k=3)["q"]) == ["c:0", "b:0", "a:0"]


def test_run_bm25_writes_the_same_outputs_as_run_eval(tmp_path):
    d = tmp_path / "task"; d.mkdir()
    (d / "corpus.jsonl").write_text("".join(json.dumps({"id": k, **v}) + "\n" for k, v in CORPUS.items()))
    (d / "queries.jsonl").write_text("".join(json.dumps({"id": k, "text": t}) + "\n" for k, t in QUERIES.items()))
    (d / "qrels_span_c090.jsonl").write_text("".join(json.dumps({"query_id": q, "doc_id": c, "relevance": r}) + "\n" for q, ds in QRELS.items() for c, r in ds.items()))
    cfg = {"models": [{"name": "bm25"}], "output_root": str(tmp_path / "res"), "top_k": 100, "seed": 5,
           "bm25": {"k1": 1.2, "b": 0.7},
           "tasks": [{"class": "pipeline_lib.my_tasks.ChemQuest", "kwargs": {"data_dir": str(d), "qrels_file": "qrels_span_c090.jsonl"}}]}
    p = tmp_path / "s6.yaml"; p.write_text(yaml.safe_dump(cfg))
    assert run_bm25.main(["--config", str(p)]) == 0
    out = next((tmp_path / "res").glob("bm25_*"))
    for f in ("ChemQuest_default_predictions.json", "ChemQuest.json", "run_top100.trec.gz", "per_query_metrics.csv",
              "metrics_summary.json", "run_manifest.json"):
        assert (out / f).exists(), f
    assert (tmp_path / "res" / "summary.json").exists()
    res = json.loads((out / "ChemQuest.json").read_text())["scores"]["test"][0]
    assert res["recall_at_10"] == pytest.approx(2 / 3, abs=1e-5)       # MTEB rounds to 5 decimals; two of three questions solved, one is empty
    ms = json.loads((out / "metrics_summary.json").read_text())
    assert ms["cross_check"]["ok"] and ms["aggregate"]["n_empty_questions"] == 1
    m = json.loads((out / "run_manifest.json").read_text())
    t = m["timing_s"]       # evaluation_time = index + search + metrics + dump (what MTEB's evaluation_time covers), not only index + search
    assert t["mteb_evaluation_time_s"] >= t["bm25_index_s"] + t["bm25_search_s"] + t["metrics_s"] + t["dump_s"] - 1e-3
    assert t["mteb_evaluation_time_s"] == pytest.approx(json.loads((out / "ChemQuest.json").read_text())["evaluation_time"])
    assert m["model"]["k1"] == 1.2 and m["seeds"]["seed"] == 5 and "bm25_index_s" in m["timing_s"] and m["prompts"] is None

"""tools/hybrid.py: reciprocal rank fusion of saved rankings."""
import json
from pathlib import Path

import pytest

from tools import hybrid, metrics as M


def test_rrf_formula_k60_and_ranks():
    a = {"q": {"x": 0.9, "y": 0.5, "z": 0.1}}      # x1 y2 z3
    b = {"q": {"y": 7.0, "w": 3.0, "x": 1.0}}      # y1 w2 x3
    f = hybrid.fuse([a, b], k=60)["q"]
    assert f["x"] == pytest.approx(1 / 61 + 1 / 63)
    assert f["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert f["z"] == pytest.approx(1 / 63) and f["w"] == pytest.approx(1 / 62)
    assert [d for d, _ in M.ranked(f)] == ["y", "x", "w", "z"]


def test_scores_do_not_matter_only_ranks():
    a = {"q": {"x": 0.9, "y": 0.5}}
    b = {"q": {"x": 1000.0, "y": -5.0}}
    assert hybrid.fuse([a], 60) == hybrid.fuse([b], 60)


def test_depth_cuts_each_input_and_missing_docs_get_nothing():
    a = {"q": {f"d{i}": 100 - i for i in range(10)}}
    f = hybrid.fuse([a], k=60, depth=3)["q"]
    assert set(f) == {"d0", "d1", "d2"}


def test_ties_break_by_document_id_descending():
    a = {"q": {"a": 1.0}}
    b = {"q": {"b": 1.0}}                           # both rank 1 in their list: equal fused score
    f = hybrid.fuse([a, b])["q"]
    assert f["a"] == f["b"] and [d for d, _ in M.ranked(f)] == ["b", "a"]


def test_different_query_sets_are_rejected():
    with pytest.raises(ValueError):
        hybrid.fuse([{"q1": {"a": 1.0}}, {"q2": {"a": 1.0}}])


def _make_run(tests, name, model, ranking, prepared, key="K"):
    d = tests / name
    out = d / "results" / model.replace("/", "_")
    out.mkdir(parents=True)
    (out / "ChemQuest_default_predictions.json").write_text(json.dumps(ranking))
    M.write_trec_run(ranking, out / "run_top100.trec.gz")
    (out / "metrics_summary.json").write_text(json.dumps({"qrels_file": "qrels_span_c090.jsonl"}))
    (d / "run_info.json").write_text(json.dumps({"model": model, "chunker": "fixed_token", "size": 128, "overlap": 0,
                                                  "prepared_dir": str(prepared), "prepared_key": key}))
    return d


def test_run_folder_end_to_end(tmp_path):
    prepared = tmp_path / "prep"
    (prepared / "mteb_task").mkdir(parents=True); (prepared / "chunks").mkdir()
    (prepared / "mteb_task" / "queries.jsonl").write_text("".join(json.dumps({"id": q, "text": "t"}) + "\n" for q in ("0", "1")))
    (prepared / "mteb_task" / "qrels_span_c090.jsonl").write_text(
        "".join(json.dumps({"query_id": q, "doc_id": d, "relevance": 1}) + "\n" for q, d in (("0", "c1"), ("1", "c3"))))
    (prepared / "chunks" / "chunks.jsonl").write_text("")
    dense = {"0": {"c0": 3.0, "c1": 2.0, "c2": 1.0}, "1": {"c3": 3.0, "c0": 2.0, "c1": 1.0}}
    sparse = {"0": {"c1": 9.0, "c2": 5.0, "c0": 1.0}, "1": {"c0": 9.0, "c3": 5.0, "c2": 1.0}}
    tests = tmp_path / "tests"; tests.mkdir()
    dd = _make_run(tests, "dense_fixed_token_c128_o0", "org/dense", dense, prepared)
    sd = _make_run(tests, "bm25_fixed_token_c128_o0", "bm25", sparse, prepared)
    r = hybrid.run_hybrid(dd, sd, tests)
    out = Path(r["run_dir"]) / "results" / "hybrid_rrf60_org_dense_bm25"
    for f in ("run_top100.trec.gz", "per_query_metrics.csv", "metrics_summary.json", "run_manifest.json", "ChemQuest_default_predictions.json"):
        assert (out / f).exists(), f
    pq = M.load_per_query_csv(out / "per_query_metrics.csv")
    assert pq["0"]["recall_at_10"] == 1.0 and pq["1"]["recall_at_10"] == 1.0
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["k"] == 60 and man["no_embedding"] is True and len(man["inputs"]) == 2
    info = json.loads((Path(r["run_dir"]) / "run_info.json").read_text())
    assert info["hybrid"]["dense"] == "org/dense" and info["model"] == "hybrid_rrf60_org_dense_bm25"
    assert hybrid.run_hybrid(dd, sd, tests)["status"] == "skipped"
    assert "org/dense" in hybrid.discover(tests)[("fixed_token", 128, 0)] and len(hybrid.discover(tests)[("fixed_token", 128, 0)]) == 2   # hybrid runs are not inputs
    with pytest.raises(ValueError, match="same prepared"):
        (sd / "run_info.json").write_text(json.dumps({**json.loads((sd / "run_info.json").read_text()), "prepared_key": "OTHER"}))
        hybrid.run_hybrid(dd, sd, tests, force=True)


def test_rerun_hybrid_step_fuses_every_dense_run_with_bm25(tmp_path):
    pytest.importorskip("transformers")
    from stages import rerun
    prepared = tmp_path / "prep"
    (prepared / "mteb_task").mkdir(parents=True); (prepared / "chunks").mkdir()
    (prepared / "mteb_task" / "queries.jsonl").write_text(json.dumps({"id": "0", "text": "t"}) + "\n")
    (prepared / "mteb_task" / "qrels_span_c090.jsonl").write_text(json.dumps({"query_id": "0", "doc_id": "c1", "relevance": 1}) + "\n")
    root = tmp_path / "art"; tests = root / "tests"; tests.mkdir(parents=True)
    run = {"0": {"c0": 3.0, "c1": 2.0}}
    _make_run(tests, "a_fixed_token_c128_o0", "org/a", run, prepared)
    _make_run(tests, "b_fixed_token_c128_o0", "org/b", run, prepared)
    _make_run(tests, "bm25_fixed_token_c128_o0", "bm25", run, prepared)
    models = [("org/a", "r"), ("org/b", "r"), ("bm25", "rank_bm25")]
    out = rerun.run_hybrids(str(root), models, ["fixed_token"], [(128, 0)], k=60, depth=100)
    assert [r["status"] for r in out] == ["done", "done"]
    assert (tests / "hybrid_rrf60_org_a_bm25_fixed_token_c128_o0" / "results" / "summary.json").exists()
    assert (tests / "hybrid_rrf60_org_b_bm25_fixed_token_c128_o0" / "results" / "summary.json").exists()
    assert [r["status"] for r in rerun.run_hybrids(str(root), models, ["fixed_token"], [(128, 0)], k=60, depth=100)] == ["skipped", "skipped"]
    assert rerun.run_hybrids(str(root), [("org/a", "r")], ["fixed_token"], [(128, 0)], k=60, depth=100) == []     # no BM25: nothing to fuse

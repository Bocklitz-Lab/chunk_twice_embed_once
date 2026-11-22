"""tools/make_report.py: the post-run step (statistics, tables) on a tiny synthetic artifacts root."""
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("scipy")
from tools import make_report as R
from tools import metrics as M


def _run(tests, name, model, chunker, size, per_query, hybrid=None):
    d = tests / name
    out = d / "results" / model.replace("/", "_")
    out.mkdir(parents=True)
    pq = {q: {k: v for k, v in zip(("ndcg_at_10", "recall_at_10", "geom_at_10", "mrr_at_100"), (n, r, (n * r) ** .5, n))}
          for q, (n, r) in per_query.items()}
    M.write_per_query_csv(out / "per_query_metrics.csv", pq)
    (out / "metrics_summary.json").write_text(json.dumps({"qrels_file": "qrels_span_c090.jsonl", "aggregate": {}}))
    info = {"model": model, "chunker": chunker, "size": size, "overlap": 0, "prepared_dir": str(tests.parent), "prepared_key": "K"}
    if hybrid:
        info["hybrid"] = hybrid
    (d / "run_info.json").write_text(json.dumps(info))
    (d / "results" / "summary.json").write_text("[]")


def test_comparison_pairs_follow_the_families():
    names = ["a/dense", "bm25", "hybrid_rrf60_a_dense_bm25"]
    infos = {"hybrid_rrf60_a_dense_bm25|fixed_token|128|0": {"hybrid": {"dense": "a/dense", "sparse": "bm25"}}}
    assert R.comparison_pairs(names, infos, ("fixed_token", 128, 0)) == [
        ("a/dense", "bm25"), ("hybrid_rrf60_a_dense_bm25", "a/dense"), ("hybrid_rrf60_a_dense_bm25", "bm25")]
    assert R.comparison_pairs(["a/dense"], {}, ("fixed_token", 128, 0)) == []          # no BM25: no pair


def test_build_report_writes_tracked_tables_manifest_and_readme(tmp_path):
    root = tmp_path / "art"
    tests = root / "tests"; tests.mkdir(parents=True)
    qa = root / "qa.jsonl"
    qa.write_text("".join(json.dumps({"question": f"q{i} word", "corpus_id": str(i // 2), "content": "word", "start_index": 0, "end_index": 4}) + "\n" for i in range(6)))
    base = {str(i): (0.5 + 0.05 * (i % 3), 0.7) for i in range(6)}
    better = {q: (n + 0.1, r + 0.1) for q, (n, r) in base.items()}
    _run(tests, "d_fixed_token_c128_o0", "org/dense", "fixed_token", 128, base)
    _run(tests, "bm25_fixed_token_c128_o0", "bm25", "fixed_token", 128, base)
    _run(tests, "h_fixed_token_c128_o0", "hybrid_rrf60_org_dense_bm25", "fixed_token", 128, better,
         hybrid={"method": "rrf", "k": 60, "dense": "org/dense", "sparse": "bm25"})
    out = tmp_path / "results" / "v2"
    m = R.build_report(root, out, qa, resamples=50, overlap=False, verbose=False)
    for f in ("table_main.csv", "runs_overview.csv", "stats_ci.csv", "stats_compare.csv", "README.md", "report_manifest.json"):
        assert (out / f).exists(), f
    assert (root / "report" / "stats_compare.csv").exists()                         # the full copy stays in the artifacts
    rows = list(csv.DictReader(open(out / "stats_compare.csv")))
    assert {(r["A"], r["B"]) for r in rows} == {("org/dense", "bm25"), ("hybrid_rrf60_org_dense_bm25", "org/dense"), ("hybrid_rrf60_org_dense_bm25", "bm25")}
    assert {r["level"] for r in rows} == {"query", "paper"}
    hy = next(r for r in rows if r["A"].startswith("hybrid") and r["B"] == "bm25" and r["metric"] == "ndcg_at_10" and r["level"] == "paper")
    assert float(hy["diff"]) == pytest.approx(0.1) and "p_bootstrap_holm" in hy
    main = list(csv.DictReader(open(out / "table_main.csv")))
    assert [r["model"] for r in main] == ["bm25", "hybrid_rrf60_org_dense_bm25", "org/dense"] and main[0]["geom_ci95_low_paper"]
    assert m["bootstrap"]["resamples"] == 50 and set(m["files"]) >= {"table_main.csv", "stats_ci.csv"}
    assert "table_main.csv" in (out / "README.md").read_text()


def test_large_files_stay_in_the_artifacts(tmp_path):
    root = tmp_path / "art"; tests = root / "tests"; tests.mkdir(parents=True)
    qa = root / "qa.jsonl"
    qa.write_text(json.dumps({"question": "q", "corpus_id": "0", "content": "q", "start_index": 0, "end_index": 1}) + "\n")
    _run(tests, "bm25_fixed_token_c128_o0", "bm25", "fixed_token", 128, {"0": (0.5, 0.5)})
    out = tmp_path / "out"
    m = R.build_report(root, out, qa, resamples=10, overlap=False, verbose=False, track_max_mb=0.0001)
    assert "runs_overview.csv" in m["kept_in_artifacts_only"] and not (out / "runs_overview.csv").exists()
    assert (root / "report" / "runs_overview.csv").exists()

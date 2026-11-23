"""stages/rerun.py end to end with BM25 only (no GPU): prepare once, evaluate several 'models', link, skip, check hashes."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers"); pytest.importorskip("mteb"); pytest.importorskip("rank_bm25")

from pipeline_lib.dataset_pin import sha256_file
from stages import rerun
from tests.test_prepare import data  # noqa: F401  (fixture)
from tools import check_identical

ROOT = Path(__file__).resolve().parents[1]


def _go(tmp_path, data, *extra, command="all"):
    argv = [command, "--artifacts-root", str(tmp_path / "art"), "--main-config", str(ROOT / "configs/main_config.yaml"),
            "--models", "bm25@rank_bm25", "bm25-b@rank_bm25", "--chunkers", "fixed_token", "--sizes", "128", "--overlaps", "0",
            "--qa-sha256", sha256_file(data / "qa.jsonl"), "--param", f"paths.chemquests_dir={data}",
            "--param", "pilot.x=1", "--report-dir", str(tmp_path / "report"), "--no-rerank", *extra]     # the report must not overwrite the tracked tables
    return rerun.main(argv)


def test_two_models_share_one_prepared_folder_and_the_check_passes(tmp_path, data):
    assert _go(tmp_path, data) == 0
    tests = tmp_path / "art" / "tests"
    runs = sorted(p for p in tests.iterdir() if p.is_dir())
    assert [r.name for r in runs] == ["bm25_b_fixed_token_c128_o0", "bm25_fixed_token_c128_o0"]
    for r in runs:
        assert r.joinpath("chunks").is_symlink() and r.joinpath("mteb_task").is_symlink()
        assert (r / "results" / "summary.json").exists() and (r / "run_info.json").exists()
        out = next((r / "results").glob("bm25_*"))
        assert (out / "run_top100.trec.gz").exists() and (out / "per_query_metrics__qrels_lexical.csv").exists()
    assert (runs[0] / "mteb_task").resolve() == (runs[1] / "mteb_task").resolve()
    assert len(list((tmp_path / "art" / "prepared").iterdir())) == 1               # chunked once
    assert len((tests / "durations.csv").read_text().splitlines()) == 3
    # second call: everything is skipped
    stamp = (runs[0] / "results" / "summary.json").stat().st_mtime_ns
    assert _go(tmp_path, data) == 0 and (runs[0] / "results" / "summary.json").stat().st_mtime_ns == stamp
    assert check_identical.main(["--tests-root", str(tests), "--chunker", "fixed_token", "--size", "128", "--overlap", "0"]) == 0


def test_check_detects_a_model_that_saw_other_labels(tmp_path, data):
    assert _go(tmp_path, data) == 0
    tests = tmp_path / "art" / "tests"
    # one model gets its own (altered) copy of the labels instead of the shared link
    run = tests / "bm25_b_fixed_token_c128_o0"
    target = (run / "mteb_task").resolve()
    (run / "mteb_task").unlink()
    (run / "mteb_task").mkdir()
    for f in target.iterdir():
        if f.is_file():
            (run / "mteb_task" / f.name).write_bytes(f.read_bytes())
    with open(run / "mteb_task" / "qrels_span_c090.jsonl", "a") as fh:
        fh.write(json.dumps({"query_id": "0", "doc_id": "0:0", "relevance": 1}) + "\n")
    assert check_identical.main(["--tests-root", str(tests)]) == 1


def test_dry_run_changes_nothing(tmp_path, data):
    assert _go(tmp_path, data, "--dry-run") == 0
    assert not (tmp_path / "art" / "tests").exists()


def test_collect_runs_gives_one_row_per_run(tmp_path, data):
    from tools import collect_runs
    assert _go(tmp_path, data) == 0
    rows = collect_runs.collect(tmp_path / "art" / "tests")
    assert [r["model"] for r in rows] == ["bm25", "bm25-b"] and all(r["chunker"] == "fixed_token" for r in rows)
    r = rows[0]
    assert r["labels"] == "qrels_span_c090.jsonl" and r["n_chunks"] and r["span_ndcg_at_10"] is not None
    assert r["lexical_ndcg_at_10"] is not None and r["span_coverage_at_5"] is not None and r["bm25_search_s"] is not None
    collect_runs.write_csv(tmp_path / "runs.csv", rows)
    assert (tmp_path / "runs.csv").read_text().count("\n") == 3

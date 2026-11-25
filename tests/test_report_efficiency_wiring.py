"""`make report` calls tools.model_properties and tools.efficiency_analysis and records what happened (audit section 3 item 3)."""
import csv
import json
from pathlib import Path

import pandas as pd

from tools import efficiency_analysis as EA
from tools import make_report as R

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "artifacts/subset_v1"


def _eff_rows(model="org/a", n=3):
    rows = []
    for i, size in enumerate((128, 256, 512)[:n]):
        rows.append({"model": model, "revision": "r", "chunker": "fixed_token", "size": size, "overlap": 0, "n_chunks": 1000 // (i + 1), "embedding_dim": 384, "index_mb_fp32": 1.5 * (i + 1),
                     "index_mb_fp16": 0.7, "n_params": 33_000_000, "weights_mb": 120.0, "weights_fp16_mb": 60.0, "peak_gpu_mem_mb": 500.0, "corpus_encode_s": 10.0,
                     "corpus_chunks_per_s": 100.0, "latency_measured": False})
    return rows


def test_efficiency_analysis_writes_the_phase1_table_with_split_columns(tmp_path, monkeypatch):
    """Regression: the string columns split / n_queries were formatted as numbers and the Phase 1 table crashed."""
    pd.DataFrame(_eff_rows()).to_csv(tmp_path / "eff2.csv", index=False)
    pd.DataFrame([{"model": "org/a", "revision": "r", "task": "ChemNQRetrieval", "n_params": 33e6, "weights_mb": 120.0, "embedding_dim": 384, "index_mb_fp32": 2.0,
                   "peak_gpu_mem_mb": 400.0, "corpus_chunks_per_s": 90.0}]).to_csv(tmp_path / "eff1.csv", index=False)
    pd.DataFrame([{"model": "org/a", "chunker": "fixed_token", "size": s, "overlap": 0, "span_geom_at_10": 0.5 + s / 2000} for s in (128, 256, 512)]).to_csv(tmp_path / "scores.csv", index=False)
    rc = EA.main(["--efficiency", str(tmp_path / "eff2.csv"), "--scores", str(tmp_path / "scores.csv"), "--phase1", str(tmp_path / "eff1.csv"), "--out", str(tmp_path / "o")])
    assert rc == 0 and (tmp_path / "o" / "table_efficiency_phase1.tex").exists() and (tmp_path / "o" / "table_efficiency_models.csv").exists()


def test_report_helper_writes_both_tables_and_never_raises(tmp_path):
    heavy = tmp_path / "report"
    heavy.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    out = R._properties_and_efficiency(tmp_path, tests, heavy, ROOT / "configs" / "main_config.yaml")
    assert "model_properties_exit" in out or "model_properties_skipped" in out
    assert "efficiency_analysis_skipped" in out                      # no efficiency_phase2.csv: recorded, not raised
    for name in ("table_model_properties.csv", "table_efficiency_models.csv", "table_index_size.csv", "table_index_chunks.csv", "table_index_size_long.csv", "table_efficiency_phase1.csv"):
        assert name in R.DESCRIPTIONS                                # copied to the tracked folder and listed in the README / manifest


def test_report_runs_efficiency_tables_when_the_csv_exists(tmp_path):
    heavy = tmp_path / "report"
    heavy.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    pd.DataFrame(_eff_rows()).to_csv(tests / "efficiency_phase2.csv", index=False)
    pd.DataFrame([{"model": "org/a", "chunker": "fixed_token", "size": s, "overlap": 0, "span_geom_at_10": 0.5 + s / 2000} for s in (128, 256, 512)]).to_csv(heavy / "runs_overview.csv", index=False)
    out = R._properties_and_efficiency(tmp_path, tests, heavy, ROOT / "configs" / "main_config.yaml")
    assert out.get("efficiency_analysis_exit") == 0 and (heavy / "table_efficiency_models.csv").exists() and (heavy / "table_index_size.csv").exists()

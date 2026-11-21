"""B11: a configuration is skipped only if its final Stage 6 result exists."""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(ROOT / "Makefile", work / "Makefile")
    shutil.copy(ROOT / "test.mk", work / "test.mk")
    (work / "models.csv").write_text("dataset,name,revision\nd,org/m,rev1\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python"
    stub.write_text('#!/bin/bash\necho "$@" >> "$PYSTUB_LOG"\nexit 0\n')
    stub.chmod(0o755)
    return work, bindir


def _make(work, bindir, tmp_path):
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "PYSTUB_LOG": str(log), "HOME": str(tmp_path)}
    r = subprocess.run(
        ["make", "stage4-6", "MODELS_CSV=models.csv", "CHUNKER=fixed_token", "SIZE=128", "OVERLAP=0"],
        cwd=work, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    calls = log.read_text()
    return r.stdout, [s for s in ("stages.chunking", "stages.stage5", "stages.run_eval") if s in calls]


def test_nonempty_folder_without_result_is_rerun(tmp_path):
    work, bindir = _setup(tmp_path)
    out = work / "artifacts/tests/org_m_fixed_token_c128_o0"
    (out / "chunks").mkdir(parents=True)
    (out / "chunks/chunks.jsonl").write_text("partial run that failed after stage 4\n")
    stdout, stages = _make(work, bindir, tmp_path)
    assert stages == ["stages.chunking", "stages.stage5", "stages.run_eval"]
    assert "Earlier run incomplete" in stdout


def test_folder_with_final_result_is_skipped(tmp_path):
    work, bindir = _setup(tmp_path)
    out = work / "artifacts/tests/org_m_fixed_token_c128_o0"
    (out / "results").mkdir(parents=True)
    (out / "results/summary.json").write_text("[]")
    stdout, stages = _make(work, bindir, tmp_path)
    assert stages == []
    assert "Skip: finished" in stdout


def test_empty_summary_file_is_not_a_result(tmp_path):
    work, bindir = _setup(tmp_path)
    out = work / "artifacts/tests/org_m_fixed_token_c128_o0"
    (out / "results").mkdir(parents=True)
    (out / "results/summary.json").write_text("")
    _, stages = _make(work, bindir, tmp_path)
    assert len(stages) == 3


def test_extra_params_reach_split_config(tmp_path):
    work, bindir = _setup(tmp_path)
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "PYSTUB_LOG": str(log), "HOME": str(tmp_path)}
    r = subprocess.run(
        ["make", "stage4-6", "MODELS_CSV=models.csv", "CHUNKER=fixed_token", "SIZE=128", "OVERLAP=0",
         "EXTRA_PARAMS=--param eval.top_k=100"],
        cwd=work, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    split_calls = [l for l in log.read_text().splitlines() if "pipeline_lib.split_config" in l]
    assert len(split_calls) == 1 and "--param eval.top_k=100" in split_calls[0]

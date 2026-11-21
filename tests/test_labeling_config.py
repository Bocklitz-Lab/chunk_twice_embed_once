"""The `labeling:` block: defaults in main_config.yaml, derived qrels file, overrides, and the generated stage files."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAIN = ROOT / "configs" / "main_config.yaml"


def _split(tmp_path, *params, sections=("stage4", "stage5", "stage6"), name="out"):
    out = tmp_path / name
    cmd = [sys.executable, "-m", "pipeline_lib.split_config", "--input", str(MAIN), "--output-dir", str(out),
           "--sections", *sections]
    for p in params:
        cmd += ["--param", p]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r, out


def _stage(out, n):
    return yaml.safe_load((out / f"stage{n}.yaml").read_text())


def _qrels_kwarg(stage6):
    (task,) = [t for t in stage6["tasks"] if isinstance(t, dict)]
    return task["kwargs"]["qrels_file"]


def test_main_config_has_the_labeling_block_with_the_documented_default():
    block = yaml.safe_load(MAIN.read_text())["params"]["labeling"]
    assert block == {"method": "span", "cover_level": 0.90, "qrels_file": None}
    assert isinstance(block["cover_level"], float)


def test_generated_stage_files_round_trip_the_labeling_block_exactly(tmp_path):
    r, out = _split(tmp_path)
    assert r.returncode == 0, r.stderr
    expected = {"method": "span", "cover_level": 0.9, "qrels_file": "qrels_span_c090.jsonl"}
    s5, s6 = _stage(out, 5), _stage(out, 6)
    assert s5["labeling"] == expected
    assert type(s5["labeling"]["cover_level"]) is float and s5["labeling"]["cover_level"] == 0.9
    assert list(s5["labeling"]) == ["method", "cover_level", "qrels_file"]
    assert _qrels_kwarg(s6) == "qrels_span_c090.jsonl"
    assert s5["full_text_jsonl_file"].endswith("chemquests_hf_db/full_text.jsonl")
    # stage 4 and 6 are not touched otherwise
    assert "labeling" not in _stage(out, 4)
    # the stage 5 file passes the strict value checks of the labelling code
    from pipeline_lib.span_labels import resolve_labeling
    assert resolve_labeling(s5["labeling"]) == expected


@pytest.mark.parametrize("value, tag", [("0.8", "c080"), ("1.0", "c100"), ("1", "c100"), ("0.875", "c087p5"), ("0.05", "c005")])
def test_cover_level_override_reaches_stage_5_and_6(tmp_path, value, tag):
    r, out = _split(tmp_path, f"labeling.cover_level={value}")
    assert r.returncode == 0, r.stderr
    s5, s6 = _stage(out, 5), _stage(out, 6)
    assert s5["labeling"]["cover_level"] == float(value)
    assert s5["labeling"]["qrels_file"] == f"qrels_span_{tag}.jsonl" == _qrels_kwarg(s6)


def test_lexical_method_uses_the_lexical_file(tmp_path):
    r, out = _split(tmp_path, "labeling.method=lexical")
    assert r.returncode == 0, r.stderr
    assert _stage(out, 5)["labeling"]["method"] == "lexical"
    assert _stage(out, 5)["labeling"]["qrels_file"] == "qrels_lexical.jsonl" == _qrels_kwarg(_stage(out, 6))


def test_an_explicit_qrels_file_is_kept(tmp_path):
    r, out = _split(tmp_path, "labeling.qrels_file=my_qrels.jsonl", "labeling.cover_level=0.8")
    assert r.returncode == 0, r.stderr
    assert _stage(out, 5)["labeling"] == {"method": "span", "cover_level": 0.8, "qrels_file": "my_qrels.jsonl"}
    assert _qrels_kwarg(_stage(out, 6)) == "my_qrels.jsonl"


@pytest.mark.parametrize("param", ["labeling.cover_level=0", "labeling.cover_level=1.5", "labeling.cover_level=-0.2",
                                   "labeling.cover_level=null", "labeling.cover_level=abc", "labeling.method=bm25"])
def test_invalid_labeling_values_stop_split_config(tmp_path, param):
    r, out = _split(tmp_path, param)
    assert r.returncode != 0 and "labeling" in r.stderr
    assert not out.exists() or not list(out.glob("*.yaml"))


def test_a_config_without_a_labeling_block_still_splits(tmp_path):
    cfg = yaml.safe_load(MAIN.read_text())
    del cfg["params"]["labeling"]
    cfg["stage5"].pop("labeling")
    cfg["stage6"]["tasks"][0]["kwargs"].pop("qrels_file")
    src = tmp_path / "old.yaml"
    src.write_text(yaml.safe_dump(cfg, sort_keys=False))
    r = subprocess.run([sys.executable, "-m", "pipeline_lib.split_config", "--input", str(src), "--output-dir",
                        str(tmp_path / "o"), "--sections", "stage5"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "labeling" not in _stage(tmp_path / "o", 5)


# ---- through the real Makefile (python stubbed except split_config)
def _make_env(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(ROOT / "Makefile", work / "Makefile")
    shutil.copy(ROOT / "test.mk", work / "test.mk")
    (work / "models.csv").write_text("dataset,name,revision\nd,org/m,rev1\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python"
    stub.write_text(
        '#!/bin/bash\n'
        'echo "$@" >> "$PYSTUB_LOG"\n'
        'if [[ "$*" == *pipeline_lib.split_config* ]]; then exec "$REAL_PYTHON" "$@"; fi\n'
        'exit 0\n'
    )
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "PYSTUB_LOG": str(log), "HOME": str(tmp_path),
           "REAL_PYTHON": sys.executable, "PYTHONPATH": str(ROOT)}
    return work, env, log


def _run_make(work, env, *extra):
    r = subprocess.run(["make", "stage4-6", "MODELS_CSV=models.csv", "CHUNKER=fixed_token", "SIZE=128", "OVERLAP=0",
                        f"INPUT_CONFIG={MAIN}", *extra], cwd=work, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return work / "artifacts/tests/org_m_fixed_token_c128_o0/config"


def test_default_c_reaches_the_generated_config_through_make(tmp_path):
    work, env, _ = _make_env(tmp_path)
    cfg = _run_make(work, env)
    assert yaml.safe_load((cfg / "stage5.yaml").read_text())["labeling"]["qrels_file"] == "qrels_span_c090.jsonl"


def test_extra_params_set_c_through_make(tmp_path):
    work, env, _ = _make_env(tmp_path)
    cfg = _run_make(work, env, "EXTRA_PARAMS=--param labeling.cover_level=0.8")
    s5 = yaml.safe_load((cfg / "stage5.yaml").read_text())
    assert s5["labeling"] == {"method": "span", "cover_level": 0.8, "qrels_file": "qrels_span_c080.jsonl"}
    assert _qrels_kwarg(yaml.safe_load((cfg / "stage6.yaml").read_text())) == "qrels_span_c080.jsonl"

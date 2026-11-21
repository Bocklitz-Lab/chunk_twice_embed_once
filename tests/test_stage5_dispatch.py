"""Stage 5 runs the span script when `labeling.method: span`, the lexical script when `lexical`; c reaches the script
from the config, from EXTRA_PARAMS (Makefile) and from the command line."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_lib.dataset_pin import sha256_file  # noqa: E402
from test_make_span_qrels import _build, _jsonl  # noqa: E402

MAIN = ROOT / "configs" / "main_config.yaml"
REAL_DATA = ROOT / "temp_files" / "artifacts_old" / "chemquests_hf_db"


def _stage5_yaml(tmp_path, data_dir, main_edit=None, params=(), name="cfg"):
    """Generate stage5.yaml with the real split_config (from a copy of main_config when `main_edit` is given) and point it
    to the synthetic data."""
    src = MAIN
    if main_edit:
        cfg = yaml.safe_load(MAIN.read_text())
        main_edit(cfg)
        src = tmp_path / f"{name}_main.yaml"
        src.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = tmp_path / name
    cmd = [sys.executable, "-m", "pipeline_lib.split_config", "--input", str(src), "--output-dir", str(out), "--sections", "stage5"]
    for p in params:
        cmd += ["--param", p]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s5 = yaml.safe_load((out / "stage5.yaml").read_text())
    s5.update(qa_jsonl_file=str(data_dir / "qa.jsonl"), chunks_jsonl_file=str(data_dir / "chunks.jsonl"),
              full_text_jsonl_file=str(data_dir / "full_text.jsonl"), output_folder=str(data_dir / "mteb_task"),
              qa_sha256=sha256_file(data_dir / "qa.jsonl"))
    path = out / "stage5.yaml"
    path.write_text(yaml.safe_dump(s5, sort_keys=False))
    return path, s5


def _stage5(cfg_path, *extra, module="stages.stage5"):
    return subprocess.run([sys.executable, "-m", module, "--config", str(cfg_path), *extra], cwd=ROOT, capture_output=True, text=True)


def _files(d):
    return sorted(p.name for p in (d / "mteb_task").iterdir())


def test_span_method_runs_the_span_script_with_c_from_the_config(tmp_path):
    d = _build(tmp_path)
    cfg, s5 = _stage5_yaml(tmp_path, d, main_edit=lambda c: c["params"]["labeling"].update(cover_level=0.7))
    assert s5["labeling"]["cover_level"] == 0.7
    r = _stage5(cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "label method: span" in r.stdout
    files = _files(d)
    assert "qrels_span_c070.jsonl" in files and "label_stats_c070.json" in files and "overlap_table.jsonl" in files
    assert not [f for f in files if f.startswith("qrels_span") and f != "qrels_span_c070.jsonl"]
    assert "qrels_lexical.jsonl" not in files and "qrels.jsonl" not in files
    assert json.loads((d / "mteb_task/label_stats_c070.json").read_text())["cover_level"] == 0.7


def test_extra_param_style_override_reaches_the_script(tmp_path):
    d = _build(tmp_path)
    cfg, _ = _stage5_yaml(tmp_path, d, params=("labeling.cover_level=0.8",))
    r = _stage5(cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "qrels_span_c080.jsonl" in _files(d)
    assert json.loads((d / "mteb_task/label_stats_c080.json").read_text())["cover_level"] == 0.8


def test_command_line_cover_levels_override_the_config(tmp_path):
    d = _build(tmp_path)
    cfg, _ = _stage5_yaml(tmp_path, d)  # config says c from main_config
    r = _stage5(cfg, "--cover-level", "0.6", "0.75", module="stages.make_span_qrels")
    assert r.returncode == 0, r.stdout + r.stderr
    files = _files(d)
    assert "qrels_span_c060.jsonl" in files and "qrels_span_c075.jsonl" in files
    assert "qrels_span_c090.jsonl" not in files  # the config value was overridden, not added


def test_explicit_qrels_file_name_from_the_config_is_written(tmp_path):
    d = _build(tmp_path)
    cfg, _ = _stage5_yaml(tmp_path, d, params=("labeling.qrels_file=custom_labels.jsonl", "labeling.cover_level=0.8"))
    r = _stage5(cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    files = _files(d)
    assert "custom_labels.jsonl" in files and "qrels_span_c080.jsonl" in files
    assert (d / "mteb_task/custom_labels.jsonl").read_text() == (d / "mteb_task/qrels_span_c080.jsonl").read_text()


def test_no_cover_level_anywhere_is_an_error_not_a_silent_default(tmp_path):
    d = _build(tmp_path)
    cfg, s5 = _stage5_yaml(tmp_path, d)
    del s5["labeling"]["cover_level"]
    cfg.write_text(yaml.safe_dump(s5, sort_keys=False))
    r = _stage5(cfg)
    assert r.returncode != 0 and "no cover level" in r.stderr


def test_lexical_method_runs_the_lexical_script_and_writes_qrels_lexical(tmp_path):
    d = _build(tmp_path)
    cfg, _ = _stage5_yaml(tmp_path, d, params=("labeling.method=lexical",))
    r = _stage5(cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    files = _files(d)
    assert "qrels_lexical.jsonl" in files and "qrels.jsonl" in files
    assert not [f for f in files if f.startswith("qrels_span")]
    assert (d / "mteb_task/qrels_lexical.jsonl").read_text() == (d / "mteb_task/qrels.jsonl").read_text()
    rows = _jsonl(d / "mteb_task/qrels_lexical.jsonl")
    assert all(r["relevance"] in (0, 1) for r in rows)


def test_span_and_lexical_labels_live_side_by_side(tmp_path):
    d = _build(tmp_path)
    cfg_l, _ = _stage5_yaml(tmp_path, d, params=("labeling.method=lexical",), name="lex")
    cfg_s, _ = _stage5_yaml(tmp_path, d, name="span")
    assert _stage5(cfg_l).returncode == 0 and _stage5(cfg_s).returncode == 0
    files = _files(d)
    assert {"qrels_lexical.jsonl", "qrels_span_c090.jsonl", "queries.jsonl", "corpus.jsonl"} <= set(files)


def test_config_without_labeling_block_keeps_the_lexical_labels(tmp_path):
    d = _build(tmp_path)
    cfg, s5 = _stage5_yaml(tmp_path, d)
    del s5["labeling"]
    cfg.write_text(yaml.safe_dump(s5, sort_keys=False))
    r = _stage5(cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "qrels.jsonl" in _files(d)


def test_unknown_method_is_an_error(tmp_path):
    d = _build(tmp_path)
    cfg, s5 = _stage5_yaml(tmp_path, d)
    s5["labeling"]["method"] = "bm25"
    cfg.write_text(yaml.safe_dump(s5, sort_keys=False))
    r = _stage5(cfg)
    assert r.returncode != 0 and "labeling.method" in r.stderr


# ---- through the real Makefile: stage 4 and 6 are stubbed, split_config and stage 5 are real
@pytest.mark.skipif(not (REAL_DATA / "qa.jsonl").exists(), reason="pinned ChemQuests files not available locally")
def test_makefile_runs_stage5_with_c_from_extra_params_and_records_the_duration(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(ROOT / "Makefile", work / "Makefile")
    shutil.copy(ROOT / "test.mk", work / "test.mk")
    (work / "models.csv").write_text("dataset,name,revision\nd,org/m,rev1\n")
    # chunks: character windows of the real full text (exact positions), written by the stubbed stage 4
    fixture = tmp_path / "chunks.jsonl"
    with open(fixture, "w") as f, open(REAL_DATA / "full_text.jsonl") as ft:
        for line in ft:
            row = json.loads(line)
            text, k, a = row["text"], 0, 0
            while a < len(text):
                b = min(len(text), a + 800)
                f.write(json.dumps({"source_doc_id": str(row["corpus_id"]), "chunk_index": k, "chunk_id": f"{row['corpus_id']}:{k}",
                                    "text": text[a:b], "start_char": a, "end_char": b}) + "\n")
                k += 1
                if b == len(text):
                    break
                a = b - 100
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python"
    stub.write_text(
        '#!/bin/bash\n'
        'echo "$@" >> "$PYSTUB_LOG"\n'
        'if [[ "$*" == *stages.chunking* ]]; then cfg="${@: -1}"; out="$(dirname "$(dirname "$cfg")")"; '
        'mkdir -p "$out/chunks"; cp "$CHUNKS_FIXTURE" "$out/chunks/chunks.jsonl"; exit 0; fi\n'
        'if [[ "$*" == *stages.run_eval* ]]; then exit 0; fi\n'
        'exec "$REAL_PYTHON" "$@"\n'
    )
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "PYSTUB_LOG": str(log), "HOME": str(tmp_path), "REAL_PYTHON": sys.executable,
           "PYTHONPATH": str(ROOT), "CHUNKS_FIXTURE": str(fixture)}

    def run(*extra, size="128"):
        r = subprocess.run(["make", "stage4-6", "MODELS_CSV=models.csv", "CHUNKER=fixed_token", f"SIZE={size}", "OVERLAP=0",
                            f"INPUT_CONFIG={MAIN}", f"EXTRA_PARAMS=--param paths.chemquests_dir={REAL_DATA} {' '.join(extra)}"],
                           cwd=work, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return work / f"artifacts/tests/org_m_fixed_token_c{size}_o0"

    out = run("--param labeling.cover_level=0.8")
    task = out / "mteb_task"
    assert sorted(p.name for p in task.iterdir()) == sorted(
        ["queries.jsonl", "corpus.jsonl", "overlap_table.jsonl", "qrels_span_c080.jsonl", "label_stats_c080.json"])
    stats = json.loads((task / "label_stats_c080.json").read_text())
    assert stats["cover_level"] == 0.8 and stats["n_questions"] == 952 and stats["n_empty_questions"] == 0
    assert yaml.safe_load((out / "config/stage6.yaml").read_text())["tasks"][0]["kwargs"]["qrels_file"] == "qrels_span_c080.jsonl"
    # the stage 5 duration is recorded as before
    assert (out / "duration_stage5_s.txt").exists()
    durations = (work / "artifacts/tests/durations.csv").read_text().splitlines()
    assert any(",0,stage5," in line for line in durations)
    # default c from the config when EXTRA_PARAMS does not set it
    shutil.rmtree(out)
    out2 = run(size="128")
    assert (out2 / "mteb_task/qrels_span_c090.jsonl").exists()

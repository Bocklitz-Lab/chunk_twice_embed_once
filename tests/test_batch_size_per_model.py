"""B5: Phase 2 uses each model's own (Phase 1) batch size; models without one use params.model.batch_size."""
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from stages import rerun

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "configs" / "main_config.yaml")


def test_stage1_table_has_only_explicit_batch_sizes():
    t = rerun.stage1_batch_sizes(MAIN)
    assert t["BAAI/bge-small-en-v1.5"] == 64 and t["hkunlp/instructor-xl"] == 16 and t["intfloat/e5-large-v2"] == 32
    assert t["sentence-transformers/gtr-t5-xl"] == 8
    assert "sentence-transformers/all-mpnet-base-v2" not in t           # no explicit batch size in stage 1


def test_batch_size_rules():
    t = {"a/b": 16}
    assert rerun.batch_size_for("a/b", t, 32) == (16, "stage1.models")
    assert rerun.batch_size_for("c/d", t, 32) == (32, "params.model.batch_size")
    assert rerun.batch_size_for("a/b", t, 32, forced=8) == (8, "--param model.batch_size")
    assert rerun.forced_batch_size(["x=1", "model.batch_size=4"]) == 4 and rerun.forced_batch_size(["x=1"]) is None


def test_the_models_whose_batch_size_changes_follow_the_stage1_table():
    """Rule, no model list: the published Phase 2 used 32 for every model; a model is reported iff its stage 1 batch size differs from params.model.batch_size."""
    table = rerun.stage1_batch_sizes(MAIN)
    default = yaml.safe_load(Path(MAIN).read_text())["params"]["model"]["batch_size"]
    names = sorted(table) + ["sentence-transformers/all-mpnet-base-v2"]            # the last one has no explicit batch size
    ch = {c["model"]: (c["before"], c["now"]) for c in rerun.batch_size_changes(MAIN, names)}
    assert ch == {m: (default, b) for m, b in table.items() if b != default}
    assert "sentence-transformers/all-mpnet-base-v2" not in ch


def _prepared(tmp_path):
    prep = tmp_path / "prep"
    (prep / "chunks").mkdir(parents=True)
    (prep / "mteb_task").mkdir()
    (prep / "manifest.json").write_text(json.dumps({"key": "K", "files_sha256": {}, "timing_s": {}}))
    return prep


def _run_one(tmp_path, monkeypatch, model, params=(), table=None):
    monkeypatch.setattr(rerun.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=0))
    rerun.run_eval_one(model, "r" * 40, "fixed_token", 128, 0, _prepared(tmp_path), main_config=MAIN, artifacts_root=str(tmp_path / "art"),
                       params=list(params), force=False, dry_run=False, batch_sizes=rerun.stage1_batch_sizes(MAIN) if table is None else table)
    run_dir = tmp_path / "art" / "tests" / f"{rerun.model_dir(model)}_fixed_token_c128_o0"
    cfg = yaml.safe_load((run_dir / "config" / "stage6.yaml").read_text())
    return cfg, json.loads((run_dir / "run_info.json").read_text())


def test_stage6_config_and_run_info_carry_the_models_batch_size(tmp_path, monkeypatch):
    cfg, info = _run_one(tmp_path, monkeypatch, "hkunlp/instructor-xl")
    assert cfg["models"][0]["batch_size"] == 16 and info["batch_size"] == 16 and info["batch_size_source"] == "stage1.models"


def test_a_model_without_a_stage1_batch_size_uses_the_default(tmp_path, monkeypatch):
    cfg, info = _run_one(tmp_path, monkeypatch, "sentence-transformers/all-mpnet-base-v2")
    assert cfg["models"][0]["batch_size"] == 32 and info["batch_size_source"] == "params.model.batch_size"


def test_param_override_forces_one_batch_size_for_all(tmp_path, monkeypatch):
    cfg, info = _run_one(tmp_path, monkeypatch, "hkunlp/instructor-xl", params=["model.batch_size=4"])
    assert cfg["models"][0]["batch_size"] == 4 and info["batch_size_source"] == "--param model.batch_size"


def test_bm25_has_no_batch_size(tmp_path, monkeypatch):
    _, info = _run_one(tmp_path, monkeypatch, "bm25")
    assert info["batch_size"] is None and info["batch_size_source"] is None

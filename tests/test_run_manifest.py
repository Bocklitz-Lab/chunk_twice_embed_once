"""Run manifest, seeds and encode timing."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline_lib.model_wrapper import STNoPrompt
from pipeline_lib.run_manifest import (Stopwatch, build_manifest, git_info, package_versions, set_seeds,
                                       summarize_encode_timings, write_manifest)

ROOT = Path(__file__).resolve().parents[1]


def test_set_seeds_makes_python_and_numpy_repeatable():
    info = set_seeds(123)
    a = (random.random(), float(np.random.rand()))
    set_seeds(123)
    assert a == (random.random(), float(np.random.rand())) and info["seed"] == 123 and info["numpy"]


def test_manifest_has_the_required_fields(tmp_path):
    f = tmp_path / "corpus.jsonl"; f.write_text("x\n")
    cfg = tmp_path / "s6.yaml"; cfg.write_text("a: 1\n")
    m = build_manifest(repo_root=ROOT, config_path=cfg, resolved_config={"a": 1}, model={"name": "m"},
                       prompts={"query": "query: ", "passage": "passage: "}, use_prompts=True, batch_size=32,
                       seeds={"seed": 1}, data_files=[f], timing={"x": 1.0})
    for k in ("git", "packages", "hardware", "prompts", "use_prompts", "batch_size", "resolved_config", "seeds",
              "data_sha256", "timing_s", "config_sha256"):
        assert k in m
    assert m["git"]["commit"] and len(m["git"]["commit"]) == 40
    assert "gpus" in m["hardware"] and m["packages"]["numpy"]
    assert len(list(m["data_sha256"].values())[0]) == 64
    p = tmp_path / "out" / "run_manifest.json"
    write_manifest(p, m)
    assert json.loads(p.read_text())["prompts"]["query"] == "query: "


class _ST:
    def encode(self, s, **k):
        return np.ones((len(s), 3), dtype=np.float32)


def test_wrapper_records_encode_timings_per_side():
    w = STNoPrompt(_ST(), prompts={"query": "q: ", "passage": "p: "})
    w.encode(["a", "b"], prompt_type="query")
    w.encode(["c", "d", "e"], prompt_type="passage")
    s = summarize_encode_timings(w.timings)
    assert s["n_query_texts"] == 2 and s["n_passage_texts"] == 3 and s["n_calls"] == 2
    assert s["encode_query_s"] >= 0 and s["encode_passage_s"] >= 0


def test_stopwatch_sections_accumulate():
    sw = Stopwatch()
    with sw.section("a"):
        pass
    with sw.section("a"):
        pass
    assert set(sw.seconds) == {"a"}


def test_run_eval_writes_manifest_before_summary(tmp_path, monkeypatch):
    pytest.importorskip("mteb"); pytest.importorskip("sentence_transformers")
    from stages import run_eval

    class FakeMTEB:
        def __init__(self, tasks): pass
        def run(self, model, **kw):
            out = Path(kw["output_folder"]); (out / "x_default_predictions.json").write_text("{}")
            return []

    cfg = {"models": [{"name": "m/x", "revision": "r"}], "tasks": ["T"], "output_root": str(tmp_path / "res"),
           "seed": 7, "use_prompts": True,
           "model_prompts": {"m/x": {"query": "query: ", "passage": "passage: ", "card": "t"}}}
    p = tmp_path / "s6.yaml"; p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(run_eval, "MTEB", FakeMTEB)
    monkeypatch.setattr(run_eval, "SentenceTransformer", lambda *a, **k: _ST())
    monkeypatch.setattr(run_eval, "resolve_tasks", lambda specs, mods: [object()])
    monkeypatch.setattr(sys, "argv", ["run_eval", "--config", str(p)])
    run_eval.main()
    m = json.loads((tmp_path / "res" / "m__x_r" / "run_manifest.json").read_text())
    assert m["prompts"]["query"] == "query: " and m["use_prompts"] is True and m["seeds"]["seed"] == 7
    assert m["resolved_config"]["seed"] == 7 and "model_load_s" in m["timing_s"]
    assert (tmp_path / "res" / "summary.json").exists()

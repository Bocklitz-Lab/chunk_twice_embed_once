"""Stage 6: optional top_k (pilot) is passed to MTEB only when set; default behaviour is unchanged."""
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mteb")
pytest.importorskip("sentence_transformers")

from stages import run_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class _FakeMTEB:
    calls = []

    def __init__(self, tasks):
        pass

    def run(self, model, **kwargs):
        _FakeMTEB.calls.append(kwargs)
        out = Path(kwargs["output_folder"])
        (out / "x_default_predictions.json").write_text("{}")
        return []


def _run(tmp_path, monkeypatch, extra):
    cfg = {"models": [{"name": "m/x", "revision": "r"}], "tasks": ["T"], "output_root": str(tmp_path / "res"), **extra}
    p = tmp_path / "s6.yaml"
    p.write_text(yaml.safe_dump(cfg))
    _FakeMTEB.calls = []
    monkeypatch.setattr(run_eval, "MTEB", _FakeMTEB)
    monkeypatch.setattr(run_eval, "SentenceTransformer", lambda *a, **k: object())
    monkeypatch.setattr(run_eval, "STNoPrompt", lambda base, normalize: object())
    monkeypatch.setattr(run_eval, "resolve_tasks", lambda specs, mods: [object()])
    monkeypatch.setattr(sys, "argv", ["run_eval", "--config", str(p)])
    run_eval.main()
    return _FakeMTEB.calls[0]


def test_default_passes_no_top_k_and_saves_predictions(tmp_path, monkeypatch):
    kw = _run(tmp_path, monkeypatch, {})
    assert "top_k" not in kw and kw["save_predictions"] is True


def test_null_top_k_is_the_default(tmp_path, monkeypatch):
    assert "top_k" not in _run(tmp_path, monkeypatch, {"top_k": None})


def test_top_k_100_is_passed_with_save_predictions(tmp_path, monkeypatch):
    kw = _run(tmp_path, monkeypatch, {"top_k": 100})
    assert kw["top_k"] == 100 and kw["save_predictions"] is True


def test_invalid_top_k_rejected():
    with pytest.raises(ValueError):
        run_eval.parse_app_config({"models": [{"name": "m"}], "tasks": ["T"], "top_k": 0})


def test_generated_stage6_carries_top_k_from_param(tmp_path):
    from pipeline_lib import split_config
    for extra, expected in [([], None), (["--param", "eval.top_k=100"], 100)]:
        out = tmp_path / str(expected)
        assert split_config.main(["--input", str(ROOT / "configs/main_config.yaml"), "--output-dir", str(out),
                                  "--sections", "stage6", "--param", "model.name=intfloat/e5-large-v2",
                                  "--param", "model.dir=intfloat_e5_large_v2", *extra]) == 0
        assert yaml.safe_load((out / "stage6.yaml").read_text())["top_k"] == expected

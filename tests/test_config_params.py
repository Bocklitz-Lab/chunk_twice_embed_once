"""Tool defaults come from the main config: the reference tokenizer of the efficiency module is the one of ``params.chunking_tokenizer``."""
from pathlib import Path

import pytest
import yaml

from pipeline_lib import config_params, efficiency


def test_reference_tokenizer_is_read_from_the_config():
    p = yaml.safe_load(config_params.DEFAULT_MAIN_CONFIG.read_text())["params"]
    assert config_params.reference_tokenizer() == (p["chunking_tokenizer"], p["chunking_tokenizer_revision"])
    assert not hasattr(efficiency, "REF_TOKENIZER") and not hasattr(efficiency, "REF_TOKENIZER_REVISION")


def test_efficiency_uses_the_configured_tokenizer(tmp_path, monkeypatch):
    cfg = tmp_path / "main.yaml"
    cfg.write_text(yaml.safe_dump({"params": {"chunking_tokenizer": "org/other-tok", "chunking_tokenizer_revision": "a" * 40}}))
    monkeypatch.setattr(config_params, "DEFAULT_MAIN_CONFIG", cfg)
    seen = {}

    class Tok:
        def __call__(self, texts, add_special_tokens, truncation):
            seen["special"] = add_special_tokens
            return {"input_ids": [[0] * len(t.split()) for t in texts]}

    monkeypatch.setattr(efficiency, "load_tokenizer", lambda name, revision=None: seen.update(name=name, revision=revision) or Tok())
    assert efficiency.ref_token_lengths(["a b c", "d"]) == [3, 1]
    assert seen == {"name": "org/other-tok", "revision": "a" * 40, "special": False}


def test_legacy_and_unpinned_settings_are_refused(tmp_path):
    for params in ({"chunking_tokenizer": "model"}, {"chunking_tokenizer": "bert-base-uncased"}):
        f = tmp_path / f"{len(params)}_{params['chunking_tokenizer']}.yaml"
        f.write_text(yaml.safe_dump({"params": params}))
        with pytest.raises(ValueError):
            config_params.reference_tokenizer(f)


def test_section_returns_a_copy():
    a = config_params.section("report")
    a["resamples"] = -1
    assert config_params.section("report")["resamples"] != -1

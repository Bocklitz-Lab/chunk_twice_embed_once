"""All 41 Phase 1 models are pinned to a commit hash; the rerun resolves a moving revision from that table."""
import re
from pathlib import Path

import pytest
import yaml

from stages import rerun

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs" / "main_config.yaml"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def test_every_phase1_model_has_a_commit_hash_in_both_configs():
    for path, key in [(MAIN, "stage1"), (ROOT / "configs" / "stages" / "stage1.yaml", None)]:
        doc = yaml.safe_load(path.read_text())
        models = doc[key]["models"] if key else doc["models"]
        assert len(models) == 41, path
        assert [m["name"] for m in models if not HEX40.match(m["revision"])] == [], path


def test_prompt_table_and_pins_cover_the_same_models():
    doc = yaml.safe_load(MAIN.read_text())
    assert set(doc["params"]["model_prompts"]) == {m["name"] for m in doc["stage1"]["models"]}


def test_both_configs_pin_the_same_commits():
    a = {m["name"]: m["revision"] for m in yaml.safe_load(MAIN.read_text())["stage1"]["models"]}
    b = {m["name"]: m["revision"] for m in yaml.safe_load((ROOT / "configs/stages/stage1.yaml").read_text())["models"]}
    assert a == b


def test_pilot_models_use_the_pinned_commits():
    table = rerun.pinned_revisions(str(MAIN))
    for spec in yaml.safe_load(MAIN.read_text())["params"]["pilot"]["models"]:
        n, r = rerun.parse_model(spec)
        if n != "bm25":
            assert r == table[n], n


def test_resolve_revision():
    t = {"a/b": "1" * 40, "c/d": "main"}
    assert rerun.resolve_revision("a/b", "main", t) == "1" * 40
    assert rerun.resolve_revision("a/b", "2" * 40, t) == "2" * 40            # an explicit commit wins
    assert rerun.resolve_revision("bm25", "rank_bm25", t) == "rank_bm25"
    for name in ("x/y", "c/d"):
        with pytest.raises(ValueError, match="no pinned revision"):
            rerun.resolve_revision(name, "main", t)

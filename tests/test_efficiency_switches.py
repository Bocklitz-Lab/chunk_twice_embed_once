"""B4: efficiency is ON by default in the Makefile, and its latency subset comes from params.efficiency."""
import argparse
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from pipeline_lib import efficiency as E, grid
from stages import rerun

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "main_config.yaml").read_text())["params"]["efficiency"]


def ns(**kw):
    base = dict(efficiency=False, latency=False, latency_queries=None, latency_warmup=None, latency_device=None, latency_chunkers=[],
                latency_sizes=[], latency_overlaps=[], latency_models=[], latency_cells=[])
    base.update(kw)
    return argparse.Namespace(**base)


def test_config_block_matches_the_requested_subset():
    cells = {(c["chunker"], c["size"], c["overlap"]) for c in CFG["latency_cells"]}
    assert cells == {("fixed_token", 128, 0), ("fixed_token", 256, 0), ("fixed_token", 512, 0), ("hierarchical_section", 512, 0)}
    assert CFG["latency_queries"] == 200 and CFG["latency_device"] == "cpu" and CFG["latency_models"] == [] and CFG["latency_warmup"] == 10
    for ch, s, o in cells:
        assert ch in grid.CHUNKERS and grid.is_valid(s, o), (ch, s, o)           # every configured cell exists in the published grid


def test_makefile_switches_are_on_by_default_and_reach_the_commands():
    mk = (ROOT / "Makefile").read_text()
    assert re.search(r"^STAGE1_ARGS \?= --efficiency$", mk, re.M) and re.search(r"^EFFICIENCY_ARGS \?= --efficiency --latency$", mk, re.M)
    s1 = subprocess.run(["make", "-n", "-C", str(ROOT), "stage1"], capture_output=True, text=True).stdout
    assert "stages.model_screening --config configs/stages/stage1.yaml --efficiency" in s1 and "--latency" not in s1
    for target in ("eval", "rerun"):
        out = subprocess.run(["make", "-n", "-C", str(ROOT), target, f"MODELS_CSV={ROOT / 'README.md'}"], capture_output=True, text=True).stdout
        assert "--efficiency --latency" in out, target
    off = subprocess.run(["make", "-n", "-C", str(ROOT), "eval", f"MODELS_CSV={ROOT / 'README.md'}", "EFFICIENCY_ARGS="], capture_output=True, text=True).stdout
    assert "--efficiency" not in off


def test_flags_off_means_nothing_is_passed():
    assert rerun.efficiency_args(ns(), "art", CFG) == []


def test_latency_flags_come_from_the_config():
    out = rerun.efficiency_args(ns(efficiency=True, latency=True), "art", CFG)
    assert out[:3] == ["--efficiency", "--efficiency-out", str(Path("art") / "tests")]
    assert out[out.index("--latency-queries") + 1] == "200" and out[out.index("--latency-warmup") + 1] == "10"
    assert out[out.index("--latency-device") + 1] == "cpu"
    i = out.index("--latency-cells")
    assert out[i + 1:i + 5] == ["fixed_token:128:0", "fixed_token:256:0", "fixed_token:512:0", "hierarchical_section:512:0"]
    assert "--latency-models" not in out and "--latency-sizes" not in out


def test_command_line_flags_win_over_the_config():
    out = rerun.efficiency_args(ns(efficiency=True, latency=True, latency_queries=0, latency_device="cuda", latency_sizes=[256]), "art", CFG)
    assert out[out.index("--latency-queries") + 1] == "0" and out[out.index("--latency-device") + 1] == "cuda"
    assert "--latency-sizes" in out and "--latency-cells" not in out           # an explicit subset replaces the configured one
    only_eff = rerun.efficiency_args(ns(efficiency=True), "art", CFG)
    assert "--latency" not in only_eff and "--latency-cells" not in only_eff


def test_library_defaults_without_a_config_block():
    out = rerun.efficiency_args(ns(latency=True), "art", None)
    assert out[out.index("--latency-queries") + 1] == "200" and out[out.index("--latency-device") + 1] == "cpu" and "--latency-cells" not in out


def test_parse_cells_and_selection():
    assert E.parse_cells(["fixed_token:128:0", "hierarchical_section:512:0"]) == [("fixed_token", 128, 0), ("hierarchical_section", 512, 0)]
    for bad in ("fixed_token:128", "fixed_token:x:0", ":1:2", "a:1:2:3"):
        with pytest.raises(ValueError):
            E.parse_cells([bad])
    o = E.EfficiencyOptions(enabled=True, latency=True, latency_cells=[("fixed_token", 128, 0)], chunker="fixed_token", size=128, overlap=0)
    assert o.latency_selected("any/model")
    o.size = 256
    assert not o.latency_selected("any/model")
    o.size, o.latency_models = 128, ["bge"]
    assert not o.latency_selected("intfloat/e5-large-v2") and o.latency_selected("BAAI/bge-m3")      # cells AND models
    none = E.EfficiencyOptions(enabled=True, latency=True, chunker="x", size=1, overlap=0)
    assert none.latency_selected("m")                                                               # no cells: no restriction


def test_cli_option_parsing_roundtrip():
    ap = argparse.ArgumentParser()
    E.add_cli_args(ap)
    o = E.options_from_args(ap.parse_args(["--efficiency", "--latency", "--latency-cells", "fixed_token:512:0", "--eff-chunker", "fixed_token",
                                           "--eff-size", "512", "--eff-overlap", "0"]))
    assert o.enabled and o.latency and o.latency_cells == [("fixed_token", 512, 0)] and o.latency_selected("m")

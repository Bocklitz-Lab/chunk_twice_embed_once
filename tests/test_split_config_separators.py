"""B4: generated stage YAMLs keep real newlines in string parameters."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_lib import split_config  # noqa: E402

EXPECTED_SEPARATORS = ["\n\n", "\n", ".", "?", "!", " ", ""]


def _dump_load(obj):
    text = yaml.dump(obj, Dumper=split_config.EscapingDumper, sort_keys=False, allow_unicode=True,
                     default_flow_style=False, width=4096)
    return yaml.safe_load(text), text


def test_strings_with_control_characters_round_trip_exactly():
    tricky = [
        "\n\n", "\n", "\r\n", "\t", " \n ", "a\nb", "it's\nquoted", 'say "hi"\n', "back\\slash\\n literal",
        "\x0b\x0c", "\x85", " ", " ", "ünï\ncode", "", " ", ".", "^\\s*graphical\\s+abstract\\s*$",
        "line1\nline2: with colon", "- dash\n", "#hash\n", "'", '"',
    ]
    got, text = _dump_load({"separators": tricky, "nested": {"x": tricky[:5]}})
    assert got["separators"] == tricky
    assert got["nested"]["x"] == tricky[:5]
    # the real newline is written as a YAML escape, never as backslash + n inside single quotes
    assert "'\\n\\n'" not in text


def test_a_literal_backslash_n_stays_literal():
    got, _ = _dump_load({"s": ["\\n\\n", "\\n"]})
    assert got["s"] == ["\\n\\n", "\\n"]
    assert got["s"] != ["\n\n", "\n"]


def _generate(tmp_path, chunker, extra=()):
    out = tmp_path / chunker
    rc = split_config.main([
        "--input", str(ROOT / "configs/main_config.yaml"), "--output-dir", str(out),
        "--sections", "stage4", "stage5", "stage6",
        "--param", "model.name=intfloat/e5-large-v2", "--param", "model.dir=intfloat_e5_large_v2",
        "--param", f"chunk.chunker={chunker}", "--param", "chunk.size=128", "--param", "chunk.overlap=32",
        *extra,
    ])
    assert rc == 0
    return out


def test_generated_recursive_stage4_has_the_exact_separators_of_main_config(tmp_path):
    main_cfg = yaml.safe_load((ROOT / "configs/main_config.yaml").read_text())
    from_main = main_cfg["params"]["strategy_presets"]["recursive_token"]["params"]["separators"]
    assert from_main == EXPECTED_SEPARATORS          # the source of truth has real newlines

    out = _generate(tmp_path, "recursive_token")
    stage4 = yaml.safe_load((out / "stage4.yaml").read_text())
    assert stage4["strategy"]["params"]["separators"] == from_main
    assert stage4["strategy"]["params"]["separators"][0] == "\n\n"
    assert "\\n" not in "".join(stage4["strategy"]["params"]["separators"])  # no literal backslash-n


@pytest.mark.parametrize("chunker", ["fixed_token", "recursive_token", "semantic_fixed", "semantic_recursive",
                                     "hierarchical_section"])
def test_every_generated_config_equals_the_resolved_input_exactly(tmp_path, chunker):
    """Strong form: for every section and every string in it (regexes, separators, ...), the file
    loads back to exactly what split_config resolved in memory."""
    out = _generate(tmp_path, chunker)
    # resolve in memory the same way main() does
    doc = yaml.safe_load((ROOT / "configs/main_config.yaml").read_text())
    for k, v in [("model.name", "intfloat/e5-large-v2"), ("model.dir", "intfloat_e5_large_v2"),
                 ("chunk.chunker", chunker), ("chunk.size", 128), ("chunk.overlap", 32)]:
        split_config.deep_set(doc["params"], k, v)
    split_config.resolve_chunking_block(doc)  # reference tokenizer / breakpoint model, as main() does
    split_config.resolve_labeling_block(doc)  # derives labeling.qrels_file, as main() does
    doc = split_config.interpolate(doc, ctx=doc)
    split_config.resolve_stage4_strategy(doc)
    doc = split_config.interpolate(doc, ctx=doc)
    for sec in ("stage4", "stage5", "stage6"):
        expected = split_config.interpolate(doc[sec], ctx=doc)
        assert yaml.safe_load((out / f"{sec}.yaml").read_text()) == expected, (chunker, sec)


def test_no_string_in_main_config_is_changed_by_the_dumper():
    """Every string anywhere in main_config.yaml (including control characters) survives a dump/load."""
    doc = yaml.safe_load((ROOT / "configs/main_config.yaml").read_text())
    got, _ = _dump_load(doc)
    assert got == doc


def test_main_config_strings_with_control_characters_are_only_the_separators():
    """Documents which string parameters contain real control characters (the B4 audit)."""
    doc = yaml.safe_load((ROOT / "configs/main_config.yaml").read_text())
    found = []

    def walk(x, path):
        if isinstance(x, str):
            if any(c in x for c in "\n\r\t"):
                found.append(path)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")

    walk(doc, "")
    assert found == [".params.strategy_presets.recursive_token.params.separators[0]",
                     ".params.strategy_presets.recursive_token.params.separators[1]"]

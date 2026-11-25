"""Control strategy fixed_token_nomatter: fixed-token windows on the text that hierarchical-section keeps (same end-matter removal, no second implementation)."""
import json
from pathlib import Path

import pytest
import yaml

from pipeline_lib import grid, split_config
from pipeline_lib.fixed_token_chunker import FixedTokenChunker
from tests.test_hierarchical_tail import _chunker, _doc

ROOT = Path(__file__).resolve().parents[1]
HIER = dict(model_name="bert-base-uncased", chunk_size=40, chunk_overlap=0, use_latex=False, use_numbered_headings=False, use_common_chem_headings=False,
            use_generic_heading_heuristics=False, join_short_sections=False, drop_from_first_match=True,
            custom_heading_regexes=[r"^\s*corresponding\s+author\s*$", r"^\s*introduction\s*$", r"^\s*results\s*$", r"^\s*references?\s*$"],
            drop_tail_sections_matching=["references", "corresponding author"])


def _nomatter(size=40, overlap=0):
    return FixedTokenChunker(model_name="bert-base-uncased", chunk_size=size, chunk_overlap=overlap, drop_end_matter=True, end_matter_config=HIER)


def test_the_cut_is_where_hierarchical_section_stops():
    text = _doc(with_references=True)
    h = _chunker()
    cut = h.end_matter_cut(text)
    assert text[cut:].lstrip().startswith("References") and cut < len(text)
    kept = h.split_text_with_offsets(text)
    last = max(e for _, _, e in kept)
    assert last <= cut and text[last:cut].strip() == ""           # hierarchical keeps text[:cut] (up to trailing whitespace)
    assert min(s for _, s, _ in kept) == 0


def test_nomatter_covers_exactly_the_hierarchical_text_with_exact_positions():
    text = _doc(with_references=True)
    cut = _chunker().end_matter_cut(text)
    triples = _nomatter().split_text_with_offsets(text)
    assert triples and all(text[s:e] == c for c, s, e in triples)               # native positions stay exact
    assert max(e for _, _, e in triples) <= cut and "Some reference" not in "\n".join(c for c, _, _ in triples)
    assert "Result sentence 59 " in "\n".join(c for c, _, _ in triples)
    assert max(e for _, _, e in triples) >= cut - 3                           # up to whitespace trimming: the same end
    plain = FixedTokenChunker(model_name="bert-base-uncased", chunk_size=40, chunk_overlap=0).split_text_with_offsets(text)
    assert max(e for _, _, e in plain) > cut and any("Some reference" in c for c, _, _ in plain)       # the plain control keeps the end matter


def test_without_end_matter_or_headings_nothing_is_cut():
    no_refs = _doc(with_references=False)
    assert _chunker().end_matter_cut(no_refs) == len(no_refs)
    assert _chunker().end_matter_cut("no headings at all, just words " * 20) == len("no headings at all, just words " * 20)
    plain = FixedTokenChunker(model_name="bert-base-uncased", chunk_size=40, chunk_overlap=0).split_text_with_offsets(no_refs)
    assert [(c, s, e) for c, s, e in _nomatter().split_text_with_offsets(no_refs)] == plain


def test_drop_end_matter_needs_the_hierarchical_parameters():
    with pytest.raises(ValueError, match="end_matter_config"):
        FixedTokenChunker(model_name="bert-base-uncased", chunk_size=40, chunk_overlap=0, drop_end_matter=True)


def test_the_preset_reuses_the_hierarchical_parameters_and_the_control_runs_at_three_cells(tmp_path):
    rc = split_config.main(["--input", str(ROOT / "configs/main_config.yaml"), "--output-dir", str(tmp_path), "--sections", "stage4",
                            "--param", "chunk.chunker=fixed_token_nomatter", "--param", "chunk.size=256", "--param", "chunk.overlap=0"])
    assert rc == 0
    st = yaml.safe_load((tmp_path / "stage4.yaml").read_text())["strategy"]
    assert st["name"] == "fixed_token" and st["params"]["drop_end_matter"] is True
    hier = yaml.safe_load((ROOT / "configs/main_config.yaml").read_text())["params"]["strategy_presets"]["hierarchical_section"]["params"]
    assert st["params"]["end_matter_config"]["drop_tail_sections_matching"] == hier["drop_tail_sections_matching"]       # one list, two strategies
    assert st["params"]["end_matter_config"]["custom_heading_regexes"] == hier["custom_heading_regexes"]
    assert grid.CONTROL_CELLS == {"fixed_token_nomatter": [(256, 0), (384, 0), (512, 0)]} and "fixed_token_nomatter" in grid.ALL_CHUNKERS and "fixed_token_nomatter" not in grid.CHUNKERS
    full = grid.grid()
    assert len(full) == 28 and grid.cells_for("fixed_token", full) == full and grid.cells_for("fixed_token_nomatter", full) == [(256, 0), (384, 0), (512, 0)]
    assert len(grid.CHUNKERS) * len(full) + sum(len(grid.cells_for(c, full)) for c in grid.CONTROL_CELLS) == 143          # 140 + 3 configurations per model


def test_run_folder_names_of_the_control_are_parsed():
    from tools import check_identical
    m = check_identical._RUN.match("org_m_fixed_token_nomatter_c256_o0")
    assert m and m.group("chunker") == "fixed_token_nomatter" and m.group("model") == "org_m"
    m = check_identical._RUN.match("org_m_fixed_token_c256_o0")
    assert m and m.group("chunker") == "fixed_token"

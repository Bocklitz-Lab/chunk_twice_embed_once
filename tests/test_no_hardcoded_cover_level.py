"""The cover level c is a parameter: no literal 0.9 / 90 (or 9/10) in the labelling or rescoring code.
The only place for the default is configs/main_config.yaml (labeling.cover_level)."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LABELLING_AND_RESCORING_CODE = [
    "pipeline_lib/span_labels.py",
    # added by the later commits of this branch:
    "stages/make_span_qrels.py",
    "stages/stage5.py",
    "tools/rescore.py",
]

# 0.9, .9, 0.90, 90, 90.0, 9/10, 9, 10 (as in Fraction(9, 10)), 0.90 in comments and strings too
PATTERNS = [
    re.compile(r"(?<![\w.])0?\.9(?![\w])"),
    re.compile(r"(?<![\w.])0?\.90+(?![\w])"),
    re.compile(r"(?<![\w.])90(?:\.0*)?(?![\w.])"),
    re.compile(r"(?<![\w.])9\s*/\s*10(?![\w.])"),
    re.compile(r"\(\s*9\s*,\s*10\s*\)"),
    re.compile(r"(?<![\w.])9\s*,\s*10(?![\w.])"),
]


def _hits(text):
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for p in PATTERNS:
            if p.search(line):
                out.append((n, line.strip()))
                break
    return out


@pytest.mark.parametrize("rel", LABELLING_AND_RESCORING_CODE)
def test_no_literal_cover_level_in_code(rel):
    path = ROOT / rel
    assert path.exists(), f"{rel} should exist"
    assert _hits(path.read_text(encoding="utf-8")) == []


def test_the_checker_itself_finds_the_forbidden_literals():
    for bad in ("c = 0.9", "x = .9", "c=0.90", "thr = 90", "Fraction(9, 10)", "9/10", "c = 90.0", "#  at 0.9 of S"):
        assert _hits(bad), bad
    for ok in ("c = 0.95", "x = 190", "tag = 'c090'", "0.99", "v0.9.1", "p9", "n = 900", "1.90"):
        assert not _hits(ok), ok


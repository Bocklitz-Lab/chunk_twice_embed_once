"""Session guard: no test may write into the repository's tracked ``results/`` folder.

Earlier versions of the report tests wrote their output (artifacts root under ``/tmp/pytest-of-...``) into
``results/v2-revision/``, and that output was committed by mistake. Tests must use ``tmp_path``. This fixture hashes every
file below ``results/`` before the first test and fails the session at the end if a file was added, removed or changed.
"""
import hashlib
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _snapshot():
    if not RESULTS.exists():
        return {}
    return {str(p.relative_to(RESULTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RESULTS.rglob("*")) if p.is_file()}


@pytest.fixture(scope="session", autouse=True)
def _results_folder_is_not_written():
    before = _snapshot()
    yield
    after = _snapshot()
    changed = sorted(set(before) ^ set(after)) + sorted(k for k in before if k in after and before[k] != after[k])
    assert not changed, f"a test wrote into the tracked results/ folder (use tmp_path): {changed[:10]}"

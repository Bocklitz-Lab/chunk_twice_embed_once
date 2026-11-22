"""Numerical agreement with the released InstructorEmbedding package, which lives in a SEPARATE virtual environment
(it needs sentence-transformers 2.2.x). Skipped unless INSTRUCTOR_OFFICIAL_PYTHON points to that environment's python
(see tools/verify_instructor.py for how to build it), a prepared folder is given in INSTRUCTOR_VERIFY_PREPARED and
hkunlp/instructor-xl is in the local Hugging Face cache.

  INSTRUCTOR_OFFICIAL_PYTHON=/path/venv_instructor/bin/python INSTRUCTOR_VERIFY_PREPARED=artifacts/.../prepared/<folder> \\
      pytest tests/test_instruction_masking_official.py -q
"""
import os

import pytest

from tools import verify_instructor as V

PY, PREP = os.environ.get("INSTRUCTOR_OFFICIAL_PYTHON"), os.environ.get("INSTRUCTOR_VERIFY_PREPARED")
pytestmark = pytest.mark.skipif(not (PY and PREP and os.path.isdir(V.snapshot_dir())),
                                reason="needs INSTRUCTOR_OFFICIAL_PYTHON, INSTRUCTOR_VERIFY_PREPARED and the cached model")


def test_same_embeddings_as_the_official_class(tmp_path):
    out = tmp_path / "report.json"
    assert V.main(["--official-python", PY, "--prepared", PREP, "--out", str(out)]) == 0

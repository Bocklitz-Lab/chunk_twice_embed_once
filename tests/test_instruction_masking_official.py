"""Numerical agreement with the official INSTRUCTOR class. Skipped unless ``InstructorEmbedding`` is installed AND the model is cached
(it is not a dependency of this repository; the masking is implemented in pipeline_lib/instruction_masking.py without it).

  pip install InstructorEmbedding      # in a throw-away environment
  pytest tests/test_instruction_masking_official.py -q
"""
import os

import numpy as np
import pytest

pytest.importorskip("InstructorEmbedding")
pytest.importorskip("sentence_transformers")
REV = "ce48b213095e647a6c3536364b9fa00daf57f436"
CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--hkunlp--instructor-xl/snapshots/" + REV)
pytestmark = pytest.mark.skipif(not os.path.isdir(CACHE), reason="hkunlp/instructor-xl is not in the local Hugging Face cache")

from pipeline_lib import instruction_masking as IM

DOC = "Represent the Wikipedia document for retrieval: "
TEXTS = ["Furfural yields ranged from 7.5% to 10% in sugar mills.", "The reaction was carried out at 150 C for 2 h in THF/H2O.", "x"]


def test_same_embeddings_as_the_official_class():
    import torch
    from InstructorEmbedding import INSTRUCTOR
    from sentence_transformers import SentenceTransformer
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    official = INSTRUCTOR(CACHE, device=dev)
    ref = official.encode([[DOC, t] for t in TEXTS], batch_size=4, normalize_embeddings=True)
    del official
    mine = IM.encode_masked(SentenceTransformer(CACHE, device=dev), DOC, TEXTS, batch_size=4)
    assert (mine * ref).sum(1).min() > 0.999
    np.testing.assert_allclose(mine, ref, atol=2e-3)

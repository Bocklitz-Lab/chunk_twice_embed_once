"""Model prompts: config table, the prefix really reaches the embedding model (queries and documents), and
nothing else (chunking stage config) changes with use_prompts."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline_lib.model_wrapper import STNoPrompt
from pipeline_lib.prompts import PromptConfigError, apply_prompt, resolve_prompts

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "main_config.yaml").read_text())
TABLE = CFG["params"]["model_prompts"]


class FakeST:
    """Records exactly what the embedding model is asked to embed."""
    name = "fake"

    def __init__(self):
        self.seen = []

    def encode(self, sentences, **kw):
        self.seen.extend(sentences)
        out = []
        for s in sentences:
            h = hashlib.sha256(s.encode()).digest()
            out.append(np.frombuffer(h, dtype=np.uint8)[:16].astype(np.float32) + 1.0)
        return np.stack(out)


def _run_retrieval(model_name, use_prompts=True):
    pytest.importorskip("mteb")
    from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator

    fake = FakeST()
    wrapper = STNoPrompt(fake, normalize=True, prompts=resolve_prompts(model_name, TABLE, use_prompts))
    corpus = {"c1": {"title": "", "text": "alpha chunk"}, "c2": {"title": "", "text": "beta chunk"}}
    queries = {"q1": "what is alpha", "q2": "what is beta"}
    RetrievalEvaluator(wrapper, task_name="T", encode_kwargs={"batch_size": 2})(corpus, queries)
    return fake.seen


@pytest.mark.parametrize("model,qp,dp", [
    ("intfloat/e5-large-v2", "query: ", "passage: "),
    ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: "),
])
def test_prefix_reaches_the_model_for_queries_and_documents(model, qp, dp):
    seen = _run_retrieval(model)
    queries = [s for s in seen if "what is" in s]
    docs = [s for s in seen if "chunk" in s]
    assert len(queries) == 2 and len(docs) == 2
    assert all(s.startswith(qp) and not s.startswith(qp + qp) for s in queries)
    assert all(s.startswith(dp) for s in docs)
    assert not any(s.startswith(dp) for s in queries) and not any(s.startswith(qp) for s in docs)


def test_use_prompts_false_reproduces_the_old_behaviour():
    seen = _run_retrieval("intfloat/e5-large-v2", use_prompts=False)
    assert sorted(seen) == sorted(["what is alpha", "what is beta", "alpha chunk", "beta chunk"])


def test_bge_instruction_on_queries_only_and_no_prompt_model_unchanged():
    seen = _run_retrieval("BAAI/bge-large-en-v1.5")
    assert all(s.startswith("Represent this sentence for searching relevant passages: ") for s in seen if "what is" in s)
    assert sorted(s for s in seen if "chunk" in s) == ["alpha chunk", "beta chunk"]
    assert sorted(_run_retrieval("BAAI/bge-m3")) == sorted(["what is alpha", "what is beta", "alpha chunk", "beta chunk"])


def test_every_phase1_model_has_an_entry_and_19_need_prompts():
    names = [m["name"] for m in CFG["stage1"]["models"]]
    assert len(names) == 42 and set(names) == set(TABLE)
    assert sum(1 for v in TABLE.values() if v["query"] or v["passage"]) == 19
    for v in TABLE.values():
        assert set(v) == {"query", "passage", "card"} and v["card"]


def test_resolve_prompts_rules():
    assert resolve_prompts("intfloat/e5-large-v2", TABLE, False) is None
    with pytest.raises(PromptConfigError):
        resolve_prompts("unknown/model", TABLE, True)
    assert resolve_prompts("unknown/model", TABLE, True, override={"query": "q: "})["query"] == "q: "
    assert apply_prompt(["a"], {"query": "q: ", "passage": ""}, "passage") == ["a"]
    assert apply_prompt("a", {"query": "q: ", "passage": ""}, "query") == "q: a"


def _split(tmp_path, *params):
    out = tmp_path / "cfg"
    cmd = [sys.executable, "-m", "pipeline_lib.split_config", "--input", str(ROOT / "configs/main_config.yaml"),
           "--output-dir", str(out), "--sections", "stage1", "stage4", "stage6"]
    for p in params:
        cmd += ["--param", p]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return {s: yaml.safe_load((out / f"{s}.yaml").read_text()) for s in ("stage1", "stage4", "stage6")}


def test_prompts_reach_the_stage_files_and_never_the_chunking_stage(tmp_path):
    on = _split(tmp_path / "on")
    off = _split(tmp_path / "off", "use_prompts=false")
    assert on["stage6"]["use_prompts"] is True and off["stage6"]["use_prompts"] is False
    assert on["stage6"]["model_prompts"]["intfloat/e5-large-v2"]["passage"] == "passage: "
    assert on["stage1"]["model_prompts"] == TABLE
    assert on["stage4"] == off["stage4"]                 # chunking does not depend on prompts
    assert "prompt" not in json.dumps(on["stage4"]).lower()

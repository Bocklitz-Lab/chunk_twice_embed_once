"""B1: what a run manifest records about the machine and the model (CPU, RAM, threads, pooling, dimension, maximum length)."""
import json
from pathlib import Path

import pytest

from pipeline_lib import model_info, run_manifest
from tools import hybrid, rerank
from test_baselines import overlap_scorer, world  # noqa: F401  (fixture)

CPUINFO = """processor\t: 0
model name\t: AMD Ryzen Threadripper 3960X 24-Core Processor
physical id\t: 0
core id\t: 0

processor\t: 1
model name\t: AMD Ryzen Threadripper 3960X 24-Core Processor
physical id\t: 0
core id\t: 0

processor\t: 2
model name\t: AMD Ryzen Threadripper 3960X 24-Core Processor
physical id\t: 0
core id\t: 1
"""


def test_cpu_info_reads_proc_files(tmp_path):
    (tmp_path / "cpuinfo").write_text(CPUINFO)
    (tmp_path / "meminfo").write_text("MemTotal:       33554432 kB\nMemFree: 1 kB\n")
    c = run_manifest.cpu_info(str(tmp_path / "cpuinfo"), str(tmp_path / "meminfo"))
    assert c["cpu_model"].startswith("AMD Ryzen") and c["cpu_physical_cores"] == 2 and c["ram_total_gb"] == 32.0 and c["cpu_logical_cores"]


def test_cpu_info_survives_missing_files(tmp_path):
    c = run_manifest.cpu_info(str(tmp_path / "none"), str(tmp_path / "none2"))
    assert c["cpu_model"] is None and c["ram_total_gb"] is None and c["cpu_physical_cores"] is None


def test_hardware_info_has_cpu_ram_threads_and_one_driver_version(monkeypatch):
    monkeypatch.setattr(run_manifest, "_run", lambda cmd, cwd=None: "610.88\n610.88")
    h = run_manifest.hardware_info()
    for k in ("cpu_model", "cpu_logical_cores", "cpu_physical_cores", "ram_total_gb", "gpus", "python", "platform"):
        assert k in h
    assert h["torch_threads"] >= 1 and h["nvidia_driver"] == "610.88"


# ---- model properties -------------------------------------------------------------------------------------
class Transformer: pass
class Normalize: pass
class Dense: pass


class Pooling:
    def __init__(self, **flags):
        for k in model_info.POOLING_FLAGS:
            setattr(self, f"pooling_mode_{k}", bool(flags.get(k)))


class FakeTok:
    name_or_path = "org/tok"


class FakeST:
    def __init__(self, mods, dim=1024, msl=512):
        self._modules = {str(i): m for i, m in enumerate(mods)}
        self._dim, self.max_seq_length, self.tokenizer = dim, msl, FakeTok()

    def get_sentence_embedding_dimension(self):
        return self._dim


def test_describe_model_pooling_dimension_length_and_tokenizer(monkeypatch):
    monkeypatch.setattr(model_info, "has_modules_json", lambda n, r: True)
    m = model_info.describe_model(FakeST([Transformer(), Pooling(cls_token=True), Normalize()], dim=1024, msl=8192), "BAAI/bge-m3", "r")
    assert m["modules"] == ["Transformer", "Pooling", "Normalize"] and m["pooling_mode"] == "cls"
    assert m["pooling_is_library_default"] is False and m["embedding_dim"] == 1024 and m["max_seq_length"] == 8192
    assert m["tokenizer_class"] == "FakeTok" and m["tokenizer_name"] == "org/tok"
    both = model_info.describe_model(FakeST([Transformer(), Pooling(cls_token=True, mean_tokens=True)]))
    assert both["pooling_mode"] == "cls+mean"
    assert both["pooling_is_library_default"] is None                       # no name given: cannot look at the snapshot


def test_a_model_without_modules_json_is_flagged_as_library_default(monkeypatch):
    monkeypatch.setattr(model_info, "has_modules_json", lambda n, r: False)
    m = model_info.describe_model(FakeST([Transformer(), Pooling(mean_tokens=True)], dim=768), "allenai/scibert_scivocab_uncased", "r")
    assert m["pooling_mode"] == "mean" and m["pooling_is_library_default"] is True and m["has_modules_json"] is False


def test_has_modules_json_looks_only_at_the_local_snapshot(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r: snap)
    assert model_info.has_modules_json("a/b", "r") is False
    (snap / "modules.json").write_text("[]")
    assert model_info.has_modules_json("a/b", "r") is True
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r: None)
    assert model_info.has_modules_json("a/b", "r") is None


def _snapshot(tmp_path, name, modules=None, pooling=None, dense=None, config=None, sbert=None, tok=None):
    d = tmp_path / name
    d.mkdir()
    if modules is not None:
        (d / "modules.json").write_text(json.dumps(modules))
    if pooling is not None:
        (d / "1_Pooling").mkdir()
        (d / "1_Pooling" / "config.json").write_text(json.dumps(pooling))
    if dense is not None:
        (d / "2_Dense").mkdir()
        (d / "2_Dense" / "config.json").write_text(json.dumps(dense))
    (d / "config.json").write_text(json.dumps(config or {}))
    if sbert:
        (d / "sentence_bert_config.json").write_text(json.dumps(sbert))
    if tok:
        (d / "tokenizer_config.json").write_text(json.dumps(tok))
    return d


def test_describe_snapshot_sentence_transformers_model(tmp_path):
    d = _snapshot(tmp_path, "bge", modules=[{"type": "sentence_transformers.models.Transformer", "path": ""},
                                            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
                                            {"type": "sentence_transformers.models.Normalize", "path": "2_Normalize"}],
                  pooling={"word_embedding_dimension": 1024, "pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False},
                  config={"hidden_size": 1024, "architectures": ["XLMRobertaModel"], "max_position_embeddings": 8194},
                  sbert={"max_seq_length": 8192}, tok={"tokenizer_class": "XLMRobertaTokenizer"})
    m = model_info.describe_snapshot(d)
    assert m["pooling_mode"] == "cls" and m["pooling_is_library_default"] is False and m["embedding_dim"] == 1024
    assert m["max_seq_length"] == 8192 and m["tokenizer_class"] == "XLMRobertaTokenizer" and m["modules"][-1] == "Normalize"


def test_describe_snapshot_dense_output_and_two_pooling_modes(tmp_path):
    mods = [{"type": "x.Transformer", "path": ""}, {"type": "x.Pooling", "path": "1_Pooling"}, {"type": "x.Dense", "path": "2_Dense"}]
    d = _snapshot(tmp_path, "gtr", modules=mods, pooling={"word_embedding_dimension": 768, "pooling_mode_mean_tokens": True},
                  dense={"in_features": 768, "out_features": 768}, config={"d_model": 768})
    assert model_info.describe_snapshot(d)["embedding_dim"] == 768
    d2 = _snapshot(tmp_path, "two", modules=mods[:2], pooling={"word_embedding_dimension": 384, "pooling_mode_cls_token": True,
                                                                 "pooling_mode_mean_tokens": True})
    m2 = model_info.describe_snapshot(d2)
    assert m2["pooling_mode"] == "cls+mean" and m2["embedding_dim"] == 768


def test_describe_snapshot_plain_encoder_gets_the_library_default(tmp_path):
    d = _snapshot(tmp_path, "scibert", config={"hidden_size": 768, "max_position_embeddings": 512, "architectures": ["BertModel"]},
                  tok={"model_max_length": 1000000000000000019884624838656})
    m = model_info.describe_snapshot(d)
    assert m["pooling_is_library_default"] is True and m["pooling_mode"] == "mean" and m["modules"] == ["Transformer", "Pooling"]
    assert m["embedding_dim"] == 768 and m["max_seq_length"] == 512                       # the huge tokenizer default is ignored


# ---- derived runs carry the machine and the seed ---------------------------------------------------------
def test_hybrid_manifest_has_hardware_seeds_and_device(world):
    _, prep, tests, dense, bm = world
    r = hybrid.run_hybrid(dense, bm, tests, seed=7)
    man = json.loads(next(Path(r["run_dir"]).glob("results/*/run_manifest.json")).read_text())
    assert man["hardware"]["cpu_logical_cores"] and "python" in man["hardware"]
    assert man["seeds"]["seed"] == 7 and man["compute_device"] == "cpu"


def test_rerank_manifest_has_hardware_seeds_and_device(world):
    _, prep, tests, dense, bm = world
    r = rerank.run_rerank(dense, tests, overlap_scorer, model="org/rr", revision="abc", top_n=4, seed=11,
                          scorer_info={"model": "org/rr", "device": "cuda"})
    man = json.loads(next(Path(r["run_dir"]).glob("results/*/run_manifest.json")).read_text())
    assert man["seeds"]["seed"] == 11 and man["compute_device"] == "cuda" and man["hardware"]["platform"]


def test_no_seed_means_no_seed_block(world):
    _, prep, tests, dense, bm = world
    r = hybrid.run_hybrid(dense, bm, tests)
    man = json.loads(next(Path(r["run_dir"]).glob("results/*/run_manifest.json")).read_text())
    assert man["seeds"] is None

"""One chunking tokenizer / breakpoint model for all evaluated models; legacy per-model mode still available."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHUNKERS = ["fixed_token", "recursive_token", "semantic_fixed", "semantic_recursive", "hierarchical_section"]
MODELS = [("intfloat/e5-large-v2", "intfloat_e5_large_v2"), ("BAAI/bge-m3", "BAAI_bge_m3"),
          ("hkunlp/instructor-xl", "hkunlp_instructor_xl")]


def _stage4(tmp_path, model, chunker, *params, size=128, overlap=0):
    name, d = model
    out = tmp_path / f"{d}_{chunker}_{len(params)}"
    cmd = [sys.executable, "-m", "pipeline_lib.split_config", "--input", str(ROOT / "configs/main_config.yaml"),
           "--output-dir", str(out), "--sections", "stage4", "--param", f"model.name={name}", "--param", f"model.dir={d}",
           "--param", f"chunk.chunker={chunker}", "--param", f"chunk.size={size}", "--param", f"chunk.overlap={overlap}"]
    for p in params:
        cmd += ["--param", p]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return yaml.safe_load((out / "stage4.yaml").read_text())


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_stage4_strategy_is_identical_for_all_models_and_uses_the_reference_tokenizer(tmp_path, chunker):
    cfgs = [_stage4(tmp_path, m, chunker) for m in MODELS]
    strategies = [c["strategy"] for c in cfgs]
    assert strategies[0] == strategies[1] == strategies[2]
    p = strategies[0]["params"]
    assert p["model_name"] == "bert-base-uncased" and p["tokenizer_revision"] == "86b5e0934494bd15c9632b12f734a8a67f723594"
    if chunker.startswith("semantic"):
        assert p["breakpoint_model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert p["breakpoint_model_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        assert p["min_chunk_unit"] == "tokens"
    assert "e5" not in json.dumps(p) and "bge" not in json.dumps(p)


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_legacy_mode_uses_each_models_own_tokenizer(tmp_path, chunker):
    for name, d in MODELS:
        p = _stage4(tmp_path, (name, d), chunker, "chunking_tokenizer=model", "semantic_breakpoint_model=model")["strategy"]["params"]
        assert p["model_name"] == name and p["tokenizer_revision"] is None
        if chunker.startswith("semantic"):
            assert p["breakpoint_model_name"] == name and p["breakpoint_model_revision"] is None


def test_unpinned_reference_tokenizer_or_breakpoint_model_is_rejected(tmp_path):
    base = [sys.executable, "-m", "pipeline_lib.split_config", "--input", str(ROOT / "configs/main_config.yaml"),
            "--output-dir", str(tmp_path / "x"), "--sections", "stage4"]
    r = subprocess.run(base + ["--param", "chunking_tokenizer_revision=null"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "pin" in r.stderr
    r = subprocess.run(base + ["--param", "semantic_breakpoint_model={name: x/y}"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "revision" in r.stderr


DOC = ("Introduction\n\n" + " ".join(
    f"Sample {i} of 2-bromo-4-nitro-toluene was dissolved in 12.5 mL of dry CH2Cl2 at -78 C under argon. "
    f"The mixture was stirred for {i + 3} h, quenched with NH4Cl (aq.) and extracted three times." for i in range(60))
    + "\n\nResults\n\n" + " ".join(f"Yield {i}: {60 + i % 30} % after column chromatography (SiO2, hexane/EtOAc)." for i in range(60))
    + "\n\nReferences\n\n1. Doe, J. J. Org. Chem. 2020, 1, 1.\n")


@pytest.mark.parametrize("chunker", ["fixed_token", "hierarchical_section", "recursive_token", "semantic_fixed"])
def test_chunk_files_are_byte_identical_across_models(tmp_path, chunker):
    pytest.importorskip("transformers")
    if chunker.startswith("semantic"):
        pytest.importorskip("langchain_experimental"); pytest.importorskip("sentence_transformers")
    docs = tmp_path / "full_text.jsonl"
    docs.write_text("".join(json.dumps({"corpus_id": i, "text": DOC.replace("Sample", f"Doc{i} sample")}) + "\n" for i in range(3)))
    hashes = []
    for name, d in MODELS:
        cfg = _stage4(tmp_path, (name, d), chunker, size=64, overlap=16)
        cfg["input_path"] = str(docs)
        cfg["output_path"] = str(tmp_path / d / "chunks.jsonl")
        if "cache_dir" in cfg["strategy"]["params"]:
            cfg["strategy"]["params"]["cache_dir"] = str(tmp_path / "sem_cache")
        cf = tmp_path / f"{d}.yaml"; cf.write_text(yaml.safe_dump(cfg))
        r = subprocess.run([sys.executable, "-m", "stages.chunking", "--config", str(cf)], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        hashes.append(hashlib.sha256(Path(cfg["output_path"]).read_bytes()).hexdigest())
    assert len(set(hashes)) == 1
    assert (tmp_path / MODELS[0][1] / "chunks.jsonl").read_text().count("\n") > 3

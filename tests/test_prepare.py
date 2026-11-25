"""stages/prepare.py: chunk + label one configuration once, cache it, reuse it for every model."""
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("transformers")
pytest.importorskip("mteb")

from pipeline_lib import chunk_cache, grid
from pipeline_lib.dataset_pin import sha256_file
from stages import prepare
from tests.test_reference_tokenizer import DOC
from tools import metrics as M

ROOT = Path(__file__).resolve().parents[1]


def test_grid_is_the_makefile_grid():
    g = grid.grid()
    assert len(g) == 28 and (128, 0) in g and (512, 0) in g and (512, 16) not in g and (128, 48) not in g
    assert (448, 64) in g and (448, 0) in g and (192, 48) in g and (192, 64) not in g
    assert grid.grid([128, 512], [0]) == [(128, 0), (512, 0)]


@pytest.fixture()
def data(tmp_path):
    d = tmp_path / "chemquests"; d.mkdir()
    texts = {str(i): DOC.replace("Sample", f"Doc{i} sample") for i in range(3)}
    with open(d / "full_text.jsonl", "w") as f:
        for k, t in texts.items():
            f.write(json.dumps({"corpus_id": k, "text": t}) + "\n")
    qa = []
    for k, t in texts.items():
        for j in (5, 20, 41):
            s = t.index(f"Doc{k} sample {j} ")
            e = t.index(". ", s) + 1
            qa.append({"question": f"What was done to sample {j} of document {k}?", "corpus_id": k,
                       "content": t[s:e], "start_index": s, "end_index": e})
    with open(d / "qa.jsonl", "w") as f:
        for r in qa:
            f.write(json.dumps(r) + "\n")
    return d


def _prep(tmp_path, data, chunker="fixed_token", size=64, overlap=16, extra=()):
    return prepare.prepare_config(
        chunker, size, overlap, artifacts_root=str(tmp_path / "art"), qa_sha256=sha256_file(data / "qa.jsonl"),
        params=[f"paths.chemquests_dir={data}", "chunk_cache.dir=" + str(tmp_path / "prepared"), *extra], verbose=False)


def test_prepare_writes_chunks_both_qrels_diagnostics_and_manifest(tmp_path, data):
    f = _prep(tmp_path, data)
    for rel in ("chunks/chunks.jsonl", "mteb_task/corpus.jsonl", "mteb_task/queries.jsonl", "mteb_task/qrels_lexical.jsonl",
                "mteb_task/qrels_span_c090.jsonl", "mteb_task/overlap_table.jsonl", "coverage_diagnostics.csv",
                "no_relevant_chunk.csv", "manifest.json", "config/stage4.yaml"):
        assert (f / rel).exists(), rel
    m = chunk_cache.read_manifest(f)
    assert m["status"] == "complete" and m["tokenizer"] == "bert-base-uncased" and m["span_qrels_file"] == "qrels_span_c090.jsonl"
    assert set(m["timing_s"]) == {"chunking_s", "qrels_span_s", "qrels_lexical_s", "diagnostics_s"}
    assert chunk_cache.verify_files(f) == {}
    assert len(M.load_qrels(f / "mteb_task" / "qrels_span_c090.jsonl")) == 9
    assert len(M.load_qrels(f / "mteb_task" / "qrels_lexical.jsonl")) == 9
    assert not (f / "mteb_task" / "qrels.jsonl").exists()                         # no generic file that an outside reader could take for THE labels
    assert "mteb_task/qrels.jsonl" not in m["files_sha256"]


def test_second_call_is_a_cache_hit_and_models_share_the_folder(tmp_path, data):
    f1 = _prep(tmp_path, data, extra=["model.name=intfloat/e5-large-v2", "model.dir=intfloat_e5_large_v2"])
    stamp = (f1 / "manifest.json").stat().st_mtime_ns
    f2 = _prep(tmp_path, data, extra=["model.name=BAAI/bge-m3", "model.dir=BAAI_bge_m3"])
    assert f1 == f2 and (f2 / "manifest.json").stat().st_mtime_ns == stamp   # nothing was rebuilt


def test_legacy_per_model_chunking_gets_a_folder_per_model(tmp_path, data):
    fs = {_prep(tmp_path, data, extra=["chunking_tokenizer=model", "semantic_breakpoint_model=model", f"model.name={n}", f"model.dir={d}"])
          for n, d in [("bert-base-uncased", "a"), ("distilbert-base-uncased", "b")]}
    assert len(fs) == 2


def test_other_configuration_or_other_data_gives_another_folder(tmp_path, data):
    base = _prep(tmp_path, data)
    assert _prep(tmp_path, data, size=64, overlap=0) != base
    assert _prep(tmp_path, data, chunker="hierarchical_section", size=64, overlap=16) != base
    qa = (data / "qa.jsonl").read_text().splitlines()
    (data / "qa.jsonl").write_text("\n".join(qa[:-1]) + "\n")
    assert _prep(tmp_path, data) != base                                      # other qa.jsonl -> other key
    assert chunk_cache.is_complete(base)                                      # the old folder is untouched


def test_tampering_with_a_prepared_file_is_detected(tmp_path, data):
    f = _prep(tmp_path, data)
    with open(f / "mteb_task" / "qrels_span_c090.jsonl", "a") as fh:
        fh.write(json.dumps({"query_id": "0", "doc_id": "0:0", "relevance": 1}) + "\n")
    assert "mteb_task/qrels_span_c090.jsonl" in chunk_cache.verify_files(f)


def _corpus(f):
    return [json.loads(l) for l in (f / "mteb_task" / "corpus.jsonl").read_text().splitlines()]


def test_chunk_title_is_not_embedded_by_default_and_ids_are_kept(tmp_path, data):
    from pipeline_lib import truncation
    f = _prep(tmp_path, data)
    rows = _corpus(f)
    assert rows and all(r["title"] == "" and r["id"] for r in rows)
    chunks = [json.loads(l) for l in (f / "chunks" / "chunks.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [str(c["chunk_id"]) for c in chunks]
    # the string MTEB builds is exactly the chunk text
    assert truncation.embedded_passages(f / "mteb_task") == [r["text"].strip() for r in rows]


def test_embed_chunk_title_true_is_the_old_behaviour_and_makes_another_folder(tmp_path, data):
    f0 = _prep(tmp_path, data)
    f1 = _prep(tmp_path, data, extra=["embed_chunk_title=true"])
    assert f0 != f1
    rows = _corpus(f1)
    assert all(r["title"].endswith(f" - Chunk {r['id'].split(':')[-1]}") or " - Chunk " in r["title"] for r in rows)
    # the qrels do not depend on the title
    assert (f0 / "mteb_task" / "qrels_span_c090.jsonl").read_bytes() == (f1 / "mteb_task" / "qrels_span_c090.jsonl").read_bytes()
    assert (f0 / "mteb_task" / "qrels_lexical.jsonl").read_bytes() == (f1 / "mteb_task" / "qrels_lexical.jsonl").read_bytes()

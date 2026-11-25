"""pipeline_lib/question_metadata.py: the official per-question labels are read once, checked against qa.jsonl, and every mismatch raises."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from pipeline_lib import dataset_pin as DP, question_metadata as QM
from pipeline_lib.question_metadata import QuestionMetadataError
from stages import download_chemquests as D
from tests.test_prepare import data  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
SUBS = ("Alpha", "Beta", "Gamma")
KW = {"n_questions": 6, "subfields": SUBS}
TYPES = list(QM.QUESTION_TYPES)


def rows(n=6):
    qa = [{"question": f"What is item {i}?", "corpus_id": f"{i // 2:03d}", "content": "c", "start_index": 0, "end_index": 1} for i in range(n)]
    meta = [{"question_id": f"q{i:04d}", "corpus_id": r["corpus_id"], "question": r["question"], "question_type": TYPES[i % 4],
             "chemistry_subfields": [SUBS[i % 3], SUBS[(i + 1) % 3]] if i % 2 == 0 else [SUBS[i % 3]]} for i, r in enumerate(qa)]
    return qa, meta


def write(tmp_path, qa, meta):
    (tmp_path / "qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qa))
    (tmp_path / "question_metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in meta))
    return tmp_path / "question_metadata.jsonl", tmp_path / "qa.jsonl"


def load(tmp_path, qa, meta, **kw):
    m, q = write(tmp_path, qa, meta)
    return QM.load(m, q, **{**KW, **kw})


def test_a_correct_file_loads_and_is_aligned_with_qa(tmp_path):
    qa, meta = rows()
    md = load(tmp_path, qa, meta)
    assert len(md) == 6 and md.question_type[1] == TYPES[1] and md.subfields[0] == ("Alpha", "Beta") and md.subfields[1] == ("Beta",)
    assert md.subfield_names == SUBS and md.type_counts()["Conceptual"] == 2
    assert md.n_labels_counts() == {1: 3, 2: 3}


def test_counts_multi_label_and_fractional(tmp_path):
    qa, meta = rows()
    md = load(tmp_path, qa, meta)
    assert sum(md.subfield_counts().values()) == 9 and sum(md.subfield_fractions().values()) == pytest.approx(6)


def test_missing_file_names_the_download_command(tmp_path):
    qa, _ = rows()
    (tmp_path / "qa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qa))
    with pytest.raises(QuestionMetadataError, match="download_chemquests"):
        QM.load(tmp_path / "question_metadata.jsonl", tmp_path / "qa.jsonl", **KW)


def test_wrong_row_count(tmp_path):
    qa, meta = rows()
    with pytest.raises(QuestionMetadataError, match="5 rows, expected 6"):
        load(tmp_path, qa, meta[:5])


def test_duplicated_and_missing_ids(tmp_path):
    qa, meta = rows()
    meta[3]["question_id"] = "q0002"
    with pytest.raises(QuestionMetadataError, match="duplicated question_id"):
        load(tmp_path, qa, meta)
    qa, meta = rows()
    meta[3]["question_id"] = "q0099"
    with pytest.raises(QuestionMetadataError, match="missing"):
        load(tmp_path, qa, meta)


def test_shifted_ids(tmp_path):
    qa, meta = rows()
    meta[0], meta[1] = meta[1], meta[0]
    with pytest.raises(QuestionMetadataError, match="not the row index"):
        load(tmp_path, qa, meta)


def test_corpus_mismatch(tmp_path):
    qa, meta = rows()
    meta[4]["corpus_id"] = "009"
    with pytest.raises(QuestionMetadataError, match="corpus_id differs.*row 4"):
        load(tmp_path, qa, meta)


def test_question_text_mismatch_lists_the_differences(tmp_path):
    qa, meta = rows()
    meta[2]["question"] = "What is item 2 in this study?"
    with pytest.raises(QuestionMetadataError) as e:
        load(tmp_path, qa, meta)
    assert "1 rows" in str(e.value) and "in this study?" in str(e.value) and "What is item 2?" in str(e.value)


def test_unknown_or_missing_type(tmp_path):
    qa, meta = rows()
    meta[1]["question_type"] = "Experimental/Synthetic"
    with pytest.raises(QuestionMetadataError, match="question_type"):
        load(tmp_path, qa, meta)
    qa, meta = rows()
    del meta[1]["question_type"]
    with pytest.raises(QuestionMetadataError, match="question_type"):
        load(tmp_path, qa, meta)


@pytest.mark.parametrize("bad", [[], "Alpha", None, ["Alpha", "Alpha"], [""]])
def test_empty_or_malformed_subfields(tmp_path, bad):
    qa, meta = rows()
    meta[0]["chemistry_subfields"] = bad
    with pytest.raises(QuestionMetadataError, match="chemistry_subfields"):
        load(tmp_path, qa, meta)


def test_subfield_count_and_names_are_checked_not_mapped(tmp_path):
    qa, meta = rows()
    meta[0]["chemistry_subfields"] = ["Alpha", "Delta"]
    with pytest.raises(QuestionMetadataError, match="4 distinct subfields, expected 3"):
        load(tmp_path, qa, meta)
    qa, meta = rows()
    for m in meta:
        m["chemistry_subfields"] = [{"Gamma": "Gamma Chemistry"}.get(x, x) for x in m["chemistry_subfields"]]
    with pytest.raises(QuestionMetadataError, match="not in the expected list"):
        load(tmp_path, qa, meta)


def test_the_17_names_of_the_paper():
    assert len(QM.SUBFIELDS) == len(set(QM.SUBFIELDS)) == 17 and list(QM.SUBFIELDS) == sorted(QM.SUBFIELDS)
    assert len(QM.QUESTION_TYPES) == 4 and QM.N_QUESTIONS == 952


# ---- the pin and the download ------------------------------------------------------------------------------------
def test_separate_pin_everywhere():
    assert DP.QUESTION_METADATA_REVISION == "790d2bc6829ccd7e1e480a6a4610149650b79e9f" != DP.CHEMQUESTS_REVISION
    assert len(DP.QUESTION_METADATA_SHA256) == 64
    for rel, key in [("configs/stages/stage3.yaml", None), ("configs/download_chemquests_config.yaml", None), ("configs/main_config.yaml", "stage3")]:
        cfg = yaml.safe_load((ROOT / rel).read_text())
        cfg = cfg[key] if key else cfg
        assert cfg["question_metadata_revision"] == DP.QUESTION_METADATA_REVISION and cfg["question_metadata_sha256"] == DP.QUESTION_METADATA_SHA256
        assert cfg["revision"] == DP.CHEMQUESTS_REVISION and DP.QUESTION_METADATA_REPO_PATH in cfg["data_files"]


def test_download_uses_its_own_revision_and_checks_the_hash(tmp_path, monkeypatch):
    qa = json.dumps({"question": "q", "corpus_id": "1", "content": "lain", "start_index": 1, "end_index": 5}) + "\n"
    qm = json.dumps({"question_id": "q0000"}) + "\n"
    store = tmp_path / "hub"; (store / "qa").mkdir(parents=True)
    (store / "qa.jsonl").write_text(qa); (store / "qa" / "question_metadata.jsonl").write_text(qm)
    calls = []

    def fake_dl(repo_id, filename, repo_type, revision, token):
        calls.append((filename, revision))
        return str(store / filename)

    monkeypatch.setattr(D, "hf_hub_download", fake_dl)
    out = tmp_path / "out"
    D.download_files("x/y", ["qa.jsonl", "qa/question_metadata.jsonl"], str(out), revision="a" * 40, expected_qa_sha256=hashlib.sha256(qa.encode()).hexdigest(),
                     question_metadata_revision="b" * 40, expected_question_metadata_sha256=hashlib.sha256(qm.encode()).hexdigest())
    assert calls == [("qa.jsonl", "a" * 40), ("qa/question_metadata.jsonl", "b" * 40)]
    assert (out / "question_metadata.jsonl").read_text() == qm            # next to qa.jsonl
    prov = DP.question_metadata_provenance(out / "qa.jsonl")
    assert prov["sha256"] == hashlib.sha256(qm.encode()).hexdigest() and prov["revision"] == DP.QUESTION_METADATA_REVISION and prov["matches_pin"] is False
    with pytest.raises(DP.DatasetIntegrityError, match="question_metadata.jsonl SHA256 mismatch"):
        D.download_files("x/y", ["qa/question_metadata.jsonl"], str(tmp_path / "o2"), revision="a" * 40,
                         question_metadata_revision="b" * 40, expected_question_metadata_sha256="0" * 64)
    with pytest.raises(DP.DatasetIntegrityError, match="own pinned revision"):
        D.download_files("x/y", ["qa/question_metadata.jsonl"], str(tmp_path / "o3"), revision="a" * 40, question_metadata_revision="main")


def test_provenance_without_the_file(tmp_path):
    assert DP.question_metadata_provenance(tmp_path / "qa.jsonl") == {"file": None}


# ---- manifests ----------------------------------------------------------------------------------------------------
def test_prepare_manifest_records_the_question_metadata_without_changing_the_cache_key(tmp_path, data):
    from pipeline_lib import chunk_cache
    from stages import prepare
    kw = dict(artifacts_root=str(tmp_path / "art"), qa_sha256=DP.sha256_file(data / "qa.jsonl"),
              params=[f"paths.chemquests_dir={data}", "chunk_cache.dir=" + str(tmp_path / "prepared")], verbose=False)
    f1 = prepare.prepare_config("fixed_token", 64, 16, **kw)
    assert chunk_cache.read_manifest(f1)["question_metadata"] == {"file": None}              # nothing is invented
    (data / "question_metadata.jsonl").write_text('{"question_id": "q0000"}\n')
    f2 = prepare.prepare_config("fixed_token", 64, 16, **kw)
    assert f1 == f2                                                                           # the cache key does not depend on the metadata
    f3 = prepare.prepare_config("fixed_token", 64, 16, force=True, **kw)
    qm = chunk_cache.read_manifest(f3)["question_metadata"]
    assert qm["revision"] == DP.QUESTION_METADATA_REVISION and qm["sha256"] == DP.sha256_file(data / "question_metadata.jsonl") and qm["matches_pin"] is False


def test_run_manifest_takes_the_record_from_the_prepared_folder(tmp_path):
    pytest.importorskip("mteb")
    from stages import run_eval
    prepared = tmp_path / "prep"; (prepared / "mteb_task").mkdir(parents=True)
    (prepared / "manifest.json").write_text(json.dumps({"question_metadata": {"revision": "r", "sha256": "s"}}))

    class T:
        spec = {"kwargs": {"data_dir": str(prepared / "mteb_task"), "qrels_file": "qrels_span_c090.jsonl"}}
    assert run_eval._question_metadata_of_task([T()]) == {"revision": "r", "sha256": "s"}
    assert run_eval._question_metadata_of_task([]) is None

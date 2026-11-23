"""tools/model_properties.py: one row per model from the pinned local snapshots; models that are not cached are skipped and listed."""
import csv
import json
from pathlib import Path

import yaml

from pipeline_lib import model_info
from tools import model_properties as MP


def _snap(root, name, with_modules=True, cls=False):
    d = root / name.replace("/", "--")
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"hidden_size": 384, "max_position_embeddings": 512, "architectures": ["BertModel"], "model_type": "bert"}))
    if with_modules:
        (d / "modules.json").write_text(json.dumps([{"type": "x.Transformer", "path": ""}, {"type": "x.Pooling", "path": "1_Pooling"}]))
        (d / "1_Pooling").mkdir()
        (d / "1_Pooling" / "config.json").write_text(json.dumps({"word_embedding_dimension": 384, "pooling_mode_cls_token": cls,
                                                                  "pooling_mode_mean_tokens": not cls}))
    return d


def test_rows_skipped_models_and_prompt_columns(tmp_path, monkeypatch):
    snaps = {"a/cls-model": _snap(tmp_path, "a/cls-model", cls=True), "b/plain-bert": _snap(tmp_path, "b/plain-bert", with_modules=False),
             "c/e5": _snap(tmp_path, "c/e5")}
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r, c=None: snaps.get(n))
    models = [{"name": n, "revision": "r" * 40} for n in ("a/cls-model", "b/plain-bert", "c/e5", "d/not-cached")]
    prompts = {"c/e5": {"query": "query: ", "passage": "passage: "}, "a/cls-model": {"query": "", "passage": ""}}
    rows, skipped = MP.build_rows(models, prompts, use_prompts=True, load_tokenizers=False, shortlist={"c/e5"})
    by = {r["model"]: r for r in rows}
    assert [s["model"] for s in skipped] == ["d/not-cached"] and len(rows) == 3
    assert by["a/cls-model"]["pooling_mode"] == "cls" and by["a/cls-model"]["pooling_is_library_default"] is False
    assert by["b/plain-bert"]["pooling_is_library_default"] is True and by["b/plain-bert"]["pooling_mode"] == "mean"
    assert by["c/e5"]["query_prompt"] == "query: " and by["c/e5"]["passage_prompt"] == "passage: " and by["c/e5"]["shortlisted"] is True
    assert by["b/plain-bert"]["shortlisted"] is False and by["c/e5"]["embedding_dim"] == 384 and by["c/e5"]["max_seq_length"] == 512
    off, _ = MP.build_rows(models[:3], prompts, use_prompts=False, load_tokenizers=False)
    assert {r["query_prompt"] for r in off} == {""}                        # use_prompts false: no prompt is used


def test_instruction_masking_flag(tmp_path, monkeypatch):
    snaps = {"i/xl": _snap(tmp_path, "i/xl")}
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r, c=None: snaps.get(n))
    prompts = {"i/xl": {"query": "Represent the question: ", "passage": "Represent the doc: ", "mask_instruction": True}}
    on, _ = MP.build_rows([{"name": "i/xl", "revision": "r"}], prompts, use_prompts=True, mask_instruction=True, load_tokenizers=False)
    off, _ = MP.build_rows([{"name": "i/xl", "revision": "r"}], prompts, use_prompts=True, mask_instruction=False, load_tokenizers=False)
    assert on[0]["instruction_masked"] is True and off[0]["instruction_masked"] is False


def test_main_writes_csv_and_skipped_list(tmp_path, monkeypatch, capsys):
    snaps = {"a/m": _snap(tmp_path, "a/m")}
    monkeypatch.setattr(model_info, "snapshot_dir", lambda n, r, c=None: snaps.get(n))
    cfg = {"params": {"use_prompts": True, "instructor_mask_instruction": True, "model_prompts": {"a/m": {"query": "q: ", "passage": "p: "}}},
           "stage1": {"models": [{"name": "a/m", "revision": "1" * 40}, {"name": "x/missing", "revision": "2" * 40}]}}
    c = tmp_path / "main.yaml"
    c.write_text(yaml.safe_dump(cfg))
    sl = tmp_path / "short.csv"
    sl.write_text("model_id,model_name\nx,a/m\n")
    assert MP.main(["--main-config", str(c), "--out-dir", str(tmp_path / "out"), "--no-tokenizers", "--shortlist", str(sl)]) == 0
    rows = list(csv.DictReader(open(tmp_path / "out" / "table_model_properties.csv")))
    assert [r["model"] for r in rows] == ["a/m"] and rows[0]["shortlisted"] == "True" and rows[0]["query_prompt"] == "q: "
    sk = list(csv.DictReader(open(tmp_path / "out" / "table_model_properties_skipped.csv")))
    assert [r["model"] for r in sk] == ["x/missing"]
    assert "skipped" in capsys.readouterr().out


def test_columns_are_stable():
    assert MP.COLUMNS[:4] == ["model", "revision", "pooling_mode", "pooling_is_library_default"]

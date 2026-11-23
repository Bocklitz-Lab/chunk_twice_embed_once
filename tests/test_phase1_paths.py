"""Phase 1 / stage 2 paths come from ONE config key; stage 1 never writes into a folder that already holds results;
the tracked stage files are what ``make stage0`` generates; MODELS_CSV is checked before a rerun starts."""
import re
from pathlib import Path

import pytest
import yaml

from pipeline_lib import split_config
from stages import model_screening, rerun

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "configs" / "main_config.yaml")


def _stage(name, params=()):
    return split_config.resolve_section(MAIN, name, params)


def test_stage1_output_and_stage2_input_are_the_same_folder():
    sub = yaml.safe_load(Path(MAIN).read_text())["params"]["paths"]["phase1_results_subdir"]
    s1, s2 = _stage("stage1"), _stage("stage2")
    assert sub != "validation"                                   # never the folder of the published results
    assert s1["output_root"] == f"{s2['root_dir']}/{s2['results_subdir']}" == f"artifacts/embedding_models_screening/{sub}"
    assert sub in s2["output_per_dataset_csv"] and sub in s2["output_union_csv"]      # the published shortlist is not overwritten


def test_changing_the_one_key_moves_both_stages():
    s1, s2 = _stage("stage1", ["paths.phase1_results_subdir=other_run"]), _stage("stage2", ["paths.phase1_results_subdir=other_run"])
    assert s1["output_root"].endswith("/other_run") and s2["results_subdir"] == "other_run" and "other_run" in s2["output_per_dataset_csv"]


def test_tracked_stage_files_are_what_stage0_generates():
    for name in ("stage1", "stage2", "stage3"):
        tracked = yaml.safe_load((ROOT / "configs" / "stages" / f"{name}.yaml").read_text())
        assert tracked == _stage(name), f"configs/stages/{name}.yaml is stale: run `make stage0`"


def test_tracked_stage1_carries_the_prompt_settings():
    s1 = yaml.safe_load((ROOT / "configs" / "stages" / "stage1.yaml").read_text())
    assert s1["use_prompts"] is True and s1["instructor_mask_instruction"] is True and s1["seed"]
    assert {m["name"] for m in s1["models"]} == set(s1["model_prompts"])


def test_makefile_default_models_csv_is_the_stage2_output():
    mk = (ROOT / "Makefile").read_text()
    default = re.search(r"^MODELS_CSV \?= (\S+)", mk, re.M).group(1)
    assert default == _stage("stage2")["output_per_dataset_csv"] == rerun.default_models_csv(MAIN)


def test_external_table_reads_the_new_phase1_folder():
    ext = yaml.safe_load(Path(MAIN).read_text())["params"]["external"]
    assert ext["phase1_subdirs"] == ["embedding_models_screening/validation_prompts"]


# ---- stage 1 refuses a folder that already holds results ----------------------------------------------------
def test_stage1_refuses_a_folder_with_results(tmp_path):
    model_screening.check_output_folder(tmp_path / "missing")            # does not exist: fine
    (tmp_path / "empty").mkdir()
    model_screening.check_output_folder(tmp_path / "empty")              # empty: fine
    (tmp_path / "logs_only").mkdir()
    (tmp_path / "logs_only" / "run.log").write_text("x")
    model_screening.check_output_folder(tmp_path / "logs_only")          # no result file: fine
    for rel in ("summary.json", "m/ChemNQRetrieval.json", "efficiency_phase1.csv"):
        d = tmp_path / rel.replace("/", "_")
        (d / Path(rel).parent).mkdir(parents=True)
        (d / rel).write_text("{}")
        with pytest.raises(SystemExit, match="already holds"):
            model_screening.check_output_folder(d)
        model_screening.check_output_folder(d, force=True)               # --force: on purpose


def test_stage1_main_stops_before_loading_a_model(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT / "configs" / "stages" / "stage1.yaml").read_text())
    cfg["output_root"] = str(tmp_path / "old")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "summary.json").write_text("[]")
    p = tmp_path / "stage1.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("sys.argv", ["model_screening", "--config", str(p)])
    monkeypatch.setattr(model_screening, "SentenceTransformer", lambda *a, **k: pytest.fail("a model was loaded"))
    with pytest.raises(SystemExit, match="already holds"):
        model_screening.main()


# ---- MODELS_CSV check ---------------------------------------------------------------------------------------
def test_missing_models_csv_gives_a_clear_message(tmp_path):
    with pytest.raises(SystemExit) as e:
        rerun.check_models_csv(str(tmp_path / "nope.csv"))
    msg = str(e.value)
    assert "MODELS_CSV" in msg and "stage 2" in msg and "cp -a temp_files/artifacts_old/embedding_models_screening artifacts/" in msg
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(SystemExit):
        rerun.check_models_csv(str(tmp_path / "empty.csv"))
    ok = tmp_path / "ok.csv"
    ok.write_text("model_id,model_name,revision\nx,a/b,1\n")
    rerun.check_models_csv(str(ok))


def test_rerun_eval_stops_without_models_csv_and_prepare_does_not_need_it(tmp_path, monkeypatch):
    monkeypatch.setattr(rerun, "ensure_data", lambda *a, **k: pytest.fail("data step reached"))
    with pytest.raises(SystemExit, match="MODELS_CSV"):
        rerun.main(["eval", "--models-csv", str(tmp_path / "nope.csv"), "--artifacts-root", str(tmp_path / "art")])
    # prepare reads no model list: it gets as far as the data step
    with pytest.raises(pytest.fail.Exception, match="data step reached"):
        rerun.main(["prepare", "--models-csv", str(tmp_path / "nope.csv"), "--artifacts-root", str(tmp_path / "art")])


def test_makefile_check_target_stops_early(tmp_path):
    import subprocess
    r = subprocess.run(["make", "-C", str(ROOT), "check-models-csv", f"MODELS_CSV={tmp_path / 'nope.csv'}"], capture_output=True, text=True)
    assert r.returncode != 0 and "MODELS_CSV not found" in r.stdout and "cp -a temp_files/artifacts_old" in r.stdout

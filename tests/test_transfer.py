"""B6: `make transfer` (stages/transfer.py): validation of params.transfer, one `stages.rerun eval` per configuration, no rerank check."""
import copy
from pathlib import Path

import pytest
import yaml

from stages import rerun, transfer

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "configs" / "main_config.yaml")


@pytest.fixture()
def doc():
    return transfer.load(MAIN)


def test_the_configured_block_is_valid_and_matches_the_reviewer1_experiment(doc):
    assert transfer.validate(doc) == []
    tr = doc["params"]["transfer"]
    roles = [m["role"] for m in tr["models"]]
    assert roles.count("non_shortlisted") == 14 and roles.count("shortlisted_reference") == 3 and roles.count("extra") == 1
    assert [c["id"] for c in tr["configurations"]] == ["C1", "C2", "C3", "C4", "C5", "C6"]
    assert {c["id"] for c in tr["configurations"] if c.get("extra")} == {"C6"}
    refs = {m["name"] for m in tr["models"] if m["role"] == "shortlisted_reference"}
    assert refs == {"intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5", "nomic-ai/nomic-embed-text-v1.5"}
    pairs = {p["id"]: p for p in tr["pairs"]}
    assert set(pairs) == {"P1", "P2", "P3", "P4a", "P4b"} and {m for p in tr["pairs"] for m in [p["base"], *p["tuned"]]} <= {m["name"] for m in tr["models"]}


def test_non_shortlisted_models_are_not_in_the_published_shortlist(doc):
    short = {"BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", "BAAI/bge-m3", "BAAI/bge-small-en-v1.5", "hkunlp/instructor-xl", "intfloat/e5-base",
             "intfloat/e5-large-v2", "intfloat/e5-large", "intfloat/e5-small", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small",
             "nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1"}
    for m in doc["params"]["transfer"]["models"]:
        assert (m["name"] in short) == (m["role"] == "shortlisted_reference") or m["role"] == "extra"
        assert m["role"] != "non_shortlisted" or m["name"] not in short


def _problems(doc, mutate):
    d = copy.deepcopy(doc)
    mutate(d["params"]["transfer"], d)
    return transfer.validate(d)


def test_validation_catches_unpinned_and_mismatching_revisions(doc):
    assert any("pinned 40-hex" in p for p in _problems(doc, lambda t, d: t["models"][0].update(revision="main")))
    assert any("differs from the Phase 1 pin" in p for p in _problems(doc, lambda t, d: t["models"][0].update(revision="0" * 40)))


def test_validation_catches_missing_prompt_entries_roles_and_duplicates(doc):
    assert any("no entry in params.model_prompts" in p for p in _problems(doc, lambda t, d: d["params"]["model_prompts"].pop(t["models"][0]["name"])))
    assert any("role must be" in p for p in _problems(doc, lambda t, d: t["models"][0].update(role="x")))
    assert any("listed twice" in p for p in _problems(doc, lambda t, d: t["models"].append(dict(t["models"][0]))))
    assert any("not in stage1.models" in p for p in _problems(doc, lambda t, d: t["models"][0].update(name="x/unknown", revision="1" * 40)))


def test_validation_catches_configurations_outside_the_grid(doc):
    assert any("not a valid configuration" in p for p in _problems(doc, lambda t, d: t["configurations"][0].update(size=512, overlap=64)))
    assert any("not a valid configuration" in p for p in _problems(doc, lambda t, d: t["configurations"][0].update(size=1024, overlap=0)))
    assert any("unknown chunker" in p for p in _problems(doc, lambda t, d: t["configurations"][0].update(chunker="nope")))
    assert any("listed twice" in p for p in _problems(doc, lambda t, d: t["configurations"].append(dict(t["configurations"][0], id="C9"))))


def test_the_transfer_root_must_not_be_the_main_root(doc):
    assert any("own folder" in p for p in _problems(doc, lambda t, d: t.update(artifacts_root=d["params"]["paths"]["artifacts_root"])))


def test_selection_switches(doc):
    all_ = transfer.selected(doc)
    assert len(all_["models"]) == 18 and len(all_["configurations"]) == 6
    no_extra = transfer.selected(doc, extras=False)
    assert len(no_extra["models"]) == 17 and [c["id"] for c in no_extra["configurations"]] == ["C1", "C2", "C3", "C4", "C5"]
    no_ref = transfer.selected(doc, references=False)
    assert len(no_ref["models"]) == 15 and all(m["role"] != "shortlisted_reference" for m in no_ref["models"])


def test_calls_are_one_per_configuration_with_shared_inputs_and_no_baselines(doc):
    calls = transfer.build_calls(doc, MAIN, efficiency=True, latency=True)
    assert len(calls) == 6
    for call in calls:
        assert call[0] == "eval" and "--no-rerank" in call and "--no-hybrid" in call and "--no-bm25" in call
        assert "--efficiency" in call and "--latency" in call
        assert call[call.index("--artifacts-root") + 1] == "artifacts/transfer"
        assert "chunk_cache.dir=artifacts/prepared" in call and "paths.chemquests_dir=artifacts/chemquests_hf_db" in call     # shared with the main run
        names = call[call.index("--models") + 1:call.index("--chunkers")]
        assert len(names) == 18 and all("@" in n and len(n.split("@")[1]) == 40 for n in names)
    assert all("--no-report" in c for c in calls[:-1]) and "--no-report" not in calls[-1] and calls[-1][calls[-1].index("--report-dir") + 1] == "results/v2-revision/transfer"
    cells = [(c[c.index("--chunkers") + 1], c[c.index("--sizes") + 1], c[c.index("--overlaps") + 1]) for c in calls]
    assert cells[3] == ("fixed_token", "384", "32") and cells[5] == ("fixed_token", "128", "0")
    off = transfer.build_calls(doc, MAIN, efficiency=False, latency=False)
    assert all("--efficiency" not in c and "--latency" not in c for c in off)


def test_a_transfer_call_does_not_trigger_the_rerank_ablation_check(doc, tmp_path, capsys):
    call = transfer.build_calls(doc, MAIN, efficiency=False, latency=False)[0]
    call = call[:call.index("--artifacts-root") + 1] + [str(tmp_path / "t")] + call[call.index("--artifacts-root") + 2:]
    assert rerun.main(call + ["--dry-run"]) == 0                       # e5-large-v2 is in the list but the ablation cell (256/0) is not: no abort
    out = capsys.readouterr().out
    assert "ablation" not in out and "dry-run" in out
    without = [a for a in call if a != "--no-rerank"]                  # the same call WITH the check: it would abort (ablation = fixed_token 256/0)
    with pytest.raises(SystemExit, match="ablation"):
        rerun.main(without + ["--dry-run"])


def test_main_dry_run_validates_and_prints(capsys):
    assert transfer.main(["--dry-run", "--efficiency", "--latency"]) == 0
    out = capsys.readouterr().out
    assert "108 runs" in out and "artifacts/transfer" in out and out.count("stages.rerun eval") == 6
    assert transfer.main(["--dry-run", "--no-extras", "--no-references"]) == 0
    assert "14 models x 5 configurations = 70 runs" in capsys.readouterr().out


def test_main_stops_on_an_invalid_block(tmp_path):
    d = yaml.safe_load(Path(MAIN).read_text())
    d["params"]["transfer"]["models"][0]["revision"] = "main"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(SystemExit, match="not valid"):
        transfer.main(["--main-config", str(p), "--dry-run"])


def test_makefile_target_calls_the_stage_with_the_efficiency_flags():
    import subprocess
    out = subprocess.run(["make", "-n", "-C", str(ROOT), "transfer"], capture_output=True, text=True).stdout
    assert "python -m stages.transfer --efficiency --latency" in out

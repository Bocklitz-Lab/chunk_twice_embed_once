"""The Phase 1 evaluation split comes from ONE key (params.phase1.eval_split); every consumer reads it and nothing falls back."""
import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

from pipeline_lib import phase1_split as P
from pipeline_lib.phase1_split import Phase1SplitError

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs" / "main_config.yaml"


def test_the_key_defaults_to_what_the_paper_used():
    assert P.eval_splits(MAIN) == {"ChemNQRetrieval": "test", "ChemHotpotQARetrieval": "test"}
    n = P.n_queries_table(MAIN)
    assert n["ChemNQRetrieval"] == {"test": 27} and n["ChemHotpotQARetrieval"] == {"train": 180, "dev": 8, "test": 18}


def test_pick_split_has_no_fallback_and_names_the_splits_present():
    sc = {"train": [{"ndcg_at_10": 1}], "dev": [{"ndcg_at_10": 2}]}
    assert P.pick_split(sc, "T", {"T": "dev"})["ndcg_at_10"] == 2
    with pytest.raises(Phase1SplitError, match=r"'test' is not in the result file.*\['dev', 'train'\]"):
        P.pick_split(sc, "T", {"T": "test"})
    with pytest.raises(Phase1SplitError, match="no evaluation split configured"):
        P.pick_split(sc, "T", {})
    with pytest.raises(Phase1SplitError):
        P.n_queries_of("T", "test", None) or P.split_of("T", {})


def test_unset_key_raises(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump({"params": {"seed": 1}}))
    with pytest.raises(Phase1SplitError, match="not set"):
        P.eval_splits(cfg)


def test_tracked_stage2_file_carries_the_key_and_no_preference_list():
    s2 = yaml.safe_load((ROOT / "configs" / "stages" / "stage2.yaml").read_text())
    assert s2["eval_split"] == P.eval_splits(MAIN) and "split_preference" not in s2
    assert "split_preference" not in MAIN.read_text().split("stage2:")[1].split("stage3")[0]


# ---- stage 2 on a small fixture of the published layout --------------------------------------------------------
def _leaf(root, name, rev, hp, nq):
    d = root / f"{name.replace('/', '__')}_{rev}" / name.replace("/", "__") / rev
    d.mkdir(parents=True)
    (d / "model_meta.json").write_text(json.dumps({"name": name, "revision": rev}))
    mk = lambda nd, rc: [{"ndcg_at_10": nd, "recall_at_10": rc}]                             # noqa: E731
    (d / "ChemHotpotQARetrieval.json").write_text(json.dumps({"task_name": "ChemHotpotQARetrieval", "evaluation_time": 10.0,
                                                              "scores": {sp: mk(*v) for sp, v in hp.items()}}))
    (d / "ChemNQRetrieval.json").write_text(json.dumps({"task_name": "ChemNQRetrieval", "evaluation_time": 10.0, "scores": {"test": mk(*nq)}}))


def _fixture(tmp_path):
    val = tmp_path / "validation"
    # 10 models; HotpotQA: test favours m0, m1; train favours m8, m9; NQ test favours m0, m5 (top 20 % of 10 = 2)
    for i in range(10):
        hp = {"test": (0.9 - 0.05 * i,) * 2, "train": (0.5 + 0.04 * i,) * 2, "dev": (0.5,) * 2}
        nq = (0.9 - 0.01 * i,) * 2 if i != 5 else (0.895, 0.895)
        _leaf(val, f"org/m{i}", "r1", hp, nq)
    return tmp_path


def _stage2(tmp_path, hot_split, monkeypatch):
    from stages import screen_top_k_models as S
    out = tmp_path / f"per_{hot_split}.csv"
    cfg = {"root_dir": str(tmp_path), "results_subdir": "validation", "metric_key": "geom_mean", "geom_mean": {"k": 10, "weights": {"ndcg": 1, "recall": 1, "time": 0}},
           "top_percent": 0.2, "output_per_dataset_csv": str(out), "eval_split": {"ChemNQRetrieval": "test", "ChemHotpotQARetrieval": hot_split},
           "n_queries": {"ChemNQRetrieval": {"test": 27}, "ChemHotpotQARetrieval": {"train": 180, "dev": 8, "test": 18}}}
    c = tmp_path / f"s2_{hot_split}.yaml"
    c.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["screen", "--config", str(c)])
    S.main()
    return list(csv.DictReader(open(out)))


def test_stage2_selects_on_the_configured_split_and_writes_split_and_n_queries(tmp_path, monkeypatch):
    _fixture(tmp_path)
    test = _stage2(tmp_path, "test", monkeypatch)
    train = _stage2(tmp_path, "train", monkeypatch)
    hp = lambda rows: {r["model_name"] for r in rows if r["dataset"] == "ChemHotpotQARetrieval"}      # noqa: E731
    assert hp(test) == {"org/m0", "org/m1"} and hp(train) == {"org/m8", "org/m9"}                      # another split, another shortlist
    r = next(r for r in test if r["dataset"] == "ChemHotpotQARetrieval")
    assert (r["split"], r["n_queries"]) == ("test", "18")
    r = next(r for r in train if r["dataset"] == "ChemHotpotQARetrieval")
    assert (r["split"], r["n_queries"]) == ("train", "180")
    assert {(r["split"], r["n_queries"]) for r in test if r["dataset"] == "ChemNQRetrieval"} == {("test", "27")}
    assert "split_used" not in test[0]


def test_stage2_stops_when_the_configured_split_is_missing(tmp_path, monkeypatch):
    _fixture(tmp_path)
    with pytest.raises(Phase1SplitError, match="'validation' is not in the result file"):
        _stage2(tmp_path, "validation", monkeypatch)                                                 # no fallback to test / train


def test_stage2_without_the_key_stops(tmp_path, monkeypatch):
    from stages import screen_top_k_models as S
    _fixture(tmp_path)
    c = tmp_path / "bad.yaml"
    c.write_text(yaml.safe_dump({"root_dir": str(tmp_path), "results_subdir": "validation", "metric_key": "ndcg_at_10", "split_preference": ["test"]}))
    monkeypatch.setattr(sys, "argv", ["screen", "--config", str(c)])
    with pytest.raises(SystemExit, match="split_preference is gone"):
        S.main()


def test_efficiency_phase1_table_names_the_split(tmp_path):
    import pandas as pd
    from tools import efficiency_analysis as E
    f = tmp_path / "e.csv"
    pd.DataFrame({"model": ["a", "a"], "n_params": [1e6, 1e6], "weights_mb": [4.0, 4.0], "embedding_dim": [8, 8], "index_mb_fp32": [1.0, 2.0]}).to_csv(f, index=False)
    t = E.phase1_table(f)
    assert t["split"][0] == "ChemHotpotQARetrieval=test; ChemNQRetrieval=test" and t["n_queries"][0] == "ChemHotpotQARetrieval=18; ChemNQRetrieval=27"

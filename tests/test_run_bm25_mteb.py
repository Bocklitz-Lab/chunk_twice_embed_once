"""stages/run_bm25_mteb.py on a tiny fake MTEB-style task (no download), the Phase 1 layout it writes, and the stage 2 guard."""
import json
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("rank_bm25")
pytest.importorskip("mteb")

from pipeline_lib.bm25_retriever import BM25Retriever
from stages import run_bm25_mteb as X


class Meta:
    eval_splits = ["test"]
    dataset = {"path": "org/fake", "revision": "deadbeef"}


class FakeTask:
    """The attributes of a loaded monolingual MTEB retrieval task."""
    metadata = Meta
    is_multilingual = False

    def __init__(self, corpus, queries, qrels, split="test"):
        self.metadata = type("M", (), {"eval_splits": [split], "dataset": Meta.dataset})
        self.corpus, self.queries, self.relevant_docs = {split: corpus}, {split: queries}, {split: qrels}


def fake_loader(name):
    corpus = {"d1": "sodium hydroxide was added", "d2": "palladium catalysed cross coupling", "d3": {"title": "zeta potential", "text": "measured by light scattering"},
              "d4": "unrelated gardening notes"}
    queries = {"q1": "which catalyst gave cross coupling", "q2": "zeta", "q3": "no judgement for this one"}
    qrels = {"q1": {"d2": 1}, "q2": {"d3": 1}}                      # q3 has no judgements: MTEB does not evaluate it
    return FakeTask(corpus, queries, qrels)


def test_title_in_text_indexes_what_mteb_gives_a_dense_model():
    corpus = {"a": {"title": "zeta potential", "text": "light scattering"}, "b": {"title": "", "text": "other words"}}
    corpus.update({f"x{i}": {"title": "", "text": f"filler document number {i}"} for i in range(5)})     # BM25 idf is 0 for a term in half of a tiny corpus
    plain, titled = BM25Retriever(), BM25Retriever(title_in_text=True)
    plain.index(corpus), titled.index(corpus)
    assert plain.search({"q": "zeta"})["q"]["a"] == 0.0 and titled.search({"q": "zeta"})["q"]["a"] > 0     # ChemQuests default: unchanged, text only
    assert titled.document_text(corpus["a"]) == "zeta potential light scattering" and titled.document_text(corpus["b"]) == "other words"


def test_split_data_accepts_string_and_dict_corpora_and_drops_queries_without_judgements():
    corpus, queries, qrels = X.split_data(fake_loader("x"), "test")
    assert corpus["d1"] == {"title": "", "text": "sodium hydroxide was added"} and corpus["d3"]["title"] == "zeta potential"
    assert sorted(queries) == ["q1", "q2"] and qrels == {"q1": {"d2": 1}, "q2": {"d3": 1}}
    t = fake_loader("x")
    t.is_multilingual = True
    with pytest.raises(NotImplementedError):
        X.split_data(t, "test")


def test_run_task_writes_the_phase_1_layout(tmp_path):
    leaf = tmp_path / "bm25_0.2.2" / "bm25" / "0.2.2"
    r = X.run_task("FakeRetrieval", leaf, loader=fake_loader, rev="0.2.2", top_k=3, seed=1, eval_split={"FakeRetrieval": "test"})
    assert r["status"] == "done" and r["ndcg_at_10"] == pytest.approx(1.0)               # title_in_text: q2 finds d3 through its title
    js = json.loads((leaf / "FakeRetrieval.json").read_text())
    sc = js["scores"]["test"][0]
    assert js["task_name"] == "FakeRetrieval" and js["dataset_revision"] == "deadbeef" and js["evaluation_time"] > 0
    assert sc["main_score"] == sc["ndcg_at_10"] == pytest.approx(1.0) and {"recall_at_10", "mrr_at_10", "map_at_100"} <= set(sc)
    meta = json.loads((leaf / "model_meta.json").read_text())
    assert meta["name"] == "bm25" and meta["revision"] == "0.2.2"
    preds = json.loads((leaf.parents[1] / "FakeRetrieval_default_predictions.json").read_text())
    assert set(preds) == {"q1", "q2"} and all(len(v) == 3 for v in preds.values())          # top_k docs per evaluated query
    man = json.loads((leaf.parents[1] / "run_manifest_FakeRetrieval.json").read_text())
    assert man["task"] == "FakeRetrieval" and man["n_docs"] == 4 and man["n_queries"] == 2 and man["model"]["k1"] == 1.5
    assert man["timing_s"]["mteb_evaluation_time_s"] == pytest.approx(js["evaluation_time"])
    # screen_top_k_models / the report read it like a dense model: leaf folder with model_meta.json + <Task>.json
    from tools import make_report
    rows = make_report.external_rows([tmp_path], eval_split={"FakeRetrieval": "test"}, n_queries={"FakeRetrieval": {"test": 2}})
    assert rows[0]["kind"] == "bm25" and rows[0]["FakeRetrieval__ndcg_at_10"] == pytest.approx(1.0)
    assert rows[0]["FakeRetrieval__split"] == "test" and rows[0]["FakeRetrieval__n_queries"] == 2
    # finished tasks are skipped, --force repeats
    before = (leaf / "FakeRetrieval.json").stat().st_mtime_ns
    assert X.run_task("FakeRetrieval", leaf, loader=fake_loader, rev="0.2.2", eval_split={"FakeRetrieval": "test"})["status"] == "skipped"
    assert (leaf / "FakeRetrieval.json").stat().st_mtime_ns == before
    assert X.run_task("FakeRetrieval", leaf, loader=fake_loader, rev="0.2.2", force=True, eval_split={"FakeRetrieval": "test"})["status"] == "done"


def test_a_task_with_two_eval_splits_gets_one_score_block_per_split(tmp_path):
    class Two(FakeTask):
        def __init__(self):
            base = fake_loader("x")
            super().__init__(base.corpus["test"], base.queries["test"], base.relevant_docs["test"])
            self.metadata = type("M", (), {"eval_splits": ["validation", "test"], "dataset": Meta.dataset})
            for attr in ("corpus", "queries", "relevant_docs"):
                getattr(self, attr)["validation"] = getattr(self, attr)["test"]

    leaf = tmp_path / "bm25_0.2.2" / "bm25" / "0.2.2"
    r = X.run_task("Two", leaf, loader=lambda n: Two(), rev="0.2.2", eval_split={"Two": "test"})
    assert r["split"] == "test" and r["n_queries"] == 2              # the configured split (params.phase1.eval_split), with its number of queries
    js = json.loads((leaf / "Two.json").read_text())
    assert set(js["scores"]) == {"validation", "test"}
    assert (leaf.parents[1] / "Two_validation_default_predictions.json").exists() and (leaf.parents[1] / "Two_test_default_predictions.json").exists()


def test_the_reported_split_is_the_configured_one_and_a_missing_one_raises(tmp_path):
    from pipeline_lib.phase1_split import Phase1SplitError
    t = FakeTask(*[getattr(fake_loader("x"), a)["test"] for a in ("corpus", "queries", "relevant_docs")], split="train")      # train only
    assert X.run_task("T", tmp_path / "a" / "bm25" / "v", loader=lambda n: t, rev="v", eval_split={"T": "train"})["split"] == "train"
    with pytest.raises(Phase1SplitError, match="not one of the task's splits"):
        X.run_task("T", tmp_path / "b" / "bm25" / "v", loader=lambda n: t, rev="v", eval_split={"T": "test"})            # no fallback to train
    with pytest.raises(Phase1SplitError, match="no evaluation split configured"):
        X.run_task("T", tmp_path / "c" / "bm25" / "v", loader=lambda n: t, rev="v", eval_split={"Other": "test"})


def test_main_reads_tasks_and_paths_from_the_config(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "load_mteb_task", fake_loader)
    cfg = {"params": {"seed": 3, "paths": {"artifacts_root": str(tmp_path / "art")}, "external": {"tasks": ["FakeA", "FakeB"], "output_subdir": "bl", "top_k": 10},
                      "phase1": {"eval_split": {"FakeA": "test", "FakeB": "test"}}},
           "stage6": {"bm25": {"k1": 1.2, "b": 0.6, "lowercase": True}}}
    p = tmp_path / "main.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert X.main(["--main-config", str(p)]) == 0
    root = tmp_path / "art" / "bl"
    leaves = list(root.glob("bm25_*/bm25/*"))
    assert len(leaves) == 1 and {f.name for f in leaves[0].glob("Fake*.json")} == {"FakeA.json", "FakeB.json"}
    man = json.loads(next(root.glob("bm25_*/run_manifest_FakeA.json")).read_text())
    assert man["model"]["k1"] == 1.2 and man["model"]["b"] == 0.6 and man["top_k"] == 10 and man["seeds"]["seed"] == 3
    rows = json.loads((root / "summary.json").read_text())
    assert {r["task"] for r in rows} == {"FakeA", "FakeB"} and (root / "summary.csv").exists()
    assert X.main(["--main-config", str(p), "--tasks", "FakeA"]) == 0                  # already finished: skipped, summary kept
    assert {r["task"] for r in json.loads((root / "summary.json").read_text())} == {"FakeA", "FakeB"}


def _leaf(root, name, rev, task, nd, rc):
    d = root / f"{name}_{rev}" / name / rev
    d.mkdir(parents=True)
    (d / "model_meta.json").write_text(json.dumps({"name": name, "revision": rev}))
    (d / f"{task}.json").write_text(json.dumps({"task_name": task, "evaluation_time": 1.0, "scores": {"test": [{"ndcg_at_10": nd, "recall_at_10": rc}]}}))


def test_stage_2_does_not_screen_the_baselines_unless_asked(tmp_path, monkeypatch, capsys):
    from stages import screen_top_k_models as S
    val = tmp_path / "validation"
    _leaf(val, "org_a", "r1", "ChemNQRetrieval", 0.5, 0.5)
    _leaf(val, "org_b", "r1", "ChemNQRetrieval", 0.4, 0.4)
    _leaf(val, "bm25", "0.2.2", "ChemNQRetrieval", 0.9, 0.9)                       # would be the best "model" and shift every rank
    out = tmp_path / "per_ds.csv"
    cfg = {"root_dir": str(tmp_path), "results_subdir": "validation", "metric_key": "ndcg_at_10", "top_percent": 0.5, "output_per_dataset_csv": str(out),
           "eval_split": {"ChemNQRetrieval": "test"}, "n_queries": {"ChemNQRetrieval": {"test": 27}}}

    def run(extra):
        c = tmp_path / "s2.yaml"
        c.write_text(yaml.safe_dump({**cfg, **extra}))
        monkeypatch.setattr(sys, "argv", ["screen", "--config", str(c)])
        S.main()
        return out.read_text()

    res = run({})
    assert "org_a" in res and "bm25" not in res and "org_b" not in res            # top 50% of two models: only a; the baseline took no rank
    assert "skip baseline" in capsys.readouterr().out
    assert "bm25" in run({"include_baselines": True})

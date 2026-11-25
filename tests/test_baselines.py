"""Baselines of the reviewer revision on tiny synthetic runs: hybrid timing, cross-encoder rerank (mock scorer), discovery, planning of
the rerank jobs, the run table and the summary tables of the report."""
import csv
import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("mteb")
pytest.importorskip("scipy")

from pipeline_lib import run_outputs
from tools import collect_runs, hybrid, rerank
from tools import make_report as R
from tools import metrics as M

DOCS = {"c0": "sodium chloride solution", "c1": "stirred under argon", "c2": "gamma ray spectra",
        "c3": "alpha beta catalyst palladium", "c4": "unrelated gardening", "c5": "alpha only"}
QUERIES = {"0": "alpha beta catalyst", "1": "gamma spectra"}
SPAN = {"0": "c3", "1": "c2"}
LEXICAL = {"0": "c5", "1": "c2"}                                   # the two label sets differ on purpose
DENSE0 = {"0": {f"c{i}": 6.0 - i for i in range(6)}, "1": {"c0": 6.0, "c4": 5.0, "c1": 4.0, "c2": 3.0, "c3": 2.0, "c5": 1.0}}
BM25 = {"0": {"c3": 9.0, "c5": 4.0, "c0": 1.0}, "1": {"c2": 9.0, "c1": 2.0, "c0": 1.0}}


def overlap_scorer(pairs):
    """Mock cross-encoder: number of words the query and the chunk share."""
    return [float(len(set(q.split()) & set(c.split()))) for q, c in pairs]


def prepared_dir(tmp_path):
    prep = tmp_path / "prep"
    task = prep / "mteb_task"
    task.mkdir(parents=True)
    (prep / "chunks").mkdir()
    (task / "corpus.jsonl").write_text("".join(json.dumps({"id": i, "title": "", "text": t}) + "\n" for i, t in DOCS.items()))
    (task / "queries.jsonl").write_text("".join(json.dumps({"id": i, "text": t}) + "\n" for i, t in QUERIES.items()))
    for name, lab in (("qrels_span_c090.jsonl", SPAN), ("qrels_lexical.jsonl", LEXICAL)):
        (task / name).write_text("".join(json.dumps({"query_id": q, "doc_id": d, "relevance": 1}) + "\n" for q, d in lab.items()))
    (prep / "chunks" / "chunks.jsonl").write_text("")
    return prep


def make_run(tests, prep, model, ranking, eval_time, chunker="fixed_token", size=128, overlap=0):
    """A finished dense / BM25 run folder (what stages/run_eval.py and run_bm25.py leave behind)."""
    d = tests / f"{model.replace('/', '_')}_{chunker}_c{size}_o{overlap}"
    out = d / "results" / model.replace("/", "_")
    out.mkdir(parents=True)
    (out / "ChemQuest_default_predictions.json").write_text(json.dumps(ranking))
    qrels = M.load_qrels(prep / "mteb_task" / "qrels_span_c090.jsonl")
    scores = run_outputs.mteb_style_scores(ranking, qrels, list(QUERIES))
    run_outputs.write_chemquest_json(out, scores, eval_time, method="test")
    run_outputs.write_per_query_outputs(out, prep / "mteb_task", "qrels_span_c090.jsonl", depth=100, run_name=model)
    (out / "run_manifest.json").write_text(json.dumps({"timing_s": {"mteb_evaluation_time_s": eval_time}, "data_sha256": {}}))
    (d / "run_info.json").write_text(json.dumps({"model": model, "chunker": chunker, "size": size, "overlap": overlap,
                                                  "prepared_dir": str(prep), "prepared_key": "K"}))
    for sub in ("chunks", "mteb_task"):
        (d / sub).symlink_to((prep / sub).resolve(), target_is_directory=True)
    (d / "results" / "summary.json").write_text("[]")
    return d


@pytest.fixture()
def world(tmp_path):
    prep = prepared_dir(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    dense = make_run(tests, prep, "org/dense", DENSE0, 100.0)
    bm = make_run(tests, prep, "bm25", BM25, 10.0)
    return tmp_path, prep, tests, dense, bm


# ---------------------------------------------------------------- hybrid timing
def test_hybrid_writes_chemquest_json_with_the_three_time_parts(world):
    _, prep, tests, dense, bm = world
    r = hybrid.run_hybrid(dense, bm, tests)
    out = Path(r["run_dir"]) / "results" / "hybrid_rrf60_org_dense_bm25"
    t = json.loads((out / "run_manifest.json").read_text())["timing_s"]
    assert t["dense_evaluation_time_s"] == pytest.approx(100.0) and t["bm25_evaluation_time_s"] == pytest.approx(10.0) and t["fusion_s"] > 0
    assert t["evaluation_time_s"] == pytest.approx(110.0 + t["fusion_s"]) == pytest.approx(t["mteb_evaluation_time_s"])
    cq = json.loads((out / "ChemQuest.json").read_text())
    assert cq["evaluation_time"] == pytest.approx(t["evaluation_time_s"]) and cq["scores"]["test"][0]["recall_at_10"] > 0
    assert json.loads((out / "metrics_summary.json").read_text())["cross_check"]["ok"]         # MTEB-style scores == our per-query metrics
    assert r["evaluation_time"] == pytest.approx(t["evaluation_time_s"])
    assert (out / "per_query_metrics__qrels_lexical.csv").exists()                              # both label sets


def test_hybrid_time_is_null_and_warns_when_an_input_has_no_time(world, capsys):
    _, prep, tests, dense, bm = world
    (dense / "results" / "org_dense" / "run_manifest.json").unlink()
    (dense / "results" / "org_dense" / "ChemQuest.json").unlink()
    r = hybrid.run_hybrid(dense, bm, tests)
    assert r["evaluation_time"] is None and "[WARN]" in capsys.readouterr().out
    out = Path(r["run_dir"]) / "results" / "hybrid_rrf60_org_dense_bm25"
    assert json.loads((out / "ChemQuest.json").read_text())["evaluation_time"] is None
    assert json.loads((out / "run_manifest.json").read_text())["timing_s"]["fusion_s"] > 0


# ---------------------------------------------------------------- rerank
def test_rerank_run_reorders_the_head_and_keeps_the_tail(world):
    _, prep, tests, dense, bm = world
    docs, queries = rerank.load_texts(prep / "mteb_task")
    new, raw, n = rerank.rerank_run(DENSE0, queries, docs, overlap_scorer, top_n=4)
    assert n == 8 and set(raw["0"]) == {"c0", "c1", "c2", "c3"}
    assert [d for d, _ in M.ranked(new["0"])] == ["c3", "c2", "c1", "c0", "c4", "c5"]    # head by score (ties: id descending), tail as before
    assert [d for d, _ in M.ranked(new["1"])][4:] == ["c3", "c5"]                         # q1 head = c0 c4 c1 c2 -> c2 first; tail c3 c5 untouched
    assert [d for d, _ in M.ranked(new["1"])][0] == "c2"
    assert min(new["0"][d] for d in raw["0"]) > max(new["0"][d] for d in ("c4", "c5"))   # the head is always above the tail
    same, _, _ = rerank.rerank_run(DENSE0, queries, docs, overlap_scorer, top_n=6)
    assert len(same["0"]) == 6                                                           # no tail: nothing below
    with pytest.raises(ValueError):
        rerank.rerank_run(DENSE0, queries, docs, overlap_scorer, top_n=7, depth=6)
    with pytest.raises(ValueError):
        rerank.rerank_run(DENSE0, queries, docs, overlap_scorer, top_n=0)


def test_run_rerank_layout_info_and_timing(world):
    _, prep, tests, dense, bm = world
    r = rerank.run_rerank(dense, tests, overlap_scorer, model="org/rr-model", revision="abc", top_n=4)
    assert r["status"] == "done"
    run_dir = Path(r["run_dir"])
    assert run_dir.name == "rerank_rr_model_top4_org_dense_fixed_token_c128_o0"
    out = run_dir / "results" / "rerank_rr_model_top4_org_dense"
    for f in ("ChemQuest.json", "ChemQuest_default_predictions.json", "run_top100.trec.gz", "per_query_metrics.csv", "per_query_metrics__qrels_lexical.csv",
              "metrics_summary.json", "run_manifest.json", "rerank_logits.json.gz"):
        assert (out / f).exists(), f
    assert (run_dir / "results" / "summary.json").exists() and (run_dir / "mteb_task").is_symlink() and (run_dir / "chunks").is_symlink()
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["rerank"] == {"model": "org/rr-model", "revision": "abc", "top_n": 4, "dense": "org/dense", "depth": 100}
    assert info["model"] == "rerank_rr_model_top4_org_dense" and info["prepared_key"] == "K"
    man = json.loads((out / "run_manifest.json").read_text())
    t = man["timing_s"]
    assert t["dense_evaluation_time_s"] == pytest.approx(100.0) and t["rerank_s"] > 0 and t["score_pairs_s"] <= t["rerank_s"]
    assert t["evaluation_time_s"] == pytest.approx(100.0 + t["rerank_s"]) == pytest.approx(t["mteb_evaluation_time_s"])
    assert json.loads((out / "ChemQuest.json").read_text())["evaluation_time"] == pytest.approx(t["evaluation_time_s"])
    assert man["n_pairs"] == 8 and man["method"] == "cross_encoder_rerank" and man["inputs"][0]["model"] == "org/dense"
    assert json.loads((out / "metrics_summary.json").read_text())["cross_check"]["ok"]
    with gzip.open(out / "rerank_logits.json.gz", "rt") as f:
        assert set(json.load(f)["0"]) == {"c0", "c1", "c2", "c3"}
    # span labels: c3 is now the top-1 for query 0 and c2 for query 1; the lexical labels (c5 for query 0) are scored too
    span = M.load_per_query_csv(out / "per_query_metrics.csv")
    lex = M.load_per_query_csv(out / "per_query_metrics__qrels_lexical.csv")
    assert span["0"]["ndcg_at_1"] == 1.0 and span["1"]["ndcg_at_1"] == 1.0 and lex["0"]["ndcg_at_1"] == 0.0


def test_rerank_is_resumable_and_only_for_dense_runs(world):
    _, prep, tests, dense, bm = world
    calls = []

    def counting(pairs):
        calls.append(len(pairs))
        return overlap_scorer(pairs)

    assert rerank.run_rerank(dense, tests, counting, revision="r", top_n=3)["status"] == "done"
    assert rerank.run_rerank(dense, tests, counting, revision="r", top_n=3)["status"] == "skipped" and len(calls) == 1
    assert rerank.run_rerank(dense, tests, counting, revision="r", top_n=3, force=True)["status"] == "done" and len(calls) == 2
    with pytest.raises(ValueError, match="only a dense run"):
        rerank.run_rerank(bm, tests, counting, revision="r", top_n=3)
    h = Path(hybrid.run_hybrid(dense, bm, tests)["run_dir"])
    with pytest.raises(ValueError, match="only a dense run"):
        rerank.run_rerank(h, tests, counting, revision="r", top_n=3)


def test_the_reranker_revision_must_be_pinned():
    with pytest.raises(ValueError, match="pinned"):
        rerank.CrossEncoderScorer("org/model", None)


def test_the_configured_reranker_is_pinned_to_a_commit():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "main_config.yaml").read_text())["params"]["rerank"]
    assert cfg["model"] == "BAAI/bge-reranker-v2-m3" and len(cfg["revision"]) == 40 and int(cfg["revision"], 16) >= 0
    assert cfg["top_n"] == 50 and cfg["ablation"] and all(isinstance(e["top_n"], list) for e in cfg["ablation"])


# ---------------------------------------------------------------- discovery and planning
def test_rerank_runs_are_never_dense_inputs_of_a_fusion(world):
    _, prep, tests, dense, bm = world
    rerank.run_rerank(dense, tests, overlap_scorer, revision="r", top_n=3)
    hybrid.run_hybrid(dense, bm, tests)
    cell = hybrid.discover(tests)[("fixed_token", 128, 0)]
    assert sorted(cell) == ["bm25", "org/dense"]                      # neither the rerank nor the hybrid run is an input
    from stages import rerun
    out = rerun.run_hybrids(str(tests.parent), [("org/dense", "r"), ("bm25", "rank_bm25")], ["fixed_token"], [(128, 0)], k=60, depth=100)
    assert [x["status"] for x in out] == ["skipped"]                  # one dense run only: no fusion of a rerank run was created
    assert not list(tests.glob("hybrid_rrf60_rerank*"))


def test_plan_jobs_primary_ablation_and_warning(tmp_path):
    prep = prepared_dir(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    for model in ("org/a", "org/b", "bm25"):
        for size in (128, 256):
            make_run(tests, prep, model, DENSE0, 1.0, size=size)
    cfg = {"top_n": 50, "models": ["org/a"], "ablation": [{"model": "a", "chunker": "fixed_token", "size": 256, "overlap": 0, "top_n": [20, 100]},
                                                           {"model": "org/b", "size": 999, "top_n": 20}]}
    jobs, warns = rerank.plan_jobs(hybrid.discover(tests), cfg)
    got = [(j[2]["model"], j[2]["size"], j[1]) for j in jobs]
    assert got == [("org/a", 128, 50), ("org/a", 256, 20), ("org/a", 256, 50), ("org/a", 256, 100)]      # b is not in params.rerank.models
    assert len(warns) == 1 and "size': 999" in warns[0]                                                 # the entry that matches nothing is reported
    jobs, _ = rerank.plan_jobs(hybrid.discover(tests), {"top_n": 50}, models=["org/b", "bm25"], wanted_cells=[("fixed_token", 128, 0)])
    assert [(j[2]["model"], j[2]["size"], j[1]) for j in jobs] == [("org/b", 128, 50)]                   # no models list: all dense runs; bm25 never
    jobs, _ = rerank.plan_jobs(hybrid.discover(tests), {"top_n": 50, "ablation": [{"top_n": 20}]})
    assert {j[1] for j in jobs} == {20, 50} and len(jobs) == 8                                          # 2 models x 2 sizes x 2 depths: no field = all
    assert rerank.plan_jobs({}, cfg)[0] == []


def test_run_configured_uses_one_scorer_for_all_jobs_and_resumes(tmp_path):
    prep = prepared_dir(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    make_run(tests, prep, "org/a", DENSE0, 5.0, size=256)

    class Scorer:
        loads = 0
        model_load_s = 1.5
        max_length = 8

        def load(self):
            Scorer.loads += 1

        def __call__(self, pairs):
            return overlap_scorer(pairs)

        def token_len(self, text):
            return len(text.split())

        def describe(self):
            return {"model": "mock"}

    cfg = {"model": "org/rr", "revision": "r", "top_n": 3, "models": ["org/a"], "ablation": [{"model": "a", "size": 256, "top_n": [2, 4]}]}
    res = rerank.run_configured(tests, cfg, scorer=Scorer())
    assert sorted((r["top_n"], r["status"]) for r in res) == [(2, "done"), (3, "done"), (4, "done")]
    assert Scorer.loads == 3                                          # load() is called before each job; one model object, loaded once by the real scorer
    again = rerank.run_configured(tests, cfg, scorer=Scorer())
    assert [r["status"] for r in again] == ["skipped"] * 3
    out = next(tests.glob("rerank_rr_top3_*")) / "results" / "rerank_rr_top3_org_a"
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["reranker"] == {"model": "mock"} and man["timing_s"]["model_load_s"] == 1.5 and man["pairs_truncated"]["max_length"] == 8
    assert man["pairs_truncated"]["n_pairs"] == 6 and man["pairs_truncated"]["n_truncated"] >= 0
    assert [r["status"] for r in rerank.run_configured(tests, cfg, scorer=Scorer(), dry_run=True)] == ["dry-run"] * 3


# ---------------------------------------------------------------- run table and report
def test_collect_runs_gives_hybrids_and_reranks_a_time_and_a_kind(world):
    _, prep, tests, dense, bm = world
    hybrid.run_hybrid(dense, bm, tests)
    rerank.run_rerank(dense, tests, overlap_scorer, revision="r", top_n=3)
    rows = {r["model"]: r for r in collect_runs.collect(tests)}
    assert {r["kind"] for r in rows.values()} == {"dense", "bm25", "hybrid", "rerank"}
    d, b = rows["org/dense"], rows["bm25"]
    h, rr = rows["hybrid_rrf60_org_dense_bm25"], rows["rerank_bge_reranker_v2_m3_top3_org_dense"]
    assert d["evaluation_time_s"] == pytest.approx(100.0) and b["evaluation_time_s"] == pytest.approx(10.0)
    assert h["evaluation_time_s"] > 110.0 and h["mteb_evaluation_time_s"] == pytest.approx(h["evaluation_time_s"]) and h["fusion_s"] > 0
    assert h["dense_evaluation_time_s"] == pytest.approx(100.0) and h["bm25_evaluation_time_s"] == pytest.approx(10.0)
    assert rr["evaluation_time_s"] > 100.0 and rr["rerank_s"] > 0 and rr["rerank_top_n"] == 3 and rr["source_dense"] == "org/dense"
    for r in rows.values():                                             # both label sets for every kind of run
        assert r["span_ndcg_at_10"] is not None and r["lexical_ndcg_at_10"] is not None


def test_comparison_pairs_include_the_primary_rerank_depth_only():
    names = ["a", "bm25", "rr20", "rr50"]
    infos = {"rr20|fixed_token|128|0": {"rerank": {"dense": "a", "top_n": 20}}, "rr50|fixed_token|128|0": {"rerank": {"dense": "a", "top_n": 50}}}
    assert R.comparison_pairs(names, infos, ("fixed_token", 128, 0), rerank_top_n=50) == [("a", "bm25"), ("rr50", "a"), ("rr50", "bm25")]    # + rerank vs BM25
    assert ("rr20", "a") in R.comparison_pairs(names, infos, ("fixed_token", 128, 0))


def _build(tmp_path, world_):
    root, prep, tests, dense, bm = world_
    d2 = make_run(tests, prep, "org/dense", DENSE0, 90.0, size=256)
    bm2 = make_run(tests, prep, "bm25", BM25, 8.0, size=256)
    for dd, bb in ((dense, bm), (d2, bm2)):
        hybrid.run_hybrid(dd, bb, tests)
    rerank.run_rerank(dense, tests, overlap_scorer, revision="r", top_n=4)
    for n in (2, 4):                                                   # depth ablation on the 256 run
        rerank.run_rerank(d2, tests, overlap_scorer, revision="r", top_n=n)
    qa = root / "qa.jsonl"
    qa.write_text("".join(json.dumps({"question": f"q{i}", "corpus_id": str(i), "content": "w", "start_index": 0, "end_index": 1}) + "\n" for i in range(2)))
    out = root / "results"
    R.build_report(root, out, qa, resamples=10, overlap=False, verbose=False, rerank_top_n=4, external_roots=[])
    return out


def test_report_writes_one_table_per_label_type_and_the_ablation(tmp_path, world):
    out = _build(tmp_path, world)
    for label in ("span", "lexical"):
        for ext in ("csv", "md"):
            assert (out / f"table_retrievers_{label}.{ext}").exists()
    span = list(csv.DictReader(open(out / "table_retrievers_span.csv")))
    lex = list(csv.DictReader(open(out / "table_retrievers_lexical.csv")))
    labels = [r["retriever"] for r in span]
    assert labels == ["org/dense", "bm25", "hybrid RRF: org/dense + bm25", "rerank@4: org/dense -> bge-reranker-v2-m3"]
    assert [r["kind"] for r in span] == ["dense", "bm25", "hybrid", "rerank"]           # the depth-2 run is not a row of the main table
    cols = [c for c in span[0] if c.startswith("fixed_token_c")]
    assert {c.split("__")[0] for c in cols} == {"fixed_token_c128_o0", "fixed_token_c256_o0"}        # the grid comes from the runs
    assert {c.split("__")[1] for c in cols} == {"geom_at_10", "ndcg_at_10", "recall_at_10", "evaluation_time_s"}
    rr, h = span[3], span[2]
    assert float(rr["fixed_token_c128_o0__evaluation_time_s"]) > 100 and float(h["fixed_token_c256_o0__evaluation_time_s"]) > 98
    assert span[0]["fixed_token_c128_o0__evaluation_time_s"] == lex[0]["fixed_token_c128_o0__evaluation_time_s"]       # time does not depend on the labels
    assert span[1]["fixed_token_c128_o0__ndcg_at_10"] != lex[1]["fixed_token_c128_o0__ndcg_at_10"]                   # labels differ (BM25 finds c3 / c5)
    assert float(span[0]["fixed_token_c128_o0__recall_at_10"]) == 1.0
    md = (out / "table_retrievers_span.md").read_text()
    assert "## Geom@10" in md and "## evaluation_time (s)" in md and "hybrid RRF: org/dense + bm25" in md
    abl = list(csv.DictReader(open(out / "table_rerank_ablation.csv")))
    assert [(r["dense"], r["size"], r["rerank_top_n"]) for r in abl] == [("org/dense", "256", "0"), ("org/dense", "256", "2"), ("org/dense", "256", "4")]
    assert abl[0]["rerank_s"] == "" and float(abl[1]["rerank_s"]) > 0 and abl[1]["lexical_geom_at_10"] != ""
    desc = (out / "README.md").read_text()
    assert "table_retrievers_span.csv" in desc and "evaluation_time (seconds)" in desc
    stats = list(csv.DictReader(open(out / "stats_compare.csv")))
    rr_name = "rerank_bge_reranker_v2_m3_top4_org_dense"
    assert {(r["A"], r["B"]) for r in stats if r["A"].startswith("rerank")} == {   # primary depth only: vs its dense run, vs BM25, vs the hybrid
        (rr_name, "org/dense"), (rr_name, "bm25"), (rr_name, "hybrid_rrf60_org_dense_bm25")}
    assert {r["family"] for r in stats if r["A"].startswith("rerank")} == {"rerank"}


def test_external_rows_put_bm25_next_to_the_dense_models(tmp_path):
    def leaf(root, name, rev, task, nd, rc, t, split="test"):
        d = root / f"{name}_{rev}" / name / rev
        d.mkdir(parents=True, exist_ok=True)
        (d / "model_meta.json").write_text(json.dumps({"name": name, "revision": rev}))
        (d / f"{task}.json").write_text(json.dumps({"task_name": task, "evaluation_time": t, "scores": {split: [{"ndcg_at_10": nd, "recall_at_10": rc}]}}))

    phase1, base = tmp_path / "validation", tmp_path / "baselines"
    leaf(phase1, "org_dense", "r1", "ChemNQRetrieval", 0.5, 0.8, 100.0)
    leaf(phase1, "org_dense", "r1", "ChemHotpotQARetrieval", 0.4, 0.6, 50.0, split="test")
    leaf(base, "bm25", "0.2.2", "ChemNQRetrieval", 0.45, 0.7, 2.0)
    leaf(tmp_path / "validation_results", "org_dense", "r1", "ChemNQRetrieval", 0.1, 0.1, 1.0)         # the first root wins
    rows = R.external_rows([phase1, base, tmp_path / "validation_results", tmp_path / "missing"])
    assert [(r["kind"], r["model"]) for r in rows] == [("bm25", "bm25"), ("dense", "org_dense")]
    bm, de = rows
    assert de["ChemNQRetrieval__geom_at_10"] == pytest.approx((0.5 * 0.8) ** 0.5) and de["ChemNQRetrieval__ndcg_at_10"] == 0.5
    assert de["ChemHotpotQARetrieval__recall_at_10"] == 0.6 and bm["ChemHotpotQARetrieval__geom_at_10"] is None and bm["ChemNQRetrieval__evaluation_time_s"] == 2.0
    assert de["ChemNQRetrieval__split"] == "test" and de["ChemNQRetrieval__n_queries"] == 27 and de["ChemHotpotQARetrieval__n_queries"] == 18
    assert R.external_rows([]) == []
    # a result file without the configured split stops the table: no fallback to train
    leaf(tmp_path / "other", "org_x", "r1", "ChemHotpotQARetrieval", 0.4, 0.6, 50.0, split="train")
    from pipeline_lib.phase1_split import Phase1SplitError
    with pytest.raises(Phase1SplitError):
        R.external_rows([tmp_path / "other"])


def test_report_includes_the_external_table(tmp_path, world):
    root = world[0]
    d = tmp_path / "ext" / "bm25_0.2.2" / "bm25" / "0.2.2"
    d.mkdir(parents=True)
    (d / "model_meta.json").write_text(json.dumps({"name": "bm25", "revision": "0.2.2"}))
    (d / "ChemNQRetrieval.json").write_text(json.dumps({"task_name": "ChemNQRetrieval", "evaluation_time": 3.0, "scores": {"test": [{"ndcg_at_10": 0.5, "recall_at_10": 0.5}]}}))
    qa = root / "qa.jsonl"
    qa.write_text("".join(json.dumps({"question": f"q{i}", "corpus_id": str(i), "content": "w", "start_index": 0, "end_index": 1}) + "\n" for i in range(2)))
    out = root / "res"
    R.build_report(root, out, qa, resamples=5, overlap=False, verbose=False, external_roots=[tmp_path / "ext"])
    rows = list(csv.DictReader(open(out / "table_external.csv")))
    assert rows[0]["model"] == "bm25" and float(rows[0]["ChemNQRetrieval__geom_at_10"]) == pytest.approx(0.5)
    assert "table_external.md" in (out / "README.md").read_text()


# ---------------------------------------------------------------- stages/rerun.py: the hybrid and rerank commands
def test_rerun_hybrid_and_rerank_commands_are_resumable(tmp_path, world, monkeypatch):
    pytest.importorskip("transformers")
    import yaml
    from stages import rerun
    root, prep, tests, dense, bm = world

    class Mock:
        def __init__(self, model, revision, **kw):
            self.revision = revision
            self.max_length = kw.get("max_length")
            self.model_load_s = 0.1

        def load(self):
            pass

        def __call__(self, pairs):
            return overlap_scorer(pairs)

        def describe(self):
            return {"model": "mock", "revision": self.revision}

    monkeypatch.setattr(rerank, "CrossEncoderScorer", Mock)
    cfg = {"params": {"paths": {"artifacts_root": str(root)}, "hybrid": {"enabled": True, "k": 60, "depth": 100},
                      "rerank": {"enabled": True, "model": "org/rr", "revision": "r", "top_n": 4, "models": [], "ablation": []}},
           "stage1": {"models": []}}
    mc = tmp_path / "main.yaml"
    mc.write_text(yaml.safe_dump(cfg))
    argv = ["--artifacts-root", str(root), "--main-config", str(mc), "--models", "org/dense@r1", "bm25@rank_bm25",
            "--chunkers", "fixed_token", "--sizes", "128", "--overlaps", "0"]
    assert rerun.main(["hybrid", *argv]) == 0
    assert (tests / "hybrid_rrf60_org_dense_bm25_fixed_token_c128_o0" / "results" / "summary.json").exists()
    assert not list(tests.glob("rerank_*"))                                              # the hybrid command does not rerank
    assert rerun.main(["rerank", *argv]) == 0
    rr = tests / "rerank_rr_top4_org_dense_fixed_token_c128_o0"
    assert (rr / "results" / "summary.json").exists() and json.loads((rr / "run_info.json").read_text())["rerank"]["revision"] == "r"
    stamp = (rr / "results" / "summary.json").stat().st_mtime_ns
    assert rerun.main(["rerank", *argv]) == 0 and (rr / "results" / "summary.json").stat().st_mtime_ns == stamp          # finished runs are skipped
    assert rerun.main(["rerank", "--no-rerank", *argv]) == 0 and len(list(tests.glob("rerank_*"))) == 1
    assert rerun.main(["rerank", "--dry-run", *argv]) == 0


def test_mteb_style_scores_break_ties_like_the_per_query_metrics():
    """Tied scores (RRF, fp16 logits): MTEB's mrr sorts stably, pytrec_eval breaks ties by id; the helper must give the per-query result."""
    run = {"0": {"a": 1.0, "b": 1.0, "c": 1.0}, "1": {"a": 2.0, "z": 1.0, "m": 1.0}}
    qrels = {"0": {"c": 1}, "1": {"z": 1}}                 # "c" and "z" win the id-descending tie-break
    sc = run_outputs.mteb_style_scores(run, qrels, ["0", "1"])
    pq = M.per_query_metrics(run, qrels, (1, 10), ["0", "1"])
    assert sc["mrr_at_1"] == pytest.approx(0.5) == pytest.approx((pq["0"]["mrr_at_1"] + pq["1"]["mrr_at_1"]) / 2)
    assert sc["ndcg_at_10"] == pytest.approx(sum(pq[q]["ndcg_at_10"] for q in pq) / 2, abs=1e-5)


def test_check_ablation_reports_entries_that_do_not_fit_the_grid():
    cfg = {"ablation": [{"model": "e5-large-v2", "chunker": "fixed_token", "size": 256, "overlap": 0, "top_n": [20, 100]}]}
    names = ["intfloat/e5-large-v2", "bge"]
    assert rerank.check_ablation(cfg, ["fixed_token", "recursive_token"], [(128, 0), (256, 0)], names) == []
    bad = rerank.check_ablation(cfg, ["fixed_token"], [(128, 0), (384, 0)], names)                        # 256 is not in the grid
    assert len(bad) == 1 and "matches no configuration" in bad[0] and "[128, 384]" in bad[0]
    assert len(rerank.check_ablation(cfg, ["recursive_token"], [(256, 0)], names)) == 1                  # chunker not in the grid
    assert len(rerank.check_ablation(cfg, ["fixed_token"], [(256, 16)], names)) == 1                      # overlap not in the grid
    assert "not among the evaluated models" in rerank.check_ablation(cfg, ["fixed_token"], [(256, 0)], ["bge"])[0]
    assert rerank.check_ablation({}, ["fixed_token"], [(256, 0)]) == [] and rerank.check_ablation({"ablation": [{"top_n": 20}]}, ["a"], [(1, 0)]) == []


def test_check_ablation_reports_rerank_models_that_are_not_on_the_model_list():
    cfg = {"models": ["intfloat/e5-large-v2"], "ablation": []}
    assert rerank.check_ablation(cfg, ["fixed_token"], [(256, 0)], ["intfloat/e5-large-v2", "bge"]) == []
    bad = rerank.check_ablation(cfg, ["fixed_token"], [(256, 0)], ["bge"])
    assert len(bad) == 1 and "params.rerank.models" in bad[0]


def test_rerun_aborts_before_any_run_when_the_ablation_does_not_fit_the_grid(tmp_path, world):
    pytest.importorskip("transformers")
    import yaml
    from stages import rerun
    root, prep, tests, dense, bm = world
    cfg = {"params": {"paths": {"artifacts_root": str(root)}, "hybrid": {"enabled": False},
                      "rerank": {"enabled": True, "model": "org/rr", "revision": "r", "top_n": 4, "models": [],
                                 "ablation": [{"model": "org/dense", "chunker": "fixed_token", "size": 999, "top_n": [2]}]}},
           "stage1": {"models": []}}
    mc = tmp_path / "main.yaml"
    mc.write_text(yaml.safe_dump(cfg))
    argv = ["--artifacts-root", str(root), "--main-config", str(mc), "--models", "org/dense@r1", "bm25@rank_bm25",
            "--chunkers", "fixed_token", "--sizes", "128", "--overlaps", "0"]
    before = sorted(p.name for p in tests.iterdir())
    for command in ("rerank", "eval", "all"):
        with pytest.raises(SystemExit, match="does not fit this grid"):
            rerun.main([command, *argv])
    assert sorted(p.name for p in tests.iterdir()) == before                      # nothing was started
    assert rerun.main(["rerank", "--no-rerank", *argv]) == 0                       # explicit opt-out
    assert rerun.main(["hybrid", *argv]) == 0                                      # steps that do not rerank are not blocked

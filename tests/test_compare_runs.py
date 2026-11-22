import json
from pathlib import Path

from tools import compare_runs as C
from tools import metrics as M

RUN = {"1": {"a": 0.9, "b": 0.8, "c": 0.1}, "2": {"b": 0.7, "a": 0.6}}
QRELS = {"1": {"a": 1}, "2": {"a": 1}}


def _folder(tmp, name, run):
    d = tmp / name; d.mkdir()
    M.write_trec_run(run, d / "run_top100.trec.gz", depth=100)
    M.write_per_query_csv(d / "per_query_metrics.csv", M.per_query_metrics(run, QRELS))
    return d


def test_identical_and_different_runs(tmp_path):
    a, b = _folder(tmp_path, "a", RUN), _folder(tmp_path, "b", RUN)
    r = C.compare(a, b)
    assert r["identical"] and r["run_files_identical"] and r["max_abs_score_diff_where_ranking_equal"] == 0
    c = _folder(tmp_path, "c", {"1": {"a": 0.9, "b": 0.95, "c": 0.1}, "2": RUN["2"]})
    r = C.compare(a, c)
    assert not r["identical"] and r["n_queries_with_different_ranking"] == 1 and r["first_different"] == ["1"]
    d = _folder(tmp_path, "d", {"1": {"a": 0.9000001, "b": 0.8, "c": 0.1}, "2": RUN["2"]})
    r = C.compare(a, d)
    assert r["identical"] and not r["run_files_identical"]   # tiny score noise, same ranking and metrics
    assert 0 < r["max_abs_score_diff_where_ranking_equal"] < 1e-6

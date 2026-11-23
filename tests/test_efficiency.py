"""Efficiency measures: pure helpers, CSV merge, latency subset, and that the probe does not alter what the model returns."""
import csv

import numpy as np
import pandas as pd

from pipeline_lib import efficiency as eff
from tools import efficiency_analysis as ana


class FakeModel:
    def __init__(self):
        self.timings = []
        self.calls = []

    def encode(self, sentences, batch_size=64, **kw):
        self.calls.append((list(sentences), batch_size, dict(kw)))
        return np.arange(len(sentences) * 3, dtype=np.float32).reshape(len(sentences), 3)


def test_percentile_and_spaced():
    assert eff.percentile([1, 2, 3, 4, 5], 50) == 3
    assert abs(eff.percentile(list(range(101)), 95) - 95) < 1e-9
    assert eff.percentile([], 95) is None
    assert eff._spaced(list(range(10)), 5) == [0, 2, 4, 6, 8]
    assert eff._spaced(list(range(3)), 0) == [0, 1, 2]


def test_index_size():
    assert eff.index_size_mb(1024 * 1024, 1, "fp32") == 4.0
    assert eff.index_size_mb(1024 * 1024, 1, "fp16") == 2.0


def test_probe_is_transparent_and_groups_passes():
    m = FakeModel()
    p = eff.EfficiencyProbe(m)
    for _ in range(2):                                   # two eval splits: queries, then corpus
        q = p.encode(["q1", "q2"], batch_size=8, task_name="T", prompt_type="query")
        c = p.encode(["a", "b", "c"], batch_size=8, task_name="T", prompt_type="passage")
        assert q.shape == (2, 3) and c.shape == (3, 3)
    assert m.calls[0] == (["q1", "q2"], 8, {"task_name": "T", "prompt_type": "query"})   # arguments forwarded unchanged
    assert p.n_passes("T") == 2
    last = p.record("T")
    assert last.passage_n == 3 and last.query_n == 2 and last.batch_size == 8 and last.queries == ["q1", "q2"]
    assert p.timings is m.timings                         # other attributes fall through


def test_latency_subset():
    o = eff.EfficiencyOptions(enabled=True, latency=True, chunker="fixed_token", size=256, overlap=0,
                              latency_sizes=[256, 512], latency_chunkers=["fixed_token"], latency_models=["bge"])
    assert o.latency_selected("BAAI/bge-m3")
    assert not o.latency_selected("intfloat/e5-base")
    o.size = 128
    assert not o.latency_selected("BAAI/bge-m3")
    assert not eff.EfficiencyOptions(enabled=True, latency=False).latency_selected("x")


def test_write_rows_merges(tmp_path):
    f = tmp_path / "e.csv"
    key = ["model", "revision", "chunker", "size", "overlap"]
    base = {"model": "m", "revision": "r", "chunker": "c", "size": 128, "overlap": 0}
    eff.write_rows(f, [{**base, "n_chunks": 10, "latency_total_ms_median": 5.0, "latency_measured": True}], key)
    eff.write_rows(f, [{**base, "n_chunks": 10, "peak_gpu_mem_mb": 99.0, "latency_measured": False, "latency_total_ms_median": None}], key)
    rows = list(csv.DictReader(open(f)))
    assert len(rows) == 1 and float(rows[0]["latency_total_ms_median"]) == 5.0 and float(rows[0]["peak_gpu_mem_mb"]) == 99.0
    eff.write_rows(f, [{**base, "size": 192, "n_chunks": 7}], key)
    assert len(list(csv.DictReader(open(f)))) == 2


def test_measure_latency_small():
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(50, 8)).astype(np.float32)
    out = eff.measure_latency(lambda t: rng.normal(size=(1, 8)).astype(np.float32), ["a", "b", "c"], corpus, warmup=2, n_queries=3)
    assert out["latency_n_queries"] == 3 and out["latency_top10_agree"] == 1.0
    assert out["latency_total_ms_p95"] >= out["latency_total_ms_median"] > 0


def test_pareto():
    d = pd.DataFrame({"x": [1, 2, 3, 4], "g": [0.5, 0.4, 0.7, 0.6]})
    assert list(ana.pareto(d, "x", "g").index) == [0, 2]

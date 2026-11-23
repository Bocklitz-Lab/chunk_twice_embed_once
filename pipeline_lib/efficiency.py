"""Deployment-relevant efficiency measures of a run (index size, model size, encoding cost, per-query latency).

Nothing here changes a retrieval result. ``EfficiencyProbe`` wraps the model that is handed to MTEB: it forwards every
``encode`` call unchanged (same arguments, same return value) and only watches it. The numbers are kept apart from the
existing stage timings: they go to ``efficiency.json`` next to the run's results and to ``efficiency_phase{1,2}.csv``,
never into ``run_manifest.json: timing_s`` or ``evaluation_time``.

Per configuration (one row; Phase 2 key: model, revision, chunker, size, overlap; Phase 1 key: model, revision, task):
  index     n_chunks, chunk token length (mean / median; reference tokenizer and the model's own), embedding dim,
            index size in MB (n_chunks x dim x 4 or 2 bytes)
  model     parameter count, weight size in MB (as loaded and as fp16), peak GPU memory while encoding the corpus
            (torch.cuda.max_memory_allocated, weights included) with the batch size used
  encoding  corpus encoding time and throughput (chunks/s), the sum of the corpus encode calls of this run
  latency   batch size 1 and a warm-up: query encoding, exact cosine top-10 search, and both together (median, p95, ms)
  hardware  GPU / CUDA / library versions (once per run, ``environment()``)

Latency search: like MTEB (``cos_sim`` + ``torch.topk``) the search runs on the CPU, on the float32 embeddings that the
run itself produced. The corpus matrix is normalised once (an index would be), the query is normalised per call; the top-10
is checked against MTEB's own ``cos_sim`` (``latency_top10_agree``). ``latency_device='cuda'`` times the search on the GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pipeline_lib.prompts import side_of

BYTES = {"fp32": 4, "fp16": 2}
UNKNOWN_TASK = "unknown"

PHASE2_KEY = ["model", "revision", "chunker", "size", "overlap"]
PHASE1_KEY = ["model", "revision", "task"]


# --------------------------------------------------------------------------------------------------
# options / CLI
# --------------------------------------------------------------------------------------------------
@dataclass
class EfficiencyOptions:
    """What to measure. ``enabled`` False = the probe is never created and nothing changes."""
    enabled: bool = False
    latency: bool = False
    latency_queries: int = 200          # queries timed per configuration (evenly spaced over the query list; 0 = all)
    latency_warmup: int = 10            # untimed queries before the timed ones
    latency_top_k: int = 10
    latency_device: str = "cpu"         # where the search runs; cpu = as in MTEB
    out_dir: Optional[str] = None       # folder of efficiency_phase{1,2}.csv (default: the run's output_root)
    phase: Optional[int] = None         # 1 / 2; default: 2 for a local task folder, 1 otherwise
    # identity of a Phase 2 configuration (passed by stages/rerun.py); the latency subset is matched against it
    chunker: Optional[str] = None
    size: Optional[int] = None
    overlap: Optional[int] = None
    # latency subset: empty list = no restriction on that field
    latency_chunkers: List[str] = field(default_factory=list)
    latency_sizes: List[int] = field(default_factory=list)
    latency_overlaps: List[int] = field(default_factory=list)
    latency_models: List[str] = field(default_factory=list)      # substring of the model name

    def latency_selected(self, model: str) -> bool:
        """Is latency measured for this configuration? (``--latency`` and the subset filters)"""
        if not (self.enabled and self.latency):
            return False
        if self.latency_chunkers and self.chunker not in self.latency_chunkers:
            return False
        if self.latency_sizes and self.size not in self.latency_sizes:
            return False
        if self.latency_overlaps and self.overlap not in self.latency_overlaps:
            return False
        if self.latency_models and not any(m in model for m in self.latency_models):
            return False
        return True


def add_cli_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("efficiency measures (pipeline_lib/efficiency.py); results and scores are not affected")
    g.add_argument("--efficiency", action="store_true", help="record index / model / encoding efficiency (efficiency_phase{1,2}.csv)")
    g.add_argument("--latency", action="store_true", help="also measure per-query latency (batch size 1, exact cosine top-10)")
    g.add_argument("--latency-queries", type=int, default=200, help="queries timed per configuration, evenly spaced (0 = all)")
    g.add_argument("--latency-warmup", type=int, default=10, help="untimed warm-up queries")
    g.add_argument("--latency-device", choices=["cpu", "cuda"], default="cpu", help="device of the search (cpu = as in MTEB)")
    g.add_argument("--latency-chunkers", nargs="*", default=[], help="latency only for these chunkers")
    g.add_argument("--latency-sizes", nargs="*", type=int, default=[], help="latency only for these chunk sizes")
    g.add_argument("--latency-overlaps", nargs="*", type=int, default=[], help="latency only for these overlaps")
    g.add_argument("--latency-models", nargs="*", default=[], help="latency only for models whose name contains one of these")
    g.add_argument("--efficiency-out", help="folder of the efficiency CSV (default: the run's output_root)")
    g.add_argument("--efficiency-phase", type=int, choices=[1, 2], help="CSV to write (default: 2 for a local task folder, else 1)")
    g.add_argument("--eff-chunker", help=argparse.SUPPRESS)
    g.add_argument("--eff-size", type=int, help=argparse.SUPPRESS)
    g.add_argument("--eff-overlap", type=int, help=argparse.SUPPRESS)


def options_from_args(args: argparse.Namespace) -> EfficiencyOptions:
    return EfficiencyOptions(
        enabled=bool(args.efficiency or args.latency), latency=bool(args.latency),
        latency_queries=args.latency_queries, latency_warmup=args.latency_warmup, latency_device=args.latency_device,
        out_dir=args.efficiency_out, phase=args.efficiency_phase,
        chunker=args.eff_chunker, size=args.eff_size, overlap=args.eff_overlap,
        latency_chunkers=list(args.latency_chunkers), latency_sizes=list(args.latency_sizes),
        latency_overlaps=list(args.latency_overlaps), latency_models=list(args.latency_models))


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def _cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _sync() -> None:
    t = _cuda()
    if t is not None:
        t.cuda.synchronize()


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in 0..100); None for an empty list."""
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def token_stats(lengths: Sequence[int]) -> Dict[str, Optional[float]]:
    if not len(lengths):
        return {"mean": None, "median": None, "max": None}
    return {"mean": sum(lengths) / len(lengths), "median": float(statistics.median(lengths)), "max": float(max(lengths))}


MB = 1024 ** 2          # "MB" in every column of this module is MiB (2^20 bytes)


def index_size_mb(n_chunks: int, dim: int, dtype: str = "fp32") -> float:
    """n_chunks x dim x bytes per value (fp32 4, fp16 2), in MB (MiB)."""
    return n_chunks * dim * BYTES[dtype] / MB


def model_size(st_model) -> Dict[str, Any]:
    """Parameter count and weight size of the loaded model (all modules of a SentenceTransformer; buffers excluded)."""
    params = list(st_model.parameters())
    n = sum(p.numel() for p in params)
    nbytes = sum(p.numel() * p.element_size() for p in params)
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in params})
    return {"n_params": n, "weights_mb": nbytes / MB, "weights_fp16_mb": n * 2 / MB, "weight_dtype": "+".join(dtypes) or None}


def environment() -> Dict[str, Any]:
    """GPU, CUDA and library versions (once per run); reuses the run manifest helpers."""
    from pipeline_lib.run_manifest import hardware_info, package_versions
    hw = hardware_info()
    env = {"gpu": hw.get("device_in_use"), "gpus": hw.get("gpus"), "cuda": hw.get("torch_cuda"), "cudnn": hw.get("cudnn"),
           "nvidia_driver": hw.get("nvidia_driver"), "python": hw.get("python"), "platform": hw.get("platform"),
           "packages": package_versions()}
    try:
        import torch
        env["torch_threads"] = torch.get_num_threads()
        env["cpu_count"] = os.cpu_count()
        if torch.cuda.is_available():
            env["gpu_total_mb"] = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / MB
    except Exception:
        pass
    return env


# --------------------------------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------------------------------
class _Pass:
    """What the probe saw during one evaluation pass of a task (one eval split: MTEB encodes the queries, then the corpus)."""
    def __init__(self):
        self.passage_s = 0.0
        self.passage_n = 0
        self.passage_calls = 0
        self.passage_peak_mb: Optional[float] = None
        self.query_s = 0.0
        self.query_n = 0
        self.query_peak_mb: Optional[float] = None
        self.batch_size: Optional[int] = None
        self.passages: List[str] = []
        self.embeddings: List[Any] = []
        self.queries: List[str] = []
        self.query_prompt_type: Any = None


class _TaskRecord:
    """All passes of a task. A task with several eval splits (ChemHotpotQA: train, dev, test) is encoded once per split; the
    efficiency row describes the LAST pass (the last eval split, ``test``), ``n_passes`` says how many there were."""
    def __init__(self):
        self.passes: List[_Pass] = []

    def current(self, new: bool = False) -> _Pass:
        if new or not self.passes:
            self.passes.append(_Pass())
        return self.passes[-1]

    @property
    def last(self) -> _Pass:
        return self.passes[-1]


class EfficiencyProbe:
    """Wraps the model given to ``MTEB.run``. ``encode`` is forwarded unchanged; the probe records, per task (MTEB passes
    ``task_name``), the time of the corpus / query encode calls, the peak GPU memory, the texts and the corpus embeddings
    (a reference, no copy). Everything else falls through to the wrapped model (as ``STNoPrompt`` does)."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tasks", {})
        object.__setattr__(self, "_weights_mb", None)

    @property
    def inner(self):
        return self._inner

    def task_names(self) -> List[str]:
        return list(self._tasks)

    def record(self, task: str) -> _Pass:
        """The last evaluation pass of ``task``."""
        return self._tasks[task].last

    def n_passes(self, task: str) -> int:
        return len(self._tasks[task].passes)

    def encode(self, sentences, batch_size=64, **kwargs):
        task = kwargs.get("task_name") or UNKNOWN_TASK
        side = side_of(kwargs.get("prompt_type"))
        task_rec = self._tasks.setdefault(task, _TaskRecord())
        rec = task_rec.current(new=(side == "query"))
        torch = _cuda()
        if torch is not None:
            _sync()
            if self._weights_mb is None:
                object.__setattr__(self, "_weights_mb", torch.cuda.memory_allocated() / MB)
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        emb = self._inner.encode(sentences, batch_size=batch_size, **kwargs)
        _sync()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / MB if torch is not None else None
        n = 1 if isinstance(sentences, str) else len(sentences)
        rec.batch_size = batch_size
        if side == "passage":
            rec.passage_s += dt
            rec.passage_n += n
            rec.passage_calls += 1
            if peak is not None:
                rec.passage_peak_mb = max(rec.passage_peak_mb or 0.0, peak)
            rec.passages += [sentences] if isinstance(sentences, str) else list(sentences)
            rec.embeddings.append(emb)
        elif side == "query":
            rec.query_s += dt
            rec.query_n += n
            if peak is not None:
                rec.query_peak_mb = max(rec.query_peak_mb or 0.0, peak)
            rec.queries += [sentences] if isinstance(sentences, str) else list(sentences)
            rec.query_prompt_type = kwargs.get("prompt_type")
        return emb

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


# --------------------------------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------------------------------
def _spaced(items: List[Any], n: int) -> List[Any]:
    """n evenly spaced items (deterministic, no random draw); all of them if n <= 0 or n >= len."""
    if n <= 0 or n >= len(items):
        return list(items)
    return [items[int(i * len(items) / n)] for i in range(n)]


def measure_latency(encode_one, queries: List[str], corpus_emb, *, top_k: int = 10, warmup: int = 10, n_queries: int = 200,
                    device: str = "cpu") -> Dict[str, Any]:
    """Per-query latency, batch size 1. ``encode_one(text) -> 1 x dim ndarray`` embeds one query (prompt included);
    ``corpus_emb`` is the n_chunks x dim matrix the run produced. Per query: encode (ms), search (ms), both (ms).
    Search = cosine against the whole corpus (``torch.mm`` of normalised vectors, as MTEB's ``cos_sim``), ``torch.topk``,
    result copied to the host as a list. The queries are evenly spaced over ``queries``; the warm-up repeats the first ones."""
    import numpy as np
    import torch
    from mteb.evaluation.evaluators.utils import cos_sim

    if not queries:
        return {}
    timed = _spaced(list(queries), n_queries)
    warm = [queries[i % len(queries)] for i in range(max(0, warmup))]
    dev = torch.device(device)
    C = torch.nn.functional.normalize(torch.as_tensor(np.asarray(corpus_emb, dtype=np.float32)), p=2, dim=1).to(dev)
    k = min(top_k, C.shape[0])

    def one(text: str):
        t0 = time.perf_counter()
        q = encode_one(text)
        _sync()
        t1 = time.perf_counter()
        qt = torch.nn.functional.normalize(torch.as_tensor(np.asarray(q, dtype=np.float32)).reshape(1, -1), p=2, dim=1).to(dev)
        sim = torch.mm(qt, C.T)
        sim[torch.isnan(sim)] = -1                       # as MTEB
        idx = torch.topk(sim, k, dim=1, largest=True, sorted=True)[1].cpu().tolist()[0]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3, idx, q

    for text in warm:
        one(text)
    enc, srch, tot, agree = [], [], [], 0
    for text in timed:
        e, s, t, idx, q = one(text)
        enc.append(e), srch.append(s), tot.append(t)
    # the same top-k with MTEB's own cos_sim, on a few of the queries (outside the timing)
    check = _spaced(timed, 20)
    Craw = torch.as_tensor(np.asarray(corpus_emb, dtype=np.float32))
    for text in check:
        _, _, _, idx, q = one(text)
        ref = torch.topk(cos_sim(torch.as_tensor(np.asarray(q, dtype=np.float32)), Craw), k, dim=1)[1].tolist()[0]
        agree += int(set(ref) == set(idx))
    out = {"latency_n_queries": len(timed), "latency_warmup": len(warm), "latency_top_k": k, "latency_device": device,
           "latency_top10_agree": agree / len(check) if check else None}
    for name, vals in (("query_encode", enc), ("search", srch), ("total", tot)):
        out[f"latency_{name}_ms_median"] = statistics.median(vals)
        out[f"latency_{name}_ms_p95"] = percentile(vals, 95)
        out[f"latency_{name}_ms_mean"] = sum(vals) / len(vals)
    return out


# --------------------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------------------
REF_TOKENIZER = "bert-base-uncased"                                  # params.chunking_tokenizer (pinned)
REF_TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"   # params.chunking_tokenizer_revision
_TOKENIZERS: Dict[tuple, Any] = {}


def load_tokenizer(name: str, revision: Optional[str] = None):
    key = (name, revision)
    if key not in _TOKENIZERS:
        from transformers import AutoTokenizer
        _TOKENIZERS[key] = AutoTokenizer.from_pretrained(name, revision=revision) if revision else AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[key]


def ref_token_lengths(texts: List[str], name: str = REF_TOKENIZER, revision: Optional[str] = REF_TOKENIZER_REVISION) -> List[int]:
    """Chunk length in tokens of the chunking (reference) tokenizer, the one that counts chunk_size, without special tokens.
    (``n_tokens`` in chunks.jsonl is another encoder's count and is not used.)"""
    tok = load_tokenizer(name, revision)
    lens: List[int] = []
    for i in range(0, len(texts), 2000):
        lens += [len(x) for x in tok(texts[i:i + 2000], add_special_tokens=False, truncation=False)["input_ids"]]
    return lens


def read_corpus_texts(data_dir: Path) -> List[str]:
    """The chunk text of every corpus entry of a prepared task folder (mteb_task/corpus.jsonl)."""
    out = []
    with open(Path(data_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)["text"])
    return out


def own_token_lengths(tokenizer, texts: List[str], batch: int = 2000) -> List[int]:
    lens: List[int] = []
    for i in range(0, len(texts), batch):
        lens += [len(x) for x in tokenizer(texts[i:i + batch], add_special_tokens=True, truncation=False)["input_ids"]]
    return lens


def static_fields(st_model, n_chunks: int, dim: int, ref_lens: Optional[Sequence[int]], own_lens: Optional[Sequence[int]]) -> Dict[str, Any]:
    """Index and model columns (items 1 and 2 without the GPU measurement): what the backfill and a live run share.
    ``ref_lens`` / ``own_lens``: chunk length in tokens of the chunking tokenizer / of the model's own tokenizer (no prompt)."""
    row: Dict[str, Any] = {"n_chunks": n_chunks, "embedding_dim": dim,
                           "index_mb_fp32": index_size_mb(n_chunks, dim, "fp32"), "index_mb_fp16": index_size_mb(n_chunks, dim, "fp16")}
    for tag, lens in (("ref", ref_lens), ("own", own_lens)):
        st = token_stats(lens or [])
        row[f"chunk_tokens_{tag}_mean"], row[f"chunk_tokens_{tag}_median"], row[f"chunk_tokens_{tag}_max"] = st["mean"], st["median"], st["max"]
    ms = model_size(st_model)
    row.update({"n_params": ms["n_params"], "weights_mb": ms["weights_mb"], "weights_fp16_mb": ms["weights_fp16_mb"],
                "weight_dtype": ms["weight_dtype"], "max_seq_length": getattr(st_model, "max_seq_length", None)})
    return row


def build_row(*, probe: EfficiencyProbe, task: str, key: Dict[str, Any], st_model, opts: EfficiencyOptions,
              data_dir: Optional[Path] = None, prompts: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """One efficiency row for ``task`` from what the probe saw (None if MTEB encoded nothing, e.g. results were reused)."""
    rec = probe.record(task) if task in probe.task_names() else None
    if rec is None or rec.passage_n == 0 or not rec.embeddings:
        return None
    import numpy as np
    emb = np.concatenate([np.asarray(e) for e in rec.embeddings], axis=0)
    n_chunks, dim = int(emb.shape[0]), int(emb.shape[1])
    row: Dict[str, Any] = dict(key)
    ref = ref_token_lengths(read_corpus_texts(Path(data_dir)) if data_dir is not None else rec.passages)
    own = own_token_lengths(st_model.tokenizer, rec.passages) if hasattr(st_model, "tokenizer") else None
    row.update(static_fields(st_model, n_chunks, dim, ref, own))
    row.update({"encode_batch_size": rec.batch_size, "peak_gpu_mem_mb": rec.passage_peak_mb,
                "weights_gpu_mem_mb": probe._weights_mb, "peak_gpu_mem_query_mb": rec.query_peak_mb,
                "corpus_encode_s": rec.passage_s, "corpus_chunks_per_s": rec.passage_n / rec.passage_s if rec.passage_s else None,
                "n_encode_calls": rec.passage_calls, "n_eval_passes": probe.n_passes(task)})
    row["latency_measured"] = False
    if opts.latency_selected(str(key.get("model", ""))) and rec.queries:
        # drop the timing records this adds to STNoPrompt.timings: they belong to the existing stage timings
        inner = probe.inner
        n_before = len(getattr(inner, "timings", []))
        pt = rec.query_prompt_type

        def encode_one(text: str):
            return inner.encode([text], batch_size=1, prompt_type=pt)
        try:
            row.update(measure_latency(encode_one, rec.queries, emb, top_k=opts.latency_top_k, warmup=opts.latency_warmup,
                                       n_queries=opts.latency_queries, device=opts.latency_device))
            row["latency_measured"] = True
        finally:
            if hasattr(inner, "timings"):
                del inner.timings[n_before:]
    row["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return row


# --------------------------------------------------------------------------------------------------
# CSV (one row per configuration; a re-measured configuration replaces its row)
# --------------------------------------------------------------------------------------------------
@contextmanager
def _locked(path: Path):
    lock = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as lf:
        try:
            import fcntl
            fcntl.flock(lf, fcntl.LOCK_EX)
        except Exception:
            pass
        yield


def _fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else v


def write_rows(path, rows: Iterable[Dict[str, Any]], key_cols: Sequence[str]) -> Path:
    """Merge ``rows`` into the CSV by key: one row per configuration. A new value replaces the old one, an empty value never
    erases one (so a backfill and a later live run, or a run without latency after one with, complete each other)."""
    path = Path(path)
    with _locked(path):
        old: Dict[tuple, Dict[str, Any]] = {}
        cols: List[str] = []
        if path.is_file() and path.stat().st_size:
            with open(path, newline="", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                cols = list(rd.fieldnames or [])
                for r in rd:
                    old[tuple(r.get(k, "") for k in key_cols)] = r
        for r in rows:
            k = tuple(str(r.get(c, "")) for c in key_cols)
            new = {c: _fmt(v) for c, v in r.items() if v is not None}       # a missing value never erases an earlier one
            if new.get("latency_measured") is False:                         # this run did not time latency: keep an earlier measurement
                new.pop("latency_measured")
            old[k] = {**old.get(k, {}), **new}
            cols += [c for c in old[k] if c not in cols]
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            for r in old.values():
                w.writerow(r)
        os.replace(tmp, path)
    return path


def csv_name(phase: int) -> str:
    return f"efficiency_phase{phase}.csv"


def finalize(probe: EfficiencyProbe, *, opts: EfficiencyOptions, st_model, model_name: str, revision: str, model_out_dir: Path,
             output_root: Path, tasks: List[str], data_dir: Optional[Path], prompts: Optional[Dict[str, Any]],
             normalize: bool) -> List[Dict[str, Any]]:
    """Build the rows of the run, write ``efficiency.json`` (rows + environment) next to the results and merge the rows into
    ``<out_dir>/efficiency_phase{1,2}.csv``. Returns the rows."""
    phase = opts.phase or (2 if data_dir is not None else 1)
    rows = []
    names = [t for t in probe.task_names()]
    for task in names:
        if tasks and task not in tasks and task != UNKNOWN_TASK:
            continue
        key: Dict[str, Any] = {"model": model_name, "revision": revision}
        if phase == 2:
            key.update({"chunker": opts.chunker, "size": opts.size, "overlap": opts.overlap})
        else:
            key["task"] = task
        row = build_row(probe=probe, task=task, key=key, st_model=st_model, opts=opts, data_dir=data_dir, prompts=prompts)
        if row is not None:
            row["normalize"], row["use_prompts"] = normalize, bool(prompts)
            rows.append(row)
    if not rows:
        print("[efficiency] nothing was encoded (existing MTEB results reused?): no efficiency row written")
        return rows
    payload = {"environment": environment(), "rows": rows, "options": {k: v for k, v in opts.__dict__.items()}}
    Path(model_out_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_out_dir) / "efficiency.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out = Path(opts.out_dir) if opts.out_dir else Path(output_root)
    p = write_rows(out / csv_name(phase), rows, PHASE2_KEY if phase == 2 else PHASE1_KEY)
    print(f"[efficiency] {len(rows)} row(s) -> {p}")
    return rows

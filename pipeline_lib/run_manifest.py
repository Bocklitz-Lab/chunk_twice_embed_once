"""Reproducibility helpers: seeds, run manifest (config, git, packages, hardware, data hashes, prompts, timing).

Every evaluation run writes ``run_manifest.json`` next to its results. Nothing here changes a result.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PACKAGES = (
    "mteb", "sentence-transformers", "transformers", "tokenizers", "torch", "numpy", "scipy", "pandas",
    "pytrec_eval-terrier", "rank-bm25", "rapidfuzz", "langchain", "langchain-experimental",
    "langchain-huggingface", "langchain-text-splitters", "huggingface-hub", "PyYAML",
)


def sha256_file(path: str | os.PathLike, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def set_seeds(seed: int) -> Dict[str, Any]:
    """Seed python, numpy and torch (CPU and CUDA). Returns what was seeded.

    ``PYTHONHASHSEED`` only works when set before the interpreter starts (the Makefile does that); the value
    seen by this process is recorded. The pipeline has no stochastic step today (inference, chunking, labels and
    ranking are deterministic); the seeds guard future sampling and the bootstrap tools.
    """
    seed = int(seed)
    random.seed(seed)
    done = {"seed": seed, "python": True, "numpy": False, "torch": False, "torch_cuda": False,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
        done["numpy"] = True
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        done["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            done["torch_cuda"] = True
    except Exception:
        pass
    return done


def _run(cmd: Iterable[str], cwd: Optional[str] = None) -> Optional[str]:
    try:
        return subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception:
        return None


def git_info(repo_root: str | os.PathLike) -> Dict[str, Any]:
    root = str(repo_root)
    commit = _run(["git", "rev-parse", "HEAD"], root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], root)
    diff = _run(["git", "diff", "HEAD"], root) if status else ""
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def package_versions(names: Iterable[str] = PACKAGES) -> Dict[str, Optional[str]]:
    from importlib import metadata
    out: Dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            out[n] = None
    return out


def cpu_info(cpuinfo: str = "/proc/cpuinfo", meminfo: str = "/proc/meminfo") -> Dict[str, Any]:
    """CPU model, logical / physical cores and total RAM from ``/proc`` and ``os`` (None where the system does not say).

    BM25, the rank fusion and the reranker's data handling run on the CPU, and the derived ``evaluation_time`` values add CPU and GPU time:
    the CPU is part of the hardware a time belongs to."""
    out: Dict[str, Any] = {"cpu_model": None, "cpu_logical_cores": os.cpu_count(), "cpu_physical_cores": None, "ram_total_gb": None}
    try:
        text = Path(cpuinfo).read_text(encoding="utf-8", errors="replace")
        models = [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.lower().startswith("model name") and ":" in ln]
        out["cpu_model"] = models[0] if models else None
        phys = {(a, b) for a, b in zip([ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("physical id")],
                                       [ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("core id")])}
        out["cpu_physical_cores"] = len(phys) or None
    except Exception:
        pass
    try:
        for ln in Path(meminfo).read_text(encoding="utf-8").splitlines():
            if ln.startswith("MemTotal:"):
                out["ram_total_gb"] = round(int(ln.split()[1]) / 1024 ** 2, 2)       # kB -> GiB
                break
    except Exception:
        pass
    return out


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": [],
        **cpu_info(),
    }
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        info["torch_threads"] = torch.get_num_threads()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["device_in_use"] = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    info["nvidia_driver"] = ",".join(dict.fromkeys(drv.split())) if drv else drv        # one driver version once, not once per GPU
    return info


def derived_run_environment(compute_device: str, seeds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``hardware`` and ``seeds`` blocks for the manifests of post-hoc runs (rank fusion, reranking), like those of an evaluation run.

    ``compute_device`` names where the step itself ran ("cpu", "cuda"); ``seeds`` is what ``set_seeds(params.seed)`` returned at the start of
    the step (None: no seed given). These steps contain no sampling; the block documents the machine the stored time belongs to."""
    return {"hardware": hardware_info(), "seeds": seeds, "compute_device": compute_device}


class Stopwatch:
    """Named wall-clock sections: ``with sw.section("model_load"): ...`` -> ``sw.seconds["model_load"]``."""

    def __init__(self):
        self.seconds: Dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - t0)


def summarize_encode_timings(timings: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Seconds spent in model.encode per side (passage = chunks, query = questions), from STNoPrompt.timings."""
    out = {"encode_passage_s": 0.0, "encode_query_s": 0.0, "encode_other_s": 0.0,
           "n_passage_texts": 0, "n_query_texts": 0, "n_calls": 0}
    for t in timings:
        side = t.get("side")
        key = {"passage": "passage", "query": "query"}.get(side, "other")
        out[f"encode_{key}_s"] += t["seconds"]
        if key != "other":
            out[f"n_{key}_texts"] += t["n_texts"]
        out["n_calls"] += 1
    return out


def build_manifest(
    *,
    repo_root: str | os.PathLike,
    config_path: Optional[str | os.PathLike] = None,
    resolved_config: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    prompts: Optional[Dict[str, Any]] = None,
    use_prompts: Optional[bool] = None,
    batch_size: Optional[int] = None,
    seeds: Optional[Dict[str, Any]] = None,
    data_files: Optional[Iterable[str | os.PathLike]] = None,
    timing: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg_text = Path(config_path).read_text(encoding="utf-8") if config_path and Path(config_path).exists() else None
    m: Dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "git": git_info(repo_root),
        "packages": package_versions(),
        "hardware": hardware_info(),
        "seeds": seeds,
        "model": model,
        "batch_size": batch_size,
        "use_prompts": use_prompts,
        "prompts": prompts,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest() if cfg_text is not None else None,
        "resolved_config": resolved_config,
        "data_sha256": {str(p): sha256_file(p) for p in (data_files or []) if Path(p).is_file()},
        "timing_s": timing,
    }
    if extra:
        m.update(extra)
    return m


def write_manifest(path: str | os.PathLike, manifest: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

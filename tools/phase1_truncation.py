"""Phase 1 truncation for runs that already exist (the published no-prompt run, or any run made before the report was added).

For every model of ``stage1.models`` and every external task (``ChemNQRetrieval``, ``ChemHotpotQARetrieval``): how many passages and queries are
longer than the model's maximum sequence length, counted with the model's OWN tokenizer on what MTEB embeds (``title + " " + text``), with
the special tokens, and with or without the model's prompt (``--prompts``). Same numbers as ``stages/model_screening.py`` now writes during
the run (``truncation_report_<Task>.json``, collected by ``make report``), computed from the pinned local snapshots and the cached datasets:
CPU only, offline, no weights are loaded.

  * a model whose pinned snapshot is not in the cache is skipped and listed (``table_truncation_phase1_skipped.csv``);
  * a task whose dataset is not cached is skipped and listed, the other tasks are still done.

The published Phase 1 ran WITHOUT prompts: ``--prompts none`` (default) reproduces that; ``--prompts config`` uses ``params.model_prompts``.

  python -m tools.phase1_truncation --out-dir results/v2-revision --prompts none
  python -m tools.phase1_truncation --out-dir /tmp/t --models intfloat/e5-large-v2 --tasks ChemNQRetrieval --cache-dir ~/.cache/huggingface/hub
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import model_info  # noqa: E402
from pipeline_lib.config_params import DEFAULT_MAIN_CONFIG  # noqa: E402
from pipeline_lib.prompts import resolve_prompts  # noqa: E402
from pipeline_lib.truncation import PHASE1_COLUMNS, phase1_report, phase1_rows  # noqa: E402


def _default_tokenizer(snap: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snap), local_files_only=True)


class CachedTask:
    """The parts of a loaded MTEB retrieval task the truncation count needs (``metadata.name`` / ``eval_splits``, ``corpus``, ``queries``,
    ``relevant_docs`` per split), read from the parquet files of a cached Hugging Face dataset snapshot."""

    def __init__(self, name: str, corpus: Dict[str, Dict[str, str]], queries: Dict[str, Dict[str, str]], qrels: Dict[str, Dict[str, Dict[str, int]]]):
        from types import SimpleNamespace
        self.metadata = SimpleNamespace(name=name, eval_splits=list(qrels))
        self.is_multilingual = False
        self.corpus, self.queries, self.relevant_docs, self.data_loaded = corpus, queries, qrels, True


def load_cached_task(name: str, cache_dir: Optional[str] = None, org: str = "BASF-AI") -> CachedTask:
    """Read ``datasets--<org>--<name>/snapshots/<ref main>/{corpus,queries,data}/*.parquet`` from the hub cache (no network, no ``datasets`` call)."""
    import pandas as pd
    from huggingface_hub import constants
    root = Path(cache_dir or constants.HF_HUB_CACHE) / f"datasets--{org}--{name}"
    ref = root / "refs" / "main"
    snaps = root / "snapshots"
    if not snaps.is_dir():
        raise FileNotFoundError(f"dataset {org}/{name} is not in the local cache ({root})")
    snap = snaps / ref.read_text().strip() if ref.is_file() and (snaps / ref.read_text().strip()).is_dir() else sorted(snaps.iterdir())[-1]
    corpus_df = pd.concat([pd.read_parquet(f) for f in sorted((snap / "corpus").glob("*.parquet"))])
    queries_df = pd.concat([pd.read_parquet(f) for f in sorted((snap / "queries").glob("*.parquet"))])
    corpus_all = {str(r["_id"]): {"title": r["title"], "text": r["text"]} for r in corpus_df.to_dict("records")}
    queries_all = {str(r["_id"]): r["text"] for r in queries_df.to_dict("records")}
    corpus: Dict[str, Any] = {}
    queries: Dict[str, Any] = {}
    qrels: Dict[str, Any] = {}
    for f in sorted((snap / "data").glob("*.parquet")):
        split = f.name.split("-")[0]
        rel: Dict[str, Dict[str, int]] = {}
        for r in pd.read_parquet(f).to_dict("records"):
            rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
        qrels[split], corpus[split], queries[split] = rel, corpus_all, queries_all
    return CachedTask(name, corpus, queries, qrels)


def _default_loader(name: str):
    """MTEB's own loader; if the dataset cannot be loaded through it offline, the cached parquet files."""
    try:
        from stages.run_bm25_mteb import load_mteb_task
        return load_mteb_task(name)
    except Exception:                                   # noqa: BLE001
        return load_cached_task(name)


def run(models: Sequence[Dict[str, Any]], tasks: Sequence[str], *, model_prompts: Dict[str, Any], use_prompts: bool, mask_instruction: bool = True,
        cache_dir: Optional[str] = None, loader: Optional[Callable[[str], Any]] = None,
        tokenizer_loader: Optional[Callable[[Path], Any]] = None) -> Dict[str, Any]:
    """{"rows": [...], "skipped": [...]} for ``models`` (dicts with name, revision) on the named tasks."""
    loader = loader or _default_loader
    tokenizer_loader = tokenizer_loader or _default_tokenizer
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    loaded: Dict[str, Any] = {}
    for t in tasks:
        try:
            loaded[t] = loader(t)
        except Exception as e:                       # noqa: BLE001 - dataset not cached (offline) or not loadable
            skipped.append({"model": "", "revision": "", "task": t, "reason": f"task not loadable: {type(e).__name__}: {str(e)[:120]}"})
    for m in models:
        name, rev = m["name"], m["revision"]
        snap = model_info.snapshot_dir(name, rev, cache_dir)
        if snap is None:
            skipped.append({"model": name, "revision": rev, "task": "", "reason": "pinned snapshot not in the local cache"})
            continue
        try:
            tok = tokenizer_loader(snap)
        except Exception as e:                       # noqa: BLE001
            skipped.append({"model": name, "revision": rev, "task": "", "reason": f"tokenizer not loadable: {type(e).__name__}"})
            continue
        msl = model_info.describe_snapshot(snap, int(getattr(tok, "model_max_length", 0)) or None)["max_seq_length"]
        prompts = resolve_prompts(name, model_prompts, use_prompts, mask_instruction=mask_instruction)
        for t, task in loaded.items():
            rows += phase1_rows(phase1_report(task, tok, msl, prompts, name, rev))
    return {"rows": rows, "skipped": skipped}


def _write(path: Path, rows: List[Dict[str, Any]], cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-config", default=str(DEFAULT_MAIN_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompts", choices=["none", "config"], default="none", help="none = the published runs (no prompt); config = params.model_prompts")
    ap.add_argument("--models", nargs="*", help="only these model names")
    ap.add_argument("--tasks", nargs="*", help="default: stage1.tasks")
    ap.add_argument("--cache-dir")
    args = ap.parse_args(argv)
    doc = yaml.safe_load(Path(args.main_config).read_text(encoding="utf-8"))
    p = doc["params"]
    models = [m for m in doc["stage1"]["models"] if not args.models or m["name"] in args.models]
    tasks = args.tasks or [t if isinstance(t, str) else t.get("name") for t in doc["stage1"]["tasks"]]
    res = run(models, tasks, model_prompts=p.get("model_prompts") or {}, use_prompts=args.prompts == "config",
              mask_instruction=bool(p.get("instructor_mask_instruction", False)), cache_dir=args.cache_dir)
    out = Path(args.out_dir)
    _write(out / "table_truncation_phase1.csv", res["rows"], PHASE1_COLUMNS)
    if res["skipped"]:
        _write(out / "table_truncation_phase1_skipped.csv", res["skipped"], ["model", "revision", "task", "reason"])
    n_models = len({r["model"] for r in res["rows"]})
    print(f"{len(res['rows'])} rows for {n_models} models -> {out / 'table_truncation_phase1.csv'} | skipped: {len(res['skipped'])}")
    for s in res["skipped"]:
        print(f"  skipped: {s['model'] or '-'} {s['task'] or '-'}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

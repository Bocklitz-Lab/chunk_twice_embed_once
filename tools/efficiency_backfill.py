"""Backfill the index and model columns of the efficiency table (items 1-2) from files that already exist: no run is repeated,
nothing is encoded, no score is read or written.

Phase 2 (default): for every finished run folder  <tests-root>/<model>_<chunker>_c<L>_o<O>/  (run_info.json + the linked
mteb_task/corpus.jsonl) it counts the chunks, measures the chunk length with the chunking tokenizer and with the model's own
tokenizer, loads the model on the CPU once for its parameter count, weight size and embedding dimension, and writes
<tests-root>/efficiency_phase2.csv (merged by key with any live measurement already there).

Phase 1: the models of a stage 1 config (``--config configs/stages/stage1.yaml``) and the corpora of the official tasks of its
``tasks`` list (cached Hugging Face datasets) -> efficiency_phase1.csv.

GPU peak memory, encoding time and latency need a run (``--efficiency`` / ``--latency`` of stages.rerun / run_eval).

  python -m tools.efficiency_backfill --tests-root artifacts/tests
  python -m tools.efficiency_backfill --tests-root artifacts/tests --models BAAI/bge-m3 --no-model   # chunk columns only
  python -m tools.efficiency_backfill --phase 1 --config configs/stages/stage1.yaml --out artifacts/embedding_models_screening/validation
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import efficiency as eff  # noqa: E402


def load_st(name: str, revision: Optional[str], model_kwargs: Optional[Dict[str, Any]] = None, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, revision=revision, device=device, **({"trust_remote_code": True} | (model_kwargs or {})))


def phase2_runs(tests_root: Path, models: List[str]) -> Dict[tuple, List[Dict[str, Any]]]:
    """{(model, revision): [run info + data_dir]} of the finished dense runs (hybrid / rerank / bm25 runs have no embeddings)."""
    out: Dict[tuple, List[Dict[str, Any]]] = {}
    for d in sorted(Path(tests_root).iterdir()):
        info_p = d / "run_info.json"
        if not info_p.is_file() or not (d / "mteb_task" / "corpus.jsonl").is_file():
            continue
        info = json.loads(info_p.read_text(encoding="utf-8"))
        if info.get("hybrid") or info.get("rerank") or str(info["model"]).startswith("bm25"):
            continue
        if models and not any(m in info["model"] for m in models):
            continue
        out.setdefault((info["model"], info["revision"]), []).append({**info, "data_dir": d / "mteb_task"})
    return out


def backfill_phase2(tests_root: Path, out_dir: Path, models: List[str], no_model: bool, device: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (name, rev), runs in phase2_runs(tests_root, models).items():
        st = None if no_model else load_st(name, rev, device=device)
        ref_cache: Dict[str, List[int]] = {}
        for r in runs:
            texts = eff.read_corpus_texts(r["data_dir"])
            prep = str(Path(r["data_dir"]).resolve())
            if prep not in ref_cache:
                ref_cache[prep] = eff.ref_token_lengths(texts)
            row: Dict[str, Any] = {"model": name, "revision": rev, "chunker": r["chunker"], "size": r["size"], "overlap": r["overlap"],
                                   "n_chunks": len(texts)}
            if st is not None:
                own = eff.own_token_lengths(st.tokenizer, texts) if hasattr(st, "tokenizer") else None
                row.update(eff.static_fields(st, len(texts), st.get_sentence_embedding_dimension(), ref_cache[prep], own))
            else:
                row.update({f"chunk_tokens_ref_{k}": v for k, v in eff.token_stats(ref_cache[prep]).items()})
            rows.append(row)
        print(f"{name}: {len(runs)} configurations")
        del st
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    eff.write_rows(out_dir / eff.csv_name(2), rows, eff.PHASE2_KEY)
    return rows


def backfill_phase1(config: Path, out_dir: Path, models: List[str], no_model: bool, device: str) -> List[Dict[str, Any]]:
    import yaml
    from mteb import get_tasks
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    corp: Dict[str, List[str]] = {}
    for t in get_tasks(tasks=[x for x in cfg["tasks"] if isinstance(x, str)]):
        t.load_data()
        split = t.metadata.eval_splits[-1]        # the last eval split, as the live probe (ChemHotpotQA: train, dev, test)
        corp[t.metadata.name] = [((d.get("title", "") + " " + d["text"]).strip() if isinstance(d, dict) else str(d))
                                 for d in t.corpus[split].values()]
    rows: List[Dict[str, Any]] = []
    for m in cfg["models"]:
        if models and not any(x in m["name"] for x in models):
            continue
        st = None if no_model else load_st(m["name"], m.get("revision"), m.get("model_kwargs"), device)
        for task, texts in corp.items():
            ref = eff.ref_token_lengths(texts)
            row: Dict[str, Any] = {"model": m["name"], "revision": m.get("revision"), "task": task, "n_chunks": len(texts)}
            if st is not None:
                own = eff.own_token_lengths(st.tokenizer, texts) if hasattr(st, "tokenizer") else None
                row.update(eff.static_fields(st, len(texts), st.get_sentence_embedding_dimension(), ref, own))
            rows.append(row)
        print(f"{m['name']}: {len(corp)} tasks")
        del st
        gc.collect()
    eff.write_rows(out_dir / eff.csv_name(1), rows, eff.PHASE1_KEY)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 2], default=2)
    ap.add_argument("--tests-root", help="phase 2: folder of the run folders (e.g. artifacts/tests)")
    ap.add_argument("--config", help="phase 1: stage 1 config (models + tasks)")
    ap.add_argument("--out", help="folder of the CSV (default: --tests-root; phase 1: required)")
    ap.add_argument("--models", nargs="*", default=[], help="only models whose name contains one of these")
    ap.add_argument("--no-model", action="store_true", help="do not load models: chunk columns only (no params / dim / sizes)")
    ap.add_argument("--device", default="cpu", help="device to load the models on (cpu: no GPU needed)")
    a = ap.parse_args(argv)
    if a.phase == 2:
        if not a.tests_root:
            ap.error("--tests-root is required for phase 2")
        out = Path(a.out or a.tests_root)
        rows = backfill_phase2(Path(a.tests_root), out, a.models, a.no_model, a.device)
    else:
        if not (a.config and a.out):
            ap.error("--config and --out are required for phase 1")
        out = Path(a.out)
        rows = backfill_phase1(Path(a.config), out, a.models, a.no_model, a.device)
    print(f"{len(rows)} rows -> {out / eff.csv_name(a.phase)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

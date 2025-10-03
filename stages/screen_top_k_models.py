import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
import argparse
import math

# ------------------ RRF helpers ------------------

def rrf_beta(recall: Optional[float], ndcg: Optional[float], beta: float = 2.0) -> Optional[float]:
    if recall is None or ndcg is None:
        return None
    if recall <= 0.0 or ndcg <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * recall * ndcg / (b2 * ndcg + recall)

def pick_split(scores_obj: Dict, split_pref: List[str]) -> Optional[Dict]:
    """
    scores_obj looks like {"train": [ {...} ], "test": [ {...} ], ...}
    Return the first matching split dict (the first element of the list)
    """
    for sp in split_pref:
        if sp in scores_obj and scores_obj[sp]:
            return scores_obj[sp][0]
    # If none match, fall back to any available
    for v in scores_obj.values():
        if isinstance(v, list) and v:
            return v[0]
    return None

def find_leaf_dirs(results_dir: Path) -> List[Path]:
    """
    Return directories that contain a model_meta.json (your leaf level),
    e.g.: validation/<folder>/<model_name>/<revision>/
    """
    return [p.parent for p in results_dir.glob("**/model_meta.json")]

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_model_meta(leaf_dir: Path) -> Dict:
    meta_path = leaf_dir / "model_meta.json"
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def derive_model_id(leaf_dir: Path, meta: Dict, fmt: str) -> str:
    name = meta.get("name")  # e.g., "BAAI/bge-base-en-v1.5"
    rev = meta.get("revision")
    if fmt == "name@rev":
        if name and rev:
            return f"{name}@{rev}"
        elif name:
            return name
    elif fmt == "name":
        if name:
            return name
    # fallback to outermost folder name under 'validation'
    # e.g., validation/BAAI__bge-base-en-v1.5_a5beb1e.../...
    return leaf_dir.parents[2].name if len(leaf_dir.parents) >= 3 else leaf_dir.name

def read_json(fp: Path) -> dict:
    try:
        with open(fp, "r") as f:
            return json.load(f)
    except Exception:
        return {}

# ------------------ metric extraction ------------------

def _get_metric_at_k(
    split_dict: Dict,
    base: str,
    k: int,
    allow_nearest: bool = True,
    candidates: List[int] = None
) -> Tuple[Optional[float], Optional[int]]:
    """
    Try exact "<base>_at_<k>", else nearest among candidates if allowed.
    Returns (value, used_k).
    """
    if candidates is None:
        candidates = [1, 3, 5, 10, 20, 50, 100]
    exact_key = f"{base}_at_{k}"
    if exact_key in split_dict and isinstance(split_dict[exact_key], (int, float)):
        return float(split_dict[exact_key]), k

    if not allow_nearest:
        return None, None

    found: List[Tuple[int, float]] = []
    for ck in candidates:
        key = f"{base}_at_{ck}"
        v = split_dict.get(key)
        if isinstance(v, (int, float)):
            found.append((ck, float(v)))
    if not found:
        # final fallback: raw "base" key, if present
        v = split_dict.get(base)
        if isinstance(v, (int, float)):
            return float(v), k
        return None, None

    # choose nearest k by absolute distance
    used_k, val = min(found, key=lambda t: abs(t[0] - k))
    return val, used_k

def compute_metric_value(
    split_dict: Dict,
    metric_key: str,
    cfg: Dict
) -> Tuple[Optional[float], Dict[str, Optional[int]]]:
    """
    Return (metric_value, extras) where extras may include which k was used for recall/ndcg.
    Supports:
      - any direct metric key (e.g., 'ndcg_at_10', 'recall_at_10', 'main_score', etc.)
      - "rrf" composite:
          Set config:
            metric_key: "rrf"
            rrf:
              k: 10
              beta: 2.0
              allow_nearest: true
              fallback_k_candidates: [1,3,5,10,20,50,100]
    """
    if metric_key.lower() != "rrf":
        # direct metric path (existing behavior)
        val = split_dict.get(metric_key, split_dict.get("main_score"))
        return (float(val) if isinstance(val, (int, float)) else None, {})

    # RRF mode
    rrf_cfg = cfg.get("rrf", {}) or {}
    k = int(rrf_cfg.get("k", 10))
    beta = float(rrf_cfg.get("beta", 2.0))
    allow_nearest = bool(rrf_cfg.get("allow_nearest", True))
    cand = rrf_cfg.get("fallback_k_candidates") or [1, 3, 5, 10, 20, 50, 100]

    recall, k_rec = _get_metric_at_k(split_dict, "recall", k, allow_nearest, cand)
    ndcg,  k_ndc = _get_metric_at_k(split_dict, "ndcg",  k, allow_nearest, cand)

    val = rrf_beta(recall, ndcg, beta=beta)
    extras = {"k_used_recall": k_rec, "k_used_ndcg": k_ndc}
    return val, extras

def read_dataset_file(
    fp: Path,
    split_pref: List[str],
    metric_key: str,
    cfg: Dict
) -> Optional[Tuple[str, float, Dict]]:
    """
    Returns (task_name, metric_value, meta_extras) or None if not available.
    meta_extras may include which k was used for recall/ndcg in RRF mode.
    """
    if fp.name == "model_meta.json" or not fp.name.endswith(".json"):
        return None
    try:
        with open(fp, "r") as f:
            js = json.load(f)
    except Exception:
        return None

    task_name = js.get("task_name")
    scores = js.get("scores", {})
    split_dict = pick_split(scores, split_pref)
    if not split_dict:
        return None

    metric_val, extras = compute_metric_value(split_dict, metric_key, cfg)
    if metric_val is None:
        return None

    return task_name, float(metric_val), extras

# ------------------ main ------------------

def main():
    parser = argparse.ArgumentParser(description="Screen embedding models by a chosen metric (supports nDCG@10 and RRF@k).")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)  # <-- uses the arg
    root_dir = Path(cfg["root_dir"]).expanduser().resolve()
    results_dir = (root_dir / cfg.get("results_subdir", "validation")).resolve()
    split_pref = cfg.get("split_preference", ["test", "validation", "train"])
    metric_key = cfg.get("metric_key", "ndcg_at_10")  # set to "rrf" in config to use RRF@k
    top_pct = float(cfg.get("top_percent", 0.30))
    datasets_include = set(cfg.get("datasets_include") or [])
    model_id_fmt = cfg.get("model_id_format", "name@rev")

    out_union = cfg.get("output_union_csv", "")
    out_per_ds = cfg.get("output_per_dataset_csv", "")

    leaf_dirs = find_leaf_dirs(results_dir)
    if not leaf_dirs:
        raise SystemExit(f"No model_meta.json found under: {results_dir}")

    rows = []
    for leaf in leaf_dirs:
        meta = read_model_meta(leaf)
        model_id = derive_model_id(leaf, meta, model_id_fmt)
        # read every dataset JSON in the same leaf dir
        for fp in leaf.glob("*.json"):
            if fp.name == "model_meta.json":
                continue
            parsed = read_dataset_file(fp, split_pref, metric_key, cfg)
            if not parsed:
                continue
            task_name, metric_val, extras = parsed
            if not task_name:
                # fallback: derive from filename (e.g., ChemNQRetrieval.json)
                task_name = fp.stem
            if datasets_include and task_name not in datasets_include:
                continue

            # determine which split was used (optional trace)
            js = read_json(fp)
            scores = js.get("scores", {})
            split_used = None
            for sp in split_pref:
                if sp in scores:
                    split_used = sp
                    break

            row = {
                "model_id": model_id,
                "model_name": meta.get("name"),
                "revision": meta.get("revision"),
                "folder": leaf.parents[2].name if len(leaf.parents) >= 3 else leaf.name,
                "dataset": task_name,
                metric_key: metric_val,
                "split_used": split_used
            }
            # include RRF bookkeeping if applicable
            if metric_key.lower() == "rrf":
                row.update({
                    "rrf_k_used_recall": extras.get("k_used_recall"),
                    "rrf_k_used_ndcg": extras.get("k_used_ndcg"),
                    "rrf_beta": float((cfg.get("rrf", {}) or {}).get("beta", 2.0)),
                })
            rows.append(row)

    if not rows:
        raise SystemExit("No metrics found. Check config (datasets_include, split_preference) and directory structure.")

    df = pd.DataFrame(rows)

    # Rank within each dataset by the metric (higher is better)
    df["rank_in_ds"] = df.groupby("dataset")[metric_key].rank(ascending=False, method="min")
    df["count_in_ds"] = df.groupby("dataset")[metric_key].transform("count")
    df["rank_pct"] = df["rank_in_ds"] / df["count_in_ds"]

    passed = df[df["rank_pct"] <= top_pct].copy()
    # Union across datasets
    passed_union = sorted(passed["model_id"].unique())

    # Print summary
    datasets = sorted(df["dataset"].unique())
    print(f"Datasets found: {datasets}")
    print(f"Total models (unique model_id): {df['model_id'].nunique()}")
    print(f"Kept rows (top {int(top_pct*100)}% per dataset): {len(passed)}")

    print("\n=== Passed models (UNION across datasets) ===")
    for m in passed_union:
        print(m)

    # Optional: write CSVs
    if out_per_ds:
        cols = ["model_id", "model_name", "revision", "dataset", metric_key, "rank_in_ds", "count_in_ds", "rank_pct", "split_used"]
        # include RRF extras if present
        if metric_key.lower() == "rrf":
            cols += ["rrf_k_used_recall", "rrf_k_used_ndcg", "rrf_beta"]
        passed[cols].sort_values(["dataset", "rank_in_ds"]).to_csv(out_per_ds, index=False)
        print(f"\nWrote per-dataset passes to: {out_per_ds}")
    if out_union:
        pd.Series(passed_union, name="model_id").to_csv(out_union, index=False)
        print(f"Wrote union list to: {out_union}")

if __name__ == "__main__":
    main()

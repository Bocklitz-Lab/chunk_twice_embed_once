import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
import argparse
import math

# ------------------ helpers ------------------

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
    name = meta.get("name")
    rev = meta.get("revision")
    if fmt == "name@rev":
        if name and rev:
            return f"{name}@{rev}"
        elif name:
            return name
    elif fmt == "name":
        if name:
            return name
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
    k: int
) -> Tuple[Optional[float], Optional[int]]:
    """
    Only try exact '<base>_at_<k>'. If missing, final fallback: raw 'base' key.
    Returns (value, used_k) where used_k is k (or None if raw base was used).
    """
    exact_key = f"{base}_at_{k}"
    if exact_key in split_dict and isinstance(split_dict[exact_key], (int, float)):
        return float(split_dict[exact_key]), k

    # Final fallback: raw "base" key, if present (kept for robustness)
    v = split_dict.get(base)
    if isinstance(v, (int, float)):
        return float(v), None
    return None, None

def _time_score_from_split(split_dict: Dict, cfg: Dict) -> Tuple[Optional[float], Dict]:
    """
    Build a [0,1]-ish time score where lower eval time => higher score.
    Config (example):
      geom_mean:
        time:
          # If you already have a normalized score field in [0,1]
          # score_key: "time_score"

          # Otherwise derive from raw time values
          value_keys: ["evaluation_time_s", "eval_seconds", "time_s", "evaluation_time"]
          unit: "seconds"            # or "ms"/"milliseconds"
          transform: "inverse"       # "inverse" (default), "qps", or "identity"
          reference_seconds: 1.0
          reference_qps: 100.0
    """
    gm_cfg = (cfg.get("geom_mean") or {})
    tcfg = (gm_cfg.get("time") or {})

    # 1) ready-to-use score
    score_key = tcfg.get("score_key")
    if score_key:
        v = split_dict.get(score_key)
        if isinstance(v, (int, float)):
            return float(v), {"time_score_source": score_key, "time_score_transform": "identity"}

    # 2) derive from raw time value
    value_keys = tcfg.get("value_keys") or ["evaluation_time_s", "eval_time_s", "eval_seconds",
                                            "evaluation_time", "latency_s", "elapsed_s", "time_s", "time"]
    unit = str(tcfg.get("unit", "seconds")).lower()
    transform = str(tcfg.get("transform", "inverse")).lower()

    for key in value_keys:
        v = split_dict.get(key)
        if not isinstance(v, (int, float)):
            continue

        time_s = float(v)
        if unit in ("ms", "millisecond", "milliseconds"):
            time_s = time_s / 1000.0

        if transform == "inverse":
            ref = float(tcfg.get("reference_seconds", 1.0)) or 1.0
            time_score = 1.0 / (1.0 + (time_s / ref))
        elif transform == "qps":
            if time_s <= 0:
                continue
            qps = 1.0 / time_s
            ref_qps = float(tcfg.get("reference_qps", 100.0)) or 100.0
            time_score = min(1.0, qps / ref_qps)
        elif transform == "identity":
            time_score = float(v)
        else:
            time_score = 1.0 / (1.0 + time_s)

        return time_score, {
            "time_value_seconds": time_s,
            "time_score_source": key,
            "time_score_transform": transform
        }

    return None, {}

def compute_metric_value(
    split_dict: Dict,
    metric_key: str,
    cfg: Dict
) -> Tuple[Optional[float], Dict]:
    """
    Return (metric_value, extras).
    Supports:
      - direct metrics (e.g., 'ndcg_at_10', 'recall_at_10', 'main_score', etc.)
      - "geom_mean": weighted geometric mean of ndcg@k, recall@k, time_score
    """
    mk = metric_key.lower()

    if mk != "geom_mean":
        val = split_dict.get(metric_key, split_dict.get("main_score"))
        return (float(val) if isinstance(val, (int, float)) else None, {})

    gm_cfg = cfg.get("geom_mean", {}) or {}
    k = int(gm_cfg.get("k", 10))

    # weights
    w_ndcg = float(gm_cfg.get("weights", {}).get("ndcg", 1.0))
    w_recall = float(gm_cfg.get("weights", {}).get("recall", 1.0))
    w_time = float(gm_cfg.get("weights", {}).get("time", 1.0))
    w_sum = w_ndcg + w_recall + w_time
    if w_sum <= 0:
        return None, {"error": "weights_sum_nonpositive"}

    # components
    ndcg, k_ndc = _get_metric_at_k(split_dict, "ndcg", k)
    recall, k_rec = _get_metric_at_k(split_dict, "recall", k)
    time_score, time_meta = _time_score_from_split(split_dict, cfg)

    if ndcg is None or recall is None or time_score is None:
        return None, {"k_used_recall": k_rec, "k_used_ndcg": k_ndc, **time_meta}

    # allow zeros -> product 0; negatives (shouldn't happen) clip to >=0
    ndcg = max(0.0, float(ndcg))
    recall = max(0.0, float(recall))
    time_score = max(0.0, float(time_score))

    # weighted geometric mean
    # geom = (ndcg^w1 * recall^w2 * time^w3)^(1/(w1+w2+w3))
    # handle 0^0 by adding tiny epsilon only if all are zero (still returns 0)
    product = 1.0
    for base, w in ((ndcg, w_ndcg), (recall, w_recall), (time_score, w_time)):
        if base == 0.0:
            if w > 0:
                product = 0.0
                break
            else:
                continue
        product *= (base ** w)

    val = product ** (1.0 / w_sum) if product > 0.0 else 0.0

    extras = {
        "geom_k_used_recall": k_rec,
        "geom_k_used_ndcg": k_ndc,
        "geom_time_score": time_score,
        "geom_time_source": time_meta.get("time_score_source"),
        "geom_time_seconds": time_meta.get("time_value_seconds"),
        "geom_time_transform": time_meta.get("time_score_transform"),
        "geom_weights_ndcg": w_ndcg,
        "geom_weights_recall": w_recall,
        "geom_weights_time": w_time,
    }
    return val, extras

def read_dataset_file(
    fp: Path,
    split_pref: List[str],
    metric_key: str,
    cfg: Dict
) -> Optional[Tuple[str, float, Dict]]:
    """
    Returns (task_name, metric_value, meta_extras) or None if not available.
    Also merges useful top-level timing fields into the split dict so time can be derived.
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

    # --- NEW: merge top-level timing fields into split_dict clone ---
    # Your files have: "evaluation_time": <seconds>  at top level.
    merged = dict(split_dict)
    top_level_time_keys = [
        "evaluation_time", "evaluation_time_s", "eval_seconds",
        "time_s", "latency_s", "elapsed_s", "time", "latency_ms", "eval_time_ms"
    ]
    for k in top_level_time_keys:
        if k in js and k not in merged:
            merged[k] = js[k]
    # ---------------------------------------------------------------

    metric_val, extras = compute_metric_value(merged, metric_key, cfg)
    if metric_val is None:
        return None

    return task_name, float(metric_val), extras


# ------------------ main ------------------

def main():
    parser = argparse.ArgumentParser(
        description="Screen embedding models by a chosen metric (supports direct metrics and weighted GEOM@k of ndcg/recall/time)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root_dir = Path(cfg["root_dir"]).expanduser().resolve()
    results_dir = (root_dir / cfg.get("results_subdir", "validation")).resolve()
    split_pref = cfg.get("split_preference", ["test", "validation", "train"])
    metric_key = cfg.get("metric_key", "ndcg_at_10")  # or "geom_mean"
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
        for fp in leaf.glob("*.json"):
            if fp.name == "model_meta.json":
                continue
            parsed = read_dataset_file(fp, split_pref, metric_key, cfg)
            if not parsed:
                continue
            task_name, metric_val, extras = parsed
            if not task_name:
                task_name = fp.stem
            if datasets_include and task_name not in datasets_include:
                continue

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
            if metric_key.lower() == "geom_mean":
                row.update({
                    "geom_k_used_recall": extras.get("geom_k_used_recall"),
                    "geom_k_used_ndcg": extras.get("geom_k_used_ndcg"),
                    "geom_time_score": extras.get("geom_time_score"),
                    "geom_time_source": extras.get("geom_time_source"),
                    "geom_time_seconds": extras.get("geom_time_seconds"),
                    "geom_time_transform": extras.get("geom_time_transform"),
                    "geom_weights_ndcg": extras.get("geom_weights_ndcg"),
                    "geom_weights_recall": extras.get("geom_weights_recall"),
                    "geom_weights_time": extras.get("geom_weights_time"),
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
        if metric_key.lower() == "geom_mean":
            cols += ["geom_k_used_recall", "geom_k_used_ndcg",
                     "geom_time_score", "geom_time_source", "geom_time_seconds", "geom_time_transform",
                     "geom_weights_ndcg", "geom_weights_recall", "geom_weights_time"]
        passed[cols].sort_values(["dataset", "rank_in_ds"]).to_csv(out_per_ds, index=False)
        print(f"\nWrote per-dataset passes to: {out_per_ds}")
    if out_union:
        pd.Series(passed_union, name="model_id").to_csv(out_union, index=False)
        print(f"Wrote union list to: {out_union}")

if __name__ == "__main__":
    main()

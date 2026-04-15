#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

CHUNKERS = {
    "fixed_token",
    "recursive_token",
    "semantic_fixed",
    "semantic_recursive",
    "hierarchical_section",
    "hybrid_multi",
}

# ---------- helpers for reading MTEB-style results ----------

def extract_test_items(data):
    """Return list of test dicts from scores.test (handles dict or list)."""
    scores = data.get("scores", {})
    test = scores.get("test")
    if isinstance(test, dict):
        return [test]
    if isinstance(test, list):
        return [t for t in test if isinstance(t, dict)]
    return []

def get_numeric_metrics_from_item(item):
    """Return {metric_name: value} for ALL numeric metrics in one test item."""
    out = {}
    for k, v in item.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out

def get_val(item, key):
    v = item.get(key)
    return float(v) if isinstance(v, (int, float)) else None

def extract_metric_for_k_item(item, base: str, k: int, warnings):
    """
    From a single test item, try <base>_at_<k>, else nearest among {1,3,5,10,20,50,100},
    else raw <base>. Returns (value, actual_k_used).
    """
    exact_key = f"{base}_at_{k}"
    v = get_val(item, exact_key)
    if v is not None:
        return v, k

    common_ks = [1, 3, 5, 10, 20, 50, 100]
    candidates = []
    for ck in common_ks:
        vv = get_val(item, f"{base}_at_{ck}")
        if vv is not None:
            candidates.append((ck, vv))

    if not candidates:
        vv = get_val(item, base)
        if vv is not None:
            warnings.append(f"Using '{base}' without @k; assuming k≈{k}")
            return vv, k
        return None, None

    ck, vv = min(candidates, key=lambda t: abs(t[0] - k))
    if ck != k:
        warnings.append(f"Missing {base}_at_{k}; using {base}_at_{ck} instead.")
    return vv, ck

# ---------- parse (model, chunker, c, o) from path ----------

def parse_model_info(rel_dir_str: str):
    """
    Parse info from folder names like:
      all_MiniLM_L6_v2_fixed_token_c192_o0
      all_MiniLM_L6_v2_recursive_token_c128_o32
    Returns dict with model_name, chunker, token_size (c), overlap (o).
    """
    parts = rel_dir_str.split("/")
    candidate = None
    for p in parts[::-1]:
        if any(ch in p for ch in CHUNKERS):
            candidate = p
            break
    if candidate is None:
        candidate = parts[-1]

    found_chunker = None
    for ch in sorted(CHUNKERS, key=len, reverse=True):
        if re.search(rf"(^|_){re.escape(ch)}(_|$)", candidate):
            found_chunker = ch
            break

    model_name = candidate
    token_size = None
    overlap = None

    if found_chunker:
        model_name = re.split(rf"_{re.escape(found_chunker)}(_|$)", candidate, maxsplit=1)[0]
        tail = candidate[len(model_name):]
        m_c = re.search(r"_c(\d+)", tail)
        m_o = re.search(r"_o(\d+)", tail)
        if m_c: token_size = int(m_c.group(1))
        if m_o: overlap = int(m_o.group(1))

    return {
        "model_name": model_name,
        "chunker": found_chunker,
        "token_size": token_size,
        "overlap": overlap,
    }

# ---------- time score + GEOM ----------

def derive_time_score(item: dict, parent: dict, args) -> float | None:
    """
    Derive a normalized time_score in (0,1], higher is better (faster).
    Priority:
      1) item[args.time_score_key] (already normalized)
      2) parent[args.time_score_key]
      3) item[args.time_seconds_key] (seconds)  -> normalize
      4) parent[args.time_seconds_key] (seconds) -> normalize
      5) item[args.time_ms_key] (ms)            -> normalize
      6) parent[args.time_ms_key] (ms)          -> normalize
    """
    # 1/2: normalized score field
    for src in (item, parent):
        if args.time_score_key and isinstance(src.get(args.time_score_key), (int, float)):
            return max(0.0, float(src[args.time_score_key]))

    # Helper: normalize seconds
    def normalize_seconds(t_sec: float) -> float:
        if args.time_transform == "inverse":
            ref = args.reference_seconds if args.reference_seconds > 0 else 1.0
            return 1.0 / (1.0 + (t_sec / ref))
        elif args.time_transform == "qps":
            if t_sec <= 0:
                return 0.0
            qps = 1.0 / t_sec
            ref_qps = args.reference_qps if args.reference_qps > 0 else 100.0
            return min(1.0, qps / ref_qps)
        elif args.time_transform == "identity":
            return float(t_sec)  # assume already normalized
        else:
            return 1.0 / (1.0 + t_sec)

    # 3/4: seconds
    for src in (item, parent):
        if args.time_seconds_key and isinstance(src.get(args.time_seconds_key), (int, float)):
            return normalize_seconds(float(src[args.time_seconds_key]))

    # 5/6: milliseconds
    for src in (item, parent):
        if args.time_ms_key and isinstance(src.get(args.time_ms_key), (int, float)):
            return normalize_seconds(float(src[args.time_ms_key]) / 1000.0)

    return None

def geom_at_k(ndcg: float | None, recall: float | None, time_score: float | None,
              w_ndcg: float, w_recall: float, w_time: float) -> float | None:
    """Weighted geometric mean of (ndcg, recall, time_score)."""
    if ndcg is None or recall is None or time_score is None:
        return None
    n = max(0.0, float(ndcg))
    r = max(0.0, float(recall))
    t = max(0.0, float(time_score))
    wsum = w_ndcg + w_recall + w_time
    if wsum <= 0:
        return None
    # Any zero with positive weight -> result 0
    if (n == 0.0 and w_ndcg > 0) or (r == 0.0 and w_recall > 0) or (t == 0.0 and w_time > 0):
        return 0.0
    # log-sum for stability
    ln = 0.0 if n == 0 else math.log(n)
    lr = 0.0 if r == 0 else math.log(r)
    lt = 0.0 if t == 0 else math.log(t)
    return math.exp((w_ndcg * ln + w_recall * lr + w_time * lt) / wsum)

def compute_item_metrics(item, parent_json, k, mrr_k, w_ndcg, w_recall, w_time, allow_time_missing_as_one, args, warnings=None):
    """Compute GEOM@k and record which k was used per metric for ONE item."""
    if warnings is None:
        warnings = []

    rec_k, rec_used_k   = extract_metric_for_k_item(item, "recall", k, warnings)
    ndcg_k, ndcg_used_k = extract_metric_for_k_item(item, "ndcg",  k, warnings)
    map_k, map_used_k   = extract_metric_for_k_item(item, "map",   k, warnings)

    if mrr_k is None:
        mrr_k = k
    mrr_k_val, mrr_used_k = extract_metric_for_k_item(item, "mrr", mrr_k, warnings)

    time_score = derive_time_score(item, parent_json, args)
    if time_score is None and allow_time_missing_as_one:
        time_score = 1.0
    geom = geom_at_k(ndcg_k, rec_k, time_score, w_ndcg, w_recall, w_time)

    return {
        "k_request": k,
        "k_recall_used": rec_used_k,
        "k_ndcg_used": ndcg_used_k,
        "k_map_used": map_used_k,
        "k_mrr_used": mrr_used_k,
        "recall_at_k": rec_k,
        "ndcg_at_k": ndcg_k,
        "map_at_k": map_k,
        "mrr_at_k": mrr_k_val,
        "time_score": time_score,
        "geom_at_k": geom,
    }

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare retrieval configs: compute GEOM@k and export ALL per-item metrics (no averaging) to CSV."
    )
    ap.add_argument("--root", type=Path, default=Path("tests"), help="Root folder to search")
    ap.add_argument("--pattern", default="ChemQuest.json", help="Filename to look for")
    ap.add_argument("--csv", type=Path, help="CSV output path (required for full dump)")

    # Evaluation parameters
    ap.add_argument("--k", type=int, default=10, help="Cutoff k for Recall@k and nDCG@k (default: 10)")
    ap.add_argument("--mrr-k", type=int, default=None, help="Optional k for MRR@k (default: use --k)")

    # GEOM weights
    ap.add_argument("--w-ndcg", type=float, default=1.0, help="Weight for nDCG component")
    ap.add_argument("--w-recall", type=float, default=1.0, help="Weight for Recall component")
    ap.add_argument("--w-time", type=float, default=1.0, help="Weight for time component")

    # Time normalization
    ap.add_argument("--time-score-key", type=str, default="time_score",
                    help="Normalized [0,1] score key (item or top-level). Used if present.")
    ap.add_argument("--time-seconds-key", type=str, default="evaluation_time",
                    help="Raw time (seconds) key (item or top-level).")
    ap.add_argument("--time-ms-key", type=str, default="latency_ms",
                    help="Raw time (milliseconds) key (item or top-level).")
    ap.add_argument("--time-transform", type=str, default="inverse", choices=["inverse", "qps", "identity"],
                    help="How to convert raw time to a [0,1] time_score.")
    ap.add_argument("--reference-seconds", type=float, default=1.0,
                    help="Baseline seconds for inverse transform.")
    ap.add_argument("--reference-qps", type=float, default=100.0,
                    help="Baseline QPS for qps transform.")
    ap.add_argument("--allow-time-missing-as-one", action="store_true",
                    help="If set, use time_score=1.0 when no time info is found.")

    args = ap.parse_args()

    files = sorted(args.root.rglob(args.pattern))
    if not files:
        print("No matching files found.", file=sys.stderr)
        return

    rows = []
    warnings = []
    # Union of all numeric metric keys across ALL items
    all_metric_keys = set()

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            warnings.append(f"Failed to read {f}: {e}")
            continue

        rel_dir = str(f.parent.relative_to(args.root))
        info = parse_model_info(rel_dir)

        items = extract_test_items(data)
        for idx, item in enumerate(items):
            per_item_metrics = compute_item_metrics(
                item, data, k=args.k, mrr_k=args.mrr_k,
                w_ndcg=args.w_ndcg, w_recall=args.w_recall, w_time=args.w_time,
                allow_time_missing_as_one=args.allow_time_missing_as_one,
                args=args, warnings=warnings
            )
            numeric_metrics = get_numeric_metrics_from_item(item)
            all_metric_keys.update(numeric_metrics.keys())

            # Optional string-ish fields
            hf_subset = item.get("hf_subset")
            languages = item.get("languages")
            if isinstance(languages, list):
                lang_str = ";".join(str(x) for x in languages)
            else:
                lang_str = languages if isinstance(languages, str) else None

            rows.append({
                "rel_path": rel_dir,
                "test_index": idx,  # which item in scores.test
                "hf_subset": hf_subset,
                "languages": lang_str,
                **info,
                **per_item_metrics,
                "_metrics": numeric_metrics,  # stash original numeric metrics for CSV expansion
            })

    # Sort by GEOM@k desc, then nDCG@k desc, then model/chunker
    def s(x):  # helper to sort desc with None at end
        return (-x if isinstance(x, (int, float)) else float('inf'))
    rows.sort(key=lambda r: (
        s(r.get("geom_at_k")),
        s(r.get("ndcg_at_k")),
        r.get("model_name",""),
        r.get("chunker",""),
        r.get("test_index", 0),
    ))

    # ---- CSV: full per-item dump ----
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        metric_cols = sorted(all_metric_keys)  # ALL numeric keys from items

        header = [
            "model_name","chunker","token_size","overlap",
            "rel_path","test_index","hf_subset","languages",
            "GEOM@k","time_score","Recall@k","nDCG@k","MRR@k","MAP@k",
            "k_req","k_recall_used","k_ndcg_used","k_mrr_used","k_map_used",
            "w_ndcg","w_recall","w_time",
        ] + metric_cols

        with args.csv.open("w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow(header)
            for r in rows:
                metrics = r["_metrics"]
                w.writerow([
                    r.get("model_name",""), r.get("chunker",""),
                    r.get("token_size",""), r.get("overlap",""),
                    r.get("rel_path",""), r.get("test_index",""),
                    r.get("hf_subset",""), r.get("languages",""),
                    fmt(r.get("geom_at_k")), fmt(r.get("time_score")),
                    fmt(r.get("recall_at_k")), fmt(r.get("ndcg_at_k")),
                    fmt(r.get("mrr_at_k")), fmt(r.get("map_at_k")),
                    r.get("k_request",""), r.get("k_recall_used",""),
                    r.get("k_ndcg_used",""), r.get("k_mrr_used",""), r.get("k_map_used",""),
                    args.w_ndcg, args.w_recall, args.w_time,
                    *[fmt(metrics.get(k)) for k in metric_cols],
                ])
        for wmsg in warnings:
            print("WARNING:", wmsg, file=sys.stderr)
        return

    # If no --csv given, show a concise per-item table to stdout
    print(f'{"Model":<28} {"Chunker":<20} {"c":>5} {"o":>5}  '
          f'{"GEOM@k":>8} {"time":>7}  {"R@k":>7} {"nDCG@k":>8} {"MRR@k":>8} {"MAP@k":>8}  '
          f'{"k":>3} {"kR":>3} {"kN":>3} {"kM":>3} {"kP":>3}  '
          f'weights(wN,wR,wT)   Rel Path [idx] / subset / langs')
    print("-" * 150)
    for r in rows:
        tag = f'{r["rel_path"]} [{r.get("test_index","")}]'
        if r.get("hf_subset"): tag += f' / {r["hf_subset"]}'
        if r.get("languages"): tag += f' / {r["languages"]}'
        print(f'{(r["model_name"] or ""):<28} {(r["chunker"] or ""):<20} '
              f'{"" if r["token_size"] is None else r["token_size"]:>5} '
              f'{"" if r["overlap"] is None else r["overlap"]:>5}  '
              f'{fmt(r["geom_at_k"]):>8} {fmt(r["time_score"]):>7}  '
              f'{fmt(r["recall_at_k"]):>7} {fmt(r["ndcg_at_k"]):>8} {fmt(r["mrr_at_k"]):>8} {fmt(r["map_at_k"]):>8}  '
              f'{r.get("k_request",""):>3} {str_or_blank(r.get("k_recall_used")):>3} {str_or_blank(r.get("k_ndcg_used")):>3} '
              f'{str_or_blank(r.get("k_mrr_used")):>3} {str_or_blank(r.get("k_map_used")):>3}  '
              f'({args.w_ndcg:.1f},{args.w_recall:.1f},{args.w_time:.1f})  '
              f'{tag}')

def fmt(x):
    if x is None:
        return "NaN"
    try:
        return f"{float(x):.6f}"
    except:
        return "NaN"

def str_or_blank(x):
    return "" if x is None else str(x)

if __name__ == "__main__":
    main()

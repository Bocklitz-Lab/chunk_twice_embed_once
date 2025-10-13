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

# ---------- composed metric (RRF only) ----------

def rrf_beta(recall, ndcg, beta=2.0):
    """
    Rank-Recall Fβ: harmonic mean of recall and nDCG with weight β on recall.
    Returns 0.0 if recall or nDCG is missing/zero.
    """
    if recall is None or ndcg is None:
        return None
    if recall <= 0.0 or ndcg <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * recall * ndcg / (b2 * ndcg + recall)

def compute_item_metrics(item, k, beta, mrr_k=None, warnings=None):
    """Compute RRF@k and record which k was used per metric for ONE item."""
    if warnings is None:
        warnings = []

    rec_k, rec_used_k = extract_metric_for_k_item(item, "recall", k, warnings)
    ndcg_k, ndcg_used_k = extract_metric_for_k_item(item, "ndcg", k, warnings)
    map_k, map_used_k   = extract_metric_for_k_item(item, "map", k, warnings)

    if mrr_k is None:
        mrr_k = k
    mrr_k_val, mrr_used_k = extract_metric_for_k_item(item, "mrr", mrr_k, warnings)

    rrf = rrf_beta(rec_k, ndcg_k, beta=beta)

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
        "rrf_beta_at_k": rrf,
    }

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare retrieval configs: compute RRF@k and export ALL per-item metrics (no averaging) to CSV."
    )
    ap.add_argument("--root", type=Path, default=Path("tests"), help="Root folder to search")
    ap.add_argument("--pattern", default="ChemQuest.json", help="Filename to look for")
    ap.add_argument("--csv", type=Path, help="CSV output path (required for full dump)")

    # Evaluation parameters
    ap.add_argument("--k", type=int, default=10, help="Cutoff k for Recall@k and nDCG@k (default: 10)")
    ap.add_argument("--beta", type=float, default=2.0, help="β for RRF@k (recall weight; default: 2.0)")
    ap.add_argument("--mrr-k", type=int, default=None, help="Optional k for MRR@k (default: use --k)")

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
                item, k=args.k, beta=args.beta, mrr_k=args.mrr_k, warnings=warnings
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

    # Sort by RRF@k desc, then nDCG@k desc, then model/chunker
    def s(x):  # helper to sort desc with None at end
        return (-x if isinstance(x, (int, float)) else float('inf'))
    rows.sort(key=lambda r: (
        s(r.get("rrf_beta_at_k")),
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
            "RRF@k","Recall@k","nDCG@k","MRR@k","MAP@k",
            "k_req","k_recall_used","k_ndcg_used","k_mrr_used","k_map_used",
            "beta",
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
                    fmt(r.get("rrf_beta_at_k")),
                    fmt(r.get("recall_at_k")), fmt(r.get("ndcg_at_k")),
                    fmt(r.get("mrr_at_k")), fmt(r.get("map_at_k")),
                    r.get("k_request",""), r.get("k_recall_used",""),
                    r.get("k_ndcg_used",""), r.get("k_mrr_used",""), r.get("k_map_used",""),
                    args.beta,
                    *[fmt(metrics.get(k)) for k in metric_cols],
                ])
        for wmsg in warnings:
            print("WARNING:", wmsg, file=sys.stderr)
        return

    # If no --csv given, show a concise per-item table to stdout
    print(f'{"Model":<28} {"Chunker":<20} {"c":>5} {"o":>5}  '
          f'{"RRF@k":>8}  {"R@k":>7} {"nDCG@k":>8} {"MRR@k":>8} {"MAP@k":>8}  '
          f'{"k":>3} {"kR":>3} {"kN":>3} {"kM":>3} {"kP":>3}  '
          f'{"β":>3}  Rel Path [idx] / subset / langs')
    print("-" * 140)
    for r in rows:
        tag = f'{r["rel_path"]} [{r.get("test_index","")}]'
        if r.get("hf_subset"): tag += f' / {r["hf_subset"]}'
        if r.get("languages"): tag += f' / {r["languages"]}'
        print(f'{(r["model_name"] or ""):<28} {(r["chunker"] or ""):<20} '
              f'{"" if r["token_size"] is None else r["token_size"]:>5} '
              f'{"" if r["overlap"] is None else r["overlap"]:>5}  '
              f'{fmt(r["rrf_beta_at_k"]):>8}  '
              f'{fmt(r["recall_at_k"]):>7} {fmt(r["ndcg_at_k"]):>8} {fmt(r["mrr_at_k"]):>8} {fmt(r["map_at_k"]):>8}  '
              f'{r.get("k_request",""):>3} {str_or_blank(r.get("k_recall_used")):>3} {str_or_blank(r.get("k_ndcg_used")):>3} '
              f'{str_or_blank(r.get("k_mrr_used")):>3} {str_or_blank(r.get("k_map_used")):>3}  '
              f'{int(r.get("beta", args.beta)):>3}  '
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

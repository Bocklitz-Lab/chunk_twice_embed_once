#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean

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

def extract_metric_mean(data, metric_key: str):
    """Mean of a metric across test items (or None)."""
    tests = extract_test_items(data)
    vals = []
    for d in tests:
        v = d.get(metric_key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return (mean(vals) if vals else None)

def extract_metric_for_k(data, base: str, k: int, warnings, prefer_exact=True):
    """
    Try to get mean of <base>_at_<k>. If unavailable, fall back to nearest
    among common ks. Returns (value, actual_k_used).
    """
    exact_key = f"{base}_at_{k}"
    v = extract_metric_mean(data, exact_key)
    if v is not None:
        return v, k

    # Try common ks and pick the nearest available
    common_ks = [1, 3, 5, 10, 20, 50, 100]
    # If exact not preferred, allow any key starting with base_
    candidates = []
    for ck in common_ks:
        vv = extract_metric_mean(data, f"{base}_at_{ck}")
        if vv is not None:
            candidates.append((ck, vv))

    if not candidates:
        # Final attempt: raw base without suffix (some reports use base name only)
        vv = extract_metric_mean(data, base)
        if vv is not None:
            warnings.append(f"Using '{base}' without @k; assuming k≈{k}")
            return vv, k
        return None, None

    # choose nearest k by absolute distance
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
    for ch in sorted(CHUNKERS, key=len, reverse=True):  # prefer longer names
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

# ---------- composed metrics ----------

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

def rrf_geomean(recall, ndcg, mrr=None, alpha=2.0, gamma=1.0, delta=0.5):
    """
    RRF-G: geometric mean of recall^alpha, ndcg^gamma, and (optional) mrr^delta.
    If mrr is None, the term is skipped (i.e., weights renormalized by alpha+gamma).
    Returns 0.0 if any required component is <= 0.
    """
    if recall is None or ndcg is None:
        return None
    # guard against non-positive values (geometric mean domain)
    if recall <= 0.0 or ndcg <= 0.0:
        return 0.0

    include_mrr = (mrr is not None)
    # If we include MRR but it's <=0, the product goes to 0.
    if include_mrr and mrr <= 0.0:
        return 0.0

    if include_mrr:
        power = 1.0 / (alpha + gamma + delta)
        prod = (recall ** alpha) * (ndcg ** gamma) * (mrr ** delta)
    else:
        power = 1.0 / (alpha + gamma)
        prod = (recall ** alpha) * (ndcg ** gamma)

    return prod ** power

def compute_metrics_from_json(data, k, beta, alpha, gamma, delta, mrr_k=None, warnings=None):
    if warnings is None:
        warnings = []

    # Pull Recall@k and nDCG@k
    rec_k, rec_used_k = extract_metric_for_k(data, "recall", k, warnings)
    ndcg_k, ndcg_used_k = extract_metric_for_k(data, "ndcg", k, warnings)

    # Optional: MAP@k, MRR@k for reporting; MRR used in RRF-G if present
    map_k, map_used_k = extract_metric_for_k(data, "map", k, warnings)
    # MRR can be sensitive to k; allow an override
    if mrr_k is None:
        mrr_k = k
    mrr_k_val, mrr_used_k = extract_metric_for_k(data, "mrr", mrr_k, warnings)

    # Compose scores
    rrf = rrf_beta(rec_k, ndcg_k, beta=beta)
    rrf_g = rrf_geomean(rec_k, ndcg_k, mrr=mrr_k_val, alpha=alpha, gamma=gamma, delta=delta)

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
        "rrf_g_at_k": rrf_g,
    }

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare retrieval configs using single-value composite metrics (RRF@k, RRF-G@k) and report underlying metrics."
    )
    ap.add_argument("--root", type=Path, default=Path("tests"), help="Root folder to search")
    ap.add_argument("--pattern", default="ChemQuest.json", help="Filename to look for")
    ap.add_argument("--csv", type=Path, help="Optional CSV output path")

    # Evaluation parameters
    ap.add_argument("--k", type=int, default=10, help="Cutoff k for Recall@k and nDCG@k (default: 10)")
    ap.add_argument("--beta", type=float, default=2.0, help="β for RRF@k (recall weight; default: 2.0)")
    ap.add_argument("--alpha", type=float, default=2.0, help="Exponent for Recall in RRF-G (default: 2.0)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Exponent for nDCG in RRF-G (default: 1.0)")
    ap.add_argument("--delta", type=float, default=0.5, help="Exponent for MRR in RRF-G (default: 0.5)")
    ap.add_argument("--mrr-k", type=int, default=None, help="Optional k for MRR@k (default: use --k)")

    args = ap.parse_args()

    files = sorted(args.root.rglob(args.pattern))
    rows, warnings = [], []

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            warnings.append(f"Failed to read {f}: {e}")
            continue

        rel_dir = str(f.parent.relative_to(args.root))
        info = parse_model_info(rel_dir)

        comps = compute_metrics_from_json(
            data,
            k=args.k,
            beta=args.beta,
            alpha=args.alpha,
            gamma=args.gamma,
            delta=args.delta,
            mrr_k=args.mrr_k,
            warnings=warnings
        )

        rows.append({
            "rel_path": rel_dir,
            **info,
            **comps,
        })

    # Sort by RRF@k desc, then RRF-G@k desc, then ndcg@k desc
    def sort_key(r):
        def nz(x):
            return -x if isinstance(x, (int, float)) and not math.isnan(x) else float('inf')
        return (nz(r.get("rrf_beta_at_k", float("nan"))),
                nz(r.get("rrf_g_at_k", float("nan"))),
                nz(r.get("ndcg_at_k", float("nan"))),
                r.get("model_name",""),
                r.get("chunker",""))

    rows.sort(key=sort_key)

    # CSV output
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow([
                "model_name","chunker","token_size","overlap",
                "RRF@k","RRF-G@k","Recall@k","nDCG@k","MRR@k","MAP@k",
                "k_req","k_recall_used","k_ndcg_used","k_mrr_used","k_map_used",
                "beta","alpha","gamma","delta",
                "rel_path"
            ])
            for r in rows:
                w.writerow([
                    r.get("model_name",""), r.get("chunker",""),
                    r.get("token_size",""), r.get("overlap",""),
                    fmt(r.get("rrf_beta_at_k")), fmt(r.get("rrf_g_at_k")),
                    fmt(r.get("recall_at_k")), fmt(r.get("ndcg_at_k")),
                    fmt(r.get("mrr_at_k")), fmt(r.get("map_at_k")),
                    r.get("k_request",""), r.get("k_recall_used",""),
                    r.get("k_ndcg_used",""), r.get("k_mrr_used",""), r.get("k_map_used",""),
                    args.beta, args.alpha, args.gamma, args.delta,
                    r.get("rel_path",""),
                ])
        for wmsg in warnings:
            print("WARNING:", wmsg, file=sys.stderr)
        return

    # Pretty table to stdout if --csv not provided
    if not rows:
        print("No valid results found.")
    else:
        hdr = (
            f'{"Model":<28} {"Chunker":<20} {"c":>5} {"o":>5}  '
            f'{"RRF@k":>8} {"RRF-G@k":>9}  {"R@k":>7} {"nDCG@k":>8} {"MRR@k":>8} {"MAP@k":>8}  '
            f'{"k":>3} {"kR":>3} {"kN":>3} {"kM":>3} {"kP":>3}  '
            f'{"β":>3} {"α":>3} {"γ":>3} {"δ":>3}  Rel Path'
        )
        print(hdr)
        print("-" * max(140, len(hdr)))
        for r in rows:
            print(f'{(r["model_name"] or ""):<28} {(r["chunker"] or ""):<20} '
                  f'{"" if r["token_size"] is None else r["token_size"]:>5} '
                  f'{"" if r["overlap"] is None else r["overlap"]:>5}  '
                  f'{fmt(r["rrf_beta_at_k"]):>8} {fmt(r["rrf_g_at_k"]):>9}  '
                  f'{fmt(r["recall_at_k"]):>7} {fmt(r["ndcg_at_k"]):>8} {fmt(r["mrr_at_k"]):>8} {fmt(r["map_at_k"]):>8}  '
                  f'{r.get("k_request",""):>3} {str_or_blank(r.get("k_recall_used")):>3} {str_or_blank(r.get("k_ndcg_used")):>3} '
                  f'{str_or_blank(r.get("k_mrr_used")):>3} {str_or_blank(r.get("k_map_used")):>3}  '
                  f'{args.beta:>3.0f} {args.alpha:>3.0f} {args.gamma:>3.0f} {args.delta:>3.1f}  '
                  f'{r["rel_path"]}')
    for wmsg in warnings:
        print("WARNING:", wmsg, file=sys.stderr)

def fmt(x):
    if x is None:
        return "NaN"
    try:
        return f"{float(x):.4f}"
    except:
        return "NaN"

def str_or_blank(x):
    return "" if x is None else str(x)

if __name__ == "__main__":
    main()

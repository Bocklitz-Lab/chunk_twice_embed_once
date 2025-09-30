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

# ---------- loss computation ----------

def precision_decay(p1, p10, eps=1e-8):
    if p1 is None or p10 is None:
        return None
    ratio = p10 / (p1 + eps)
    ratio = min(1.0, max(0.0, ratio))
    return 1.0 - ratio  # bigger if tail is worse

def robustness_penalty(nauc_ndcg10_max, nauc_ndcg10_std):
    """Return L_rob in [0,1]. If inputs missing -> None."""
    if nauc_ndcg10_max is None or nauc_ndcg10_std is None:
        return None
    r_rob = max(0.0, float(nauc_ndcg10_max) - 0.5 * abs(float(nauc_ndcg10_std)))
    r_rob_norm = min(1.0, r_rob / 0.2)  # 0.2 is a practical cap
    return 1.0 - r_rob_norm

def num_chunks_for_doc(T, L, O):
    """Approximate number of chunks for a document of T tokens."""
    if T <= 0: return 0
    if L is None or O is None: return None
    if T <= L: return 1
    step = max(1, L - O)
    return 1 + max(0, math.ceil((T - L) / step))

def cost_norm_vs_baseline(doc_token_counts, L, O, L0, O0):
    """Relative chunk count vs baseline. Returns float or None if inputs missing."""
    if not doc_token_counts or L is None or O is None or L0 is None or O0 is None:
        return None
    num = 0
    den = 0
    for T in doc_token_counts:
        n = num_chunks_for_doc(T, L, O)
        b = num_chunks_for_doc(T, L0, O0)
        if n is None or b is None: return None
        num += n
        den += max(1, b)
    if den == 0: return None
    return num / den

def compute_loss_from_json(data, cost_norm=None):
    # Required metrics (means across test items)
    ndcg10 = extract_metric_mean(data, "ndcg_at_10")
    map10  = extract_metric_mean(data, "map_at_10")
    mrr10  = extract_metric_mean(data, "mrr_at_10")
    rec20  = extract_metric_mean(data, "recall_at_20")
    p1     = extract_metric_mean(data, "precision_at_1")
    p10    = extract_metric_mean(data, "precision_at_10")
    nauc_m = extract_metric_mean(data, "nauc_ndcg_at_10_max")
    nauc_s = extract_metric_mean(data, "nauc_ndcg_at_10_std")

    # Convert to losses
    if None in (ndcg10, map10, mrr10, rec20, p1, p10):
        missing = [k for k,v in [("ndcg@10",ndcg10),("map@10",map10),("mrr@10",mrr10),
                                 ("recall@20",rec20),("precision@1",p1),("precision@10",p10)] if v is None]
        raise ValueError(f"Missing required metric(s): {', '.join(missing)}")

    L_ndcg = 1.0 - ndcg10
    L_map  = 1.0 - map10
    L_mrr  = 1.0 - mrr10
    L_rec  = 1.0 - rec20
    d_p    = precision_decay(p1, p10)
    L_rob  = robustness_penalty(nauc_m, nauc_s)
    if L_rob is None:
        L_rob = 0.0  # if NAUC missing, don’t penalize; you can change to a fixed penalty if preferred
    if cost_norm is None:
        L_cost = 0.0
    else:
        L_cost = max(0.0, float(cost_norm) - 1.0)

    loss = (0.45*L_ndcg + 0.20*L_map + 0.10*L_mrr +
            0.10*L_rec + 0.05*d_p + 0.05*L_rob + 0.05*L_cost)

    components = {
        "loss_total": loss,
        "loss_ndcg": L_ndcg,
        "loss_map": L_map,
        "loss_mrr": L_mrr,
        "loss_rec": L_rec,
        "loss_decay": d_p,
        "loss_rob": L_rob,
        "loss_cost": L_cost,
        # raw metrics shown for convenience
        "ndcg_at_10": ndcg10, "map_at_10": map10, "mrr_at_10": mrr10,
        "recall_at_20": rec20, "precision_at_1": p1, "precision_at_10": p10,
        "nauc_ndcg_at_10_max": nauc_m, "nauc_ndcg_at_10_std": nauc_s,
        "cost_norm": cost_norm if cost_norm is not None else 1.0,
    }
    return components

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Compare composite retrieval loss across result files and enrich with run metadata from folder names."
    )
    ap.add_argument("--root", type=Path, default=Path("tests"), help="Root folder to search")
    ap.add_argument("--pattern", default="ChemQuest.json", help="Filename to look for")
    ap.add_argument("--csv", type=Path, help="Optional CSV output path")
    # Optional cost term inputs:
    ap.add_argument("--doc-token-counts", type=Path,
                    help="Optional path to a CSV/TSV/TXT with one integer per line (token count per document).")
    ap.add_argument("--baseline-c", type=int, help="Baseline chunk size (c) for cost normalization")
    ap.add_argument("--baseline-o", type=int, help="Baseline overlap (o) for cost normalization")
    args = ap.parse_args()

    # Load optional doc token counts
    doc_token_counts = []
    if args.doc_token_counts:
        try:
            with args.doc_token_counts.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip().split(",")[0].split("\t")[0]
                    if not line: continue
                    try:
                        doc_token_counts.append(int(line))
                    except:
                        pass
        except Exception as e:
            print(f"WARNING: Failed to read --doc-token-counts: {e}", file=sys.stderr)
            doc_token_counts = []

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
        L = info.get("token_size")
        O = info.get("overlap")

        # compute cost_norm if possible
        cost_norm = None
        if doc_token_counts and args.baseline_c is not None and args.baseline_o is not None and L is not None and O is not None:
            try:
                cost_norm = cost_norm_vs_baseline(doc_token_counts, L, O, args.baseline_c, args.baseline_o)
            except Exception as e:
                warnings.append(f"Failed cost_norm for {rel_dir}: {e}")
                cost_norm = None

        try:
            comps = compute_loss_from_json(data, cost_norm=cost_norm)
        except Exception as e:
            warnings.append(f"{rel_dir}: {e}")
            continue

        rows.append({
            "rel_path": rel_dir,
            **info,
            **comps,
        })

    # Sort by loss asc, then model/chunker for stability
    rows.sort(key=lambda r: (r.get("loss_total", float("inf")), r["model_name"] or "", r["chunker"] or ""))

    # CSV output
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow([
                "model_name","chunker","token_size","overlap","loss_total",
                "ndcg_at_10","map_at_10","mrr_at_10","recall_at_20",
                "precision_at_1","precision_at_10",
                "loss_ndcg","loss_map","loss_mrr","loss_rec","loss_decay","loss_rob","loss_cost",
                "nauc_ndcg_at_10_max","nauc_ndcg_at_10_std","cost_norm","rel_path"
            ])
            for r in rows:
                w.writerow([
                    r.get("model_name",""), r.get("chunker",""),
                    r.get("token_size",""), r.get("overlap",""),
                    f'{r.get("loss_total", float("nan")):.6f}',
                    f'{r.get("ndcg_at_10", float("nan")):.6f}',
                    f'{r.get("map_at_10", float("nan")):.6f}',
                    f'{r.get("mrr_at_10", float("nan")):.6f}',
                    f'{r.get("recall_at_20", float("nan")):.6f}',
                    f'{r.get("precision_at_1", float("nan")):.6f}',
                    f'{r.get("precision_at_10", float("nan")):.6f}',
                    f'{r.get("loss_ndcg", float("nan")):.6f}',
                    f'{r.get("loss_map", float("nan")):.6f}',
                    f'{r.get("loss_mrr", float("nan")):.6f}',
                    f'{r.get("loss_rec", float("nan")):.6f}',
                    f'{r.get("loss_decay", float("nan")):.6f}',
                    f'{r.get("loss_rob", float("nan")):.6f}',
                    f'{r.get("loss_cost", float("nan")):.6f}',
                    f'{r.get("nauc_ndcg_at_10_max", float("nan")):.6f}',
                    f'{r.get("nauc_ndcg_at_10_std", float("nan")):.6f}',
                    f'{r.get("cost_norm", float("nan")):.6f}',
                    r.get("rel_path",""),
                ])
        for wmsg in warnings:
            print("WARNING:", wmsg, file=sys.stderr)
        return

    # Pretty table to stdout if --csv not provided
    if not rows:
        print("No valid results found.")
    else:
        header = (
            f'{"Model":<28} {"Chunker":<20} {"c":>5} {"o":>5}  '
            f'{"Loss":>8}  {"nDCG@10":>8} {"MAP@10":>8} {"MRR@10":>8} {"R@20":>8} '
            f'{"P@1":>8} {"P@10":>8}  {"Decay":>7} {"Rob":>7} {"CostN":>7}  Rel Path'
        )
        print(header)
        print("-" * max(120, len(header)))
        for r in rows:
            print(f'{(r["model_name"] or ""):<28} {(r["chunker"] or ""):<20} '
                  f'{"" if r["token_size"] is None else r["token_size"]:>5} '
                  f'{"" if r["overlap"] is None else r["overlap"]:>5}  '
                  f'{r["loss_total"]:>8.4f}  '
                  f'{r["ndcg_at_10"]:>8.4f} {r["map_at_10"]:>8.4f} {r["mrr_at_10"]:>8.4f} {r["recall_at_20"]:>8.4f} '
                  f'{r["precision_at_1"]:>8.4f} {r["precision_at_10"]:>8.4f}  '
                  f'{r["loss_decay"]:>7.4f} {r["loss_rob"]:>7.4f} {r["cost_norm"]:>7.3f}  '
                  f'{r["rel_path"]}')
    for wmsg in warnings:
        print("WARNING:", wmsg, file=sys.stderr)

if __name__ == "__main__":
    main()

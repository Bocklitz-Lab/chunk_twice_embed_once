#!/usr/bin/env python3
import argparse
import json
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

def extract_metric(data, metric_key: str):
    """Extract a metric from MTEB-style results where scores.test is dict or list.
    Returns a single float (mean if list), or None if not found."""
    scores = data.get("scores", {})
    test = scores.get("test")
    if isinstance(test, dict):
        val = test.get(metric_key)
        return float(val) if isinstance(val, (int, float)) else None
    if isinstance(test, list):
        vals = []
        for d in test:
            if isinstance(d, dict) and metric_key in d and isinstance(d[metric_key], (int, float)):
                vals.append(float(d[metric_key]))
        return mean(vals) if vals else None
    return None

def parse_model_info(rel_dir_str: str):
    """
    Parse info from folder names like:
      all_MiniLM_L6_v2_fixed_token_c192_o0
      all_MiniLM_L6_v2_recursive_token_c128_o32
      ...
    Returns dict with model_name, chunker, token_size, overlap.
    If multiple path segments, we pick the first segment that contains a known chunker.
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

def main():
    ap = argparse.ArgumentParser(
        description="Compare metric across ChemQuest.json files and enrich with run metadata from folder names."
    )
    ap.add_argument("--root", type=Path, default=Path("tests"), help="Root folder to search")
    ap.add_argument("--pattern", default="ChemQuest.json", help="Filename to look for")
    ap.add_argument("--csv", type=Path, help="Optional CSV output path")
    ap.add_argument("--metric", default="ndcg_at_10", help="Metric key to extract from scores.test (default: ndcg_at_10)")
    args = ap.parse_args()

    metric_key = args.metric
    files = sorted(args.root.rglob(args.pattern))
    rows, warnings = [], []

    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            warnings.append(f"Failed to read {f}: {e}")
            continue

        score = extract_metric(data, metric_key)
        if score is None:
            warnings.append(f"Missing scores.test.{metric_key} in {f}")
            continue

        rel_dir = str(f.parent.relative_to(args.root))
        info = parse_model_info(rel_dir)
        rows.append({
            "rel_path": rel_dir,
            metric_key: float(score),
            **info,
        })

    # Sort by chosen metric desc, then model/chunker for stability
    rows.sort(key=lambda r: (-r.get(metric_key, float("-inf")), r["model_name"] or "", r["chunker"] or ""))

    # CSV output
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as out:
            out.write(f"model_name,chunker,token_size,overlap,{metric_key},rel_path\n")
            for r in rows:
                out.write(f'{r["model_name"] or ""},{r["chunker"] or ""},'
                          f'{"" if r["token_size"] is None else r["token_size"]},'
                          f'{"" if r["overlap"] is None else r["overlap"]},'
                          f'{r[metric_key]:.6f},{r["rel_path"]}\n')
        for w in warnings:
            print("WARNING:", w, file=sys.stderr)
        return

    # Pretty table to stdout if --csv not provided
    if not rows:
        print("No valid results found.")
    else:
        metric_label = metric_key
        # compute width for label cell
        header = f'{"Model":<28} {"Chunker":<20} {"c":>5} {"o":>5}  {metric_label:>10}  Rel Path'
        print(header)
        print("-" * max(90, len(header)))
        for r in rows:
            c = "" if r["token_size"] is None else r["token_size"]
            o = "" if r["overlap"] is None else r["overlap"]
            print(f'{(r["model_name"] or ""):<28} {(r["chunker"] or ""):<20} {c:>5} {o:>5}  {r[metric_key]:>10.6f}  {r["rel_path"]}')
    for w in warnings:
        print("WARNING:", w, file=sys.stderr)

if __name__ == "__main__":
    main()

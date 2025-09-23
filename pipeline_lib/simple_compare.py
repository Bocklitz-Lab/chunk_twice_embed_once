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

def extract_ndcg10(data):
    """Handle MTEB formats where scores.test is dict or list."""
    scores = data.get("scores", {})
    test = scores.get("test")
    if isinstance(test, dict):
        return test.get("ndcg_at_10")
    if isinstance(test, list):
        vals = [d.get("ndcg_at_10") for d in test if isinstance(d, dict) and "ndcg_at_10" in d]
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
    # pick the segment most likely to be the run folder
    parts = rel_dir_str.split("/")
    candidate = None
    for p in parts[::-1]:
        if any(ch in p for ch in CHUNKERS):
            candidate = p
            break
    if candidate is None:
        candidate = parts[-1]

    # find the chunker token
    found_chunker = None
    for ch in sorted(CHUNKERS, key=len, reverse=True):  # prefer longer names
        # enforce boundary with underscores or string edges
        if re.search(rf"(^|_){re.escape(ch)}(_|$)", candidate):
            found_chunker = ch
            break

    model_name = candidate
    token_size = None
    overlap = None

    if found_chunker:
        # model name is everything before the chunker (strip trailing underscore)
        model_name = re.split(rf"_{re.escape(found_chunker)}(_|$)", candidate, maxsplit=1)[0]
        # look for _cNNN and _oNNN anywhere after the chunker
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
    ap = argparse.ArgumentParser(description="Compare ndcg@10 across ChemQuest.json files and enrich with run metadata from folder names.")
    ap.add_argument("--root", type=Path, default=Path("tests"), help="Root folder to search")
    ap.add_argument("--pattern", default="ChemQuest.json", help="Filename to look for")
    ap.add_argument("--csv", type=Path, help="Optional CSV output path")
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

        score = extract_ndcg10(data)
        if score is None:
            warnings.append(f"Missing scores.test.ndcg_at_10 in {f}")
            continue

        rel_dir = str(f.parent.relative_to(args.root))
        info = parse_model_info(rel_dir)
        rows.append({
            "rel_path": rel_dir,
            "ndcg_at_10": float(score),
            **info,
        })

    # Sort by score desc, then model/chunker for stability
    rows.sort(key=lambda r: (-r["ndcg_at_10"], r["model_name"] or "", r["chunker"] or ""))

    # CSV output (recommended)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as out:
            out.write("model_name,chunker,token_size,overlap,ndcg_at_10,rel_path\n")
            for r in rows:
                out.write(f'{r["model_name"] or ""},{r["chunker"] or ""},'
                          f'{"" if r["token_size"] is None else r["token_size"]},'
                          f'{"" if r["overlap"] is None else r["overlap"]},'
                          f'{r["ndcg_at_10"]:.6f},{r["rel_path"]}\n')
        for w in warnings:
            print("WARNING:", w, file=sys.stderr)
        return

    # Pretty table to stdout if --csv not provided
    if not rows:
        print("No valid results found.")
    else:
        print(f'{"Model":<28} {"Chunker":<20} {"c":>5} {"o":>5}  {"ndcg@10":>10}  Rel Path')
        print("-"*90)
        for r in rows:
            c = "" if r["token_size"] is None else r["token_size"]
            o = "" if r["overlap"] is None else r["overlap"]
            print(f'{(r["model_name"] or ""):<28} {(r["chunker"] or ""):<20} {c:>5} {o:>5}  {r["ndcg_at_10"]:>10.6f}  {r["rel_path"]}')
    for w in warnings:
        print("WARNING:", w, file=sys.stderr)

if __name__ == "__main__":
    main()

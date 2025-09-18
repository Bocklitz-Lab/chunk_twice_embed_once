#!/usr/bin/env python3
import os, json
from collections import defaultdict
from pipeline_lib.utils import (
    load_config, cfg_get, ensure_dir, load_jsonl, save_jsonl, normalize_text,
    jaccard, rougeL_like, cosine, overlap_ratio
)

def main():
    cfg = load_config("config.yaml")
    out_dir = cfg_get(cfg, "stage4_match.output_dir", required=True)
    ensure_dir(out_dir)

    # Inputs
    corpus_dir = cfg_get(cfg, "stage4_match.input_corpus_dir", required=True)
    qa_dir = cfg_get(cfg, "stage4_match.input_qa_dir", required=True)
    emb_dir = cfg_get(cfg, "stage4_match.input_embed_dir", required=True)

    with open(os.path.join(corpus_dir, "doc_chunks.json"), "r", encoding="utf-8") as f:
        doc_chunks = json.load(f)
    corpus = load_jsonl(os.path.join(corpus_dir, "corpus.jsonl"))
    qa_pairs = json.load(open(os.path.join(qa_dir, "qa_pairs.json"), "r", encoding="utf-8"))
    chunk_norm = {row["id"]: row["norm_text"] for row in load_jsonl(os.path.join(emb_dir, "chunk_norm.jsonl"))}
    chunk_emb = {row["id"]: row["embedding"] for row in load_jsonl(os.path.join(emb_dir, "chunk_embeddings.jsonl"))}

    # Config
    window_k = int(cfg_get(cfg, "stage4_match.matching.window_k", 2))
    thr = {
        "cos_strong": float(cfg_get(cfg, "stage4_match.thresholds.cos_strong", 0.72)),
        "cos_med": float(cfg_get(cfg, "stage4_match.thresholds.cos_med", 0.60)),
        "jaccard_strong": float(cfg_get(cfg, "stage4_match.thresholds.jaccard_strong", 0.40)),
        "jaccard_med": float(cfg_get(cfg, "stage4_match.thresholds.jaccard_med", 0.25)),
        "overlap_strong": float(cfg_get(cfg, "stage4_match.thresholds.overlap_strong", 0.90)),
        "overlap_med": float(cfg_get(cfg, "stage4_match.thresholds.overlap_med", 0.30)),
        "overlap_low": float(cfg_get(cfg, "stage4_match.thresholds.overlap_low", 0.10)),
    }

    def grade(overlap: float, jacc: float, rougeL: float, cos_sim: float, in_window: bool) -> int:
        if overlap >= thr["overlap_strong"] or (cos_sim >= thr["cos_strong"] and rougeL >= 0.45) or (cos_sim >= 0.68 and jacc >= thr["jaccard_strong"]):
            return 3
        if (thr["overlap_med"] <= overlap < thr["overlap_strong"]) or (cos_sim >= thr["cos_med"]) or (rougeL >= 0.45) or (jacc >= thr["jaccard_strong"]):
            return 2
        if (thr["overlap_low"] <= overlap < thr["overlap_med"]) or ((0.30 <= rougeL < 0.45) or (thr["jaccard_med"] <= jacc < thr["jaccard_strong"]) or (0.50 <= cos_sim < thr["cos_med"] and in_window)):
            return 1
        return 0

    # Helper for candidate generation
    def candidate_chunk_ids(doc_name: str, rs: int, re: int):
        out, base = [], set()
        chunks = doc_chunks[doc_name]
        doc_id = os.path.splitext(doc_name)[0]
        for idx, ch in enumerate(chunks):
            cs, ce = ch["start_index"], ch["end_index"]
            if not (re <= cs or rs >= ce):
                out.append(f"{doc_id}_{idx}"); base.add(idx)
        expanded = set(out)
        for idx in list(base):
            lo = max(0, idx - window_k); hi = min(len(chunks)-1, idx + window_k)
            for j in range(lo, hi+1):
                expanded.add(f"{doc_id}_{j}")
        return list(expanded), base

    # Build queries + qrels
    queries, qrels = [], []
    for qid, (question, references, doc_name) in enumerate(qa_pairs):
        queries.append({"id": str(qid), "title": f"Query {qid}", "text": question})
        rel = defaultdict(int)
        ref_norms = [normalize_text(ref.get("content", "")) for ref in references]

        for r_idx, ref in enumerate(references):
            rs, re = ref["start_index"], ref["end_index"]
            ref_norm = ref_norms[r_idx]; ref_toks = ref_norm.split()
            cand_ids, base_indices = candidate_chunk_ids(doc_name, rs, re)

            for cid in cand_ids:
                idx = int(cid.split("_")[-1])
                ch = doc_chunks[doc_name][idx]
                cs, ce = ch["start_index"], ch["end_index"]
                ov = overlap_ratio(rs, re, cs, ce)
                cn = chunk_norm.get(cid, "")
                cn_toks = cn.split()
                j = jaccard(ref_toks, cn_toks)
                rl = rougeL_like(ref_toks, cn_toks)
                cos_sim = cosine(chunk_emb.get(cid, []), None)  # placeholder to keep signature
                # We want cosine(ref_emb, chunk_emb). But we only precomputed chunk_emb.
                # Use proxy: cosine between normalized ref text embedding and chunk embedding by embedding refs on-the-fly:
                # To avoid API calls here, we approximate: use lexical + overlap only if no ref embedding is available.
                cos_sim = 0.0  # keep 0; set up optional ref embedding stage later if needed.

                in_window = any(abs(idx - b) <= window_k for b in base_indices) or bool(base_indices)
                g = grade(ov, j, rl, cos_sim, in_window)
                if g > 0: rel[cid] = max(rel[cid], g)

        for cid, g in rel.items():
            qrels.append({"query_id": str(qid), "doc_id": cid, "relevance": g})
        if not rel:
            print(f"⚠️ No relevant chunk found for query {qid} in {doc_name}")

    save_jsonl(queries, os.path.join(out_dir, "queries.jsonl"))
    save_jsonl(qrels, os.path.join(out_dir, "qrels.jsonl"))

    # also copy corpus.jsonl path for completeness (already created in stage1)
    print("Stage4 done.")

if __name__ == "__main__":
    main()

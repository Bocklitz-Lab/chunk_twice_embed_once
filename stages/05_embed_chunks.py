#!/usr/bin/env python3
import os, json
from pipeline_lib.utils import load_config, cfg_get, ensure_dir, save_jsonl, normalize_text
from pipeline_lib.embedders import build_embedder

def main():
    cfg = load_config("config.yaml")
    in_dir = cfg_get(cfg, "stage3_embeddings.input_dir", required=True)
    out_dir = cfg_get(cfg, "stage3_embeddings.output_dir", required=True)
    ensure_dir(out_dir)

    doc_chunks_path = os.path.join(in_dir, "doc_chunks.json")
    with open(doc_chunks_path, "r", encoding="utf-8") as f:
        doc_chunks = json.load(f)

    # Collect all chunk texts
    ids, texts = [], []
    for doc_name, chunks in doc_chunks.items():
        doc_id = os.path.splitext(doc_name)[0]
        for idx, ch in enumerate(chunks):
            cid = f"{doc_id}_{idx}"
            ids.append(cid)
            texts.append(normalize_text(ch["chunk_text"]))

    # Save normalized texts
    norm_rows = [{"id": cid, "norm_text": t} for cid, t in zip(ids, texts)]
    save_jsonl(norm_rows, os.path.join(out_dir, "chunk_norm.jsonl"))

    # Compute embeddings
    embedder = build_embedder(cfg)
    print("Embedding chunks ...")
    vectors = embedder.embed(texts)
    emb_rows = [{"id": cid, "embedding": vec} for cid, vec in zip(ids, vectors)]
    save_jsonl(emb_rows, os.path.join(out_dir, "chunk_embeddings.jsonl"))

    print("Stage3 done.")

if __name__ == "__main__":
    main()

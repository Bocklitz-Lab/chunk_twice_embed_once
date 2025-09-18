#!/usr/bin/env python3
import os, json, glob
from pipeline_lib.utils import load_config, cfg_get, ensure_dir, save_json, save_jsonl
from pipeline_lib.embedders import build_embedder, make_embedding_function_from_embedder
from pipeline_lib.chunkers import ChunkerFactory

def from_chunked_jsons(input_dir: str):
    corpus, doc_chunks, id_map = [], {}, {}
    for fn in os.listdir(input_dir):
        if not fn.endswith(".json"): continue
        path = os.path.join(input_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if chunks and "document_name" in chunks[0]:
            doc_name = chunks[0]["document_name"]
            doc_id = os.path.splitext(doc_name)[0]
            id_map[doc_id] = doc_name
            doc_chunks[doc_name] = chunks
            for idx, ch in enumerate(chunks):
                corpus.append({"id": f"{doc_id}_{idx}", "title": f"{doc_id} - Chunk {idx}", "text": ch["chunk_text"]})
    return corpus, doc_chunks, id_map

def from_raw_docs(cfg, raw_dir: str):
    embedder = build_embedder(cfg)  # used by semantic chunkers if needed
    chunker = ChunkerFactory(cfg, make_embedding_function_from_embedder(embedder)).build()
    from pipeline_lib.chunkers import ChunkerFactory as CF
    corpus, doc_chunks, id_map = [], {}, {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.txt"))):
        doc_name = os.path.basename(path); doc_id = os.path.splitext(doc_name)[0]
        id_map[doc_id] = doc_name
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks_strs = chunker.split_text(text)
        chunk_records = CF.materialize_with_offsets(text, doc_name, chunks_strs)
        dicts = [{"document_name": c.document_name, "start_index": c.start_index, "end_index": c.end_index, "chunk_text": c.chunk_text} for c in chunk_records]
        doc_chunks[doc_name] = dicts
        for idx, ch in enumerate(dicts):
            corpus.append({"id": f"{doc_id}_{idx}", "title": f"{doc_id} - Chunk {idx}", "text": ch["chunk_text"]})
    return corpus, doc_chunks, id_map

def main():
    cfg = load_config("config.yaml")
    mode = cfg_get(cfg, "stage1_corpus.mode", required=True)
    out_dir = cfg_get(cfg, "stage1_corpus.output_dir", required=True)
    ensure_dir(out_dir)

    if mode == "chunked_jsons":
        input_dir = cfg_get(cfg, "stage1_corpus.input_dir", required=True)
        corpus, doc_chunks, id_map = from_chunked_jsons(input_dir)
    elif mode == "raw_docs":
        raw_dir = cfg_get(cfg, "stage1_corpus.raw_docs_dir", required=True)
        corpus, doc_chunks, id_map = from_raw_docs(cfg, raw_dir)
    else:
        raise ValueError("stage1_corpus.mode must be 'chunked_jsons' or 'raw_docs'")

    save_jsonl(corpus, os.path.join(out_dir, "corpus.jsonl"))
    save_json(doc_chunks, os.path.join(out_dir, "doc_chunks.json"))
    save_json(id_map, os.path.join(out_dir, "corpus_id_mapping.json"))
    print("Stage1 done.")

if __name__ == "__main__":
    main()

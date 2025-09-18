import sys
import os
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from mteb.evaluation.evaluators import RetrievalEvaluator  # ✅ Correct import

# Ensure the correct MTEB path is added
sys.path.insert(0, os.path.abspath("./mteb"))

import mteb
print("Using local MTEB version from:", mteb.__file__)

# 🔹 **Step 1: Check GPU availability**
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# 🔹 **Step 2: Load dataset from Hugging Face**
dataset_name = "mahmoudamiri/FSUChemRxivQest"

# ✅ Extract the correct splits
corpus = load_dataset(dataset_name, "corpus")["corpus"]
queries = load_dataset(dataset_name, "queries")["queries"]
qrels_raw = load_dataset(dataset_name, "qrels")["qrels"]

print("✅ Corpus loaded:", len(corpus))
print("✅ Queries loaded:", len(queries))
print("✅ Qrels loaded:", len(qrels_raw))

# 🔹 **Step 3: Convert Qrels to MTEB format**
qrels = {}
for entry in qrels_raw:
    qid = entry["query_id"]
    doc_id = entry["doc_id"]
    relevance = entry["relevance"]
    if qid not in qrels:
        qrels[qid] = {}
    qrels[qid][doc_id] = relevance  # Convert list format to dict[qid][doc_id] = relevance

print("📌 Sample Qrels Entry:", list(qrels.items())[:3])

# 🔹 **Step 4: Load Sentence Transformer Model (Move to GPU)**
model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name).to(device)  # ✅ Move model to GPU

# 🔹 **Step 5: Convert Corpus & Queries to Dict Format**
corpus_dict = {doc["id"]: {"title": doc["title"], "text": doc["text"]} for doc in corpus}
queries_dict = {query["id"]: query["text"] for query in queries}

# 🔹 **Step 6: Define Retrieval Evaluator**
evaluator = RetrievalEvaluator(
    dataset_name=dataset_name,
    corpus=corpus_dict,
    queries=queries_dict,
    qrels=qrels,
    retriever=model,
    batch_size=32,
    score_function="cos_sim",
)

# 🔹 **Step 7: Batch Encode Corpus & Queries on GPU**
print("🚀 Encoding corpus in batches (GPU-accelerated)...")

# ✅ Encode entire corpus in batches
corpus_ids = list(corpus_dict.keys())
corpus_texts = [corpus_dict[doc_id]["text"] for doc_id in corpus_ids]

corpus_embeddings = model.encode(
    corpus_texts, batch_size=32, convert_to_tensor=True, device=device, show_progress_bar=True
)

# Store embeddings in a dictionary (ID → embedding)
doc_embeddings = {doc_id: emb for doc_id, emb in zip(corpus_ids, corpus_embeddings)}

print("✅ Corpus encoding complete.")

# ✅ Encode queries in batches
print("🚀 Encoding queries in batches (GPU-accelerated)...")

query_ids = list(queries_dict.keys())
query_texts = [queries_dict[qid] for qid in query_ids]

query_embeddings = model.encode(
    query_texts, batch_size=32, convert_to_tensor=True, device=device, show_progress_bar=True
)

# Store query embeddings in a dictionary (ID → embedding)
query_embeddings_dict = {qid: emb for qid, emb in zip(query_ids, query_embeddings)}

print("✅ Query encoding complete.")

# 🔹 **Step 8: Compute Similarity & Retrieve Documents Efficiently**
print("🔎 Computing similarity scores and ranking documents...")

retrieved_results = {}  # Stores retrieved document rankings

# ✅ Compute similarity scores in a vectorized way
for query_id, query_embedding in query_embeddings_dict.items():
    # Compute cosine similarity with all document embeddings
    scores = torch.cosine_similarity(query_embedding.unsqueeze(0), corpus_embeddings, dim=-1)
    
    # Rank documents by similarity score
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]

    # Store ranked results
    retrieved_results[query_id] = {doc_id: rank + 1 for rank, doc_id in enumerate(ranked_doc_ids)}

print("✅ Retrieval complete.")

# 🔹 **Step 9: Run Evaluation**
k_values = [1, 3, 5, 10]  # Common retrieval metrics

results = evaluator.evaluate(
    qrels=qrels,  # Ground truth relevance
    results=retrieved_results,  # Model retrieval results
    k_values=k_values  # Metrics at k
)

print("📊 Evaluation Results:", results)

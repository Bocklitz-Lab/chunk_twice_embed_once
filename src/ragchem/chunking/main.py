import json
import os
from chunking_evaluation.chunking import (
    FixedTokenChunker,
    ClusterSemanticChunker,
    LLMSemanticChunker,
    KamradtModifiedChunker,
    RecursiveTokenChunker
)
from chunking_evaluation.evaluation_framework.general_evaluation import GeneralEvaluation
from chunking_evaluation.utils import openai_token_count
# Initialize the evaluator with built-in dataset
evaluator = GeneralEvaluation(chroma_db_path="mydb/")



chunking_methods = {
    # "fixed": FixedTokenChunker(chunk_size=64, chunk_overlap=12),
    "cluster_semantic": ClusterSemanticChunker(),
    "llm_semantic": LLMSemanticChunker(),
    "kamradt": KamradtModifiedChunker(avg_chunk_size=50),
    # "recursive_token_512_128": RecursiveTokenChunker(chunk_size=512,  chunk_overlap=128, length_function=openai_token_count),
    # "recursive_token_256_64": RecursiveTokenChunker(chunk_size=256,  chunk_overlap=64, length_function=openai_token_count),
    # "recursive_token_128_32": RecursiveTokenChunker(chunk_size=128,  chunk_overlap=32, length_function=openai_token_count),
    # "recursive_token_64_16": RecursiveTokenChunker(chunk_size=64,  chunk_overlap=16, length_function=openai_token_count),
    # "recursive_token_100_20": RecursiveTokenChunker(chunk_size=100,  chunk_overlap=20, length_function=openai_token_count),
    "recursive_token_100_80": RecursiveTokenChunker(chunk_size=100,  chunk_overlap=80, length_function=openai_token_count),
    "fixed_100_80": FixedTokenChunker(chunk_size=100, chunk_overlap=80),
    "recursive_token_100_60": RecursiveTokenChunker(chunk_size=100,  chunk_overlap=60, length_function=openai_token_count),
    "fixed_100_60": FixedTokenChunker(chunk_size=100, chunk_overlap=60),
    "recursive_token_100_40": RecursiveTokenChunker(chunk_size=100,  chunk_overlap=40, length_function=openai_token_count),
    "fixed_100_40": FixedTokenChunker(chunk_size=100, chunk_overlap=40),
    "recursive_token_100_20": RecursiveTokenChunker(chunk_size=100,  chunk_overlap=20, length_function=openai_token_count),
    "fixed_100_20": FixedTokenChunker(chunk_size=100, chunk_overlap=20),
    "recursive_token_100_0": RecursiveTokenChunker(chunk_size=100,  chunk_overlap=0, length_function=openai_token_count),
    "fixed_100_0": FixedTokenChunker(chunk_size=100, chunk_overlap=0),
}

# Define output directory
output_dir = "results_overlap"
os.makedirs(output_dir, exist_ok=True)

# Store results
results = {}

for method_name, chunker in chunking_methods.items():
    print(f"Running evaluation for {method_name} chunking...")

    # Run evaluation
    evaluation_metrics = evaluator.run(chunker, retrieve=5)

    # Store results
    results[method_name] = {
        "iou_mean": float(evaluation_metrics["iou_mean"]),
        "iou_std": float(evaluation_metrics["iou_std"]),
        "recall_mean": float(evaluation_metrics["recall_mean"]),
        "recall_std": float(evaluation_metrics["recall_std"]),
        "precision_omega_mean": float(evaluation_metrics["precision_omega_mean"]),
        "precision_omega_std": float(evaluation_metrics["precision_omega_std"]),
        "precision_mean": float(evaluation_metrics["precision_mean"]),
        "precision_std": float(evaluation_metrics["precision_std"]),
        "corpora_scores": evaluation_metrics["corpora_scores"]  # Storing detailed corpus results
    }

    # Save results in a readable JSON format
    output_path = os.path.join(output_dir, f"{method_name}.json")  # Use f-string formatting
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)



print(f"Results saved to {output_path}")

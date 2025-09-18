from __future__ import annotations
import logging
from datasets import load_dataset
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)

class FSUChemRxivQest(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FSUChemRxivQest",
        dataset={
            "path": "mahmoudamiri/FSUChemRxivQest",
            "revision": "main",
        },
        description="A retrieval dataset for FSUChemRxivQest.",
        reference="https://huggingface.co/datasets/mahmoudamiri/FSUChemRxivQest",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" 
        @dataset{FSUChemRxivQest,
        title={FSUChemRxivQest Dataset},
        author={Mahmoud Amiri},
        year={2024},
        publisher={Hugging Face},
        url={https://huggingface.co/datasets/mahmoudamiri/FSUChemRxivQest}
        } 
        """,
    )

    # def load_data(self, **kwargs):
    #     """Load dataset and structure it correctly for MTEB Retrieval."""
    #     try:
    #         # Load dataset splits
    #         dataset_corpus = load_dataset(self.metadata.dataset["path"], name="corpus", split="corpus")
    #         dataset_queries = load_dataset(self.metadata.dataset["path"], name="queries", split="queries")
    #         dataset_qrels = load_dataset(self.metadata.dataset["path"], name="qrels", split="qrels")

    #         # Detect correct column names
    #         corpus_id_col = "_id" if "_id" in dataset_corpus.column_names else "id"
    #         query_id_col = "_id" if "_id" in dataset_queries.column_names else "id"
    #         qrels_query_col = "query-id" if "query-id" in dataset_qrels.column_names else "query_id"
    #         qrels_doc_col = "corpus-id" if "corpus-id" in dataset_qrels.column_names else "doc_id"

    #         # Transform corpus into expected format
    #         self.corpus = {str(doc[corpus_id_col]): {"title": doc.get("title", ""), "text": doc["text"]}
    #                        for doc in dataset_corpus}

    #         # Transform queries into expected format
    #         self.queries = {str(query[query_id_col]): query["text"]
    #                         for query in dataset_queries}

    #         # Transform qrels into expected format
    #         self.qrels = {}
    #         for qrel in dataset_qrels:
    #             query_id = str(qrel[qrels_query_col])
    #             doc_id = str(qrel[qrels_doc_col])
    #             if query_id not in self.qrels:
    #                 self.qrels[query_id] = {}
    #             self.qrels[query_id][doc_id] = int(qrel.get("score", 1))  # Default to 1 if missing

    #         # ✅ Also store inside "test"
    #         self.dataset = {
    #             "test": {
    #                 "corpus": self.corpus,
    #                 "queries": self.queries,
    #                 "qrels": self.qrels
    #             }
    #         }

    #         logging.info(f"✅ Successfully loaded dataset: {self.metadata.name}")

    #     except Exception as e:
    #         logging.error(f"❌ Error loading dataset {self.metadata.name}: {e}")
    #         self.corpus, self.queries, self.qrels = None, None, None

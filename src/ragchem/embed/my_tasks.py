# my_tasks.py
from __future__ import annotations
import logging
from typing import Dict, Any
from datasets import load_dataset
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChemQuest(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChemQuest",
        dataset={
            "path": "Bocklitz-Lab/ChemQuest",
            "revision": "main",   # tip: pin to a commit hash for reproducibility
        },
        description="A retrieval dataset for ChemQuest.",
        reference="https://huggingface.co/datasets/Bocklitz-Lab/ChemQuest",
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
        bibtex_citation=r"""
@dataset{ChemQuest,
  title={ChemQuest Dataset},
  author={Mahmoud Amiri},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Bocklitz-Lab/ChemQuest}
}
""",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k_values = [1, 3, 5, 10, 100]

    def _to_corpus(self, rows) -> Dict[str, Dict[str, Any]]:
        out = {}
        for r in rows:
            doc_id = r.get("_id", r.get("id", r.get("doc_id")))
            if doc_id is None:
                raise ValueError("Corpus row missing an id-like column (_id/id/doc_id).")
            doc_id = str(doc_id)
            title = r.get("title")
            text = r.get("text") or r.get("contents") or r.get("passage")
            if text is None:
                raise ValueError("Corpus row missing a text-like column (text/contents/passage).")
            out[doc_id] = {"title": title, "text": text}
        return out

    def _to_queries(self, rows) -> Dict[str, str]:
        out = {}
        for r in rows:
            qid = r.get("_id", r.get("id", r.get("query_id")))
            if qid is None:
                raise ValueError("Query row missing an id-like column (_id/id/query_id).")
            qid = str(qid)
            text = r.get("text") or r.get("query")
            if text is None:
                raise ValueError("Query row missing text/query column.")
            out[qid] = text
        return out

    def _to_qrels(self, rows) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in rows:
            qid = r.get("query-id", r.get("query_id", r.get("qid", r.get("_query_id"))))
            did = r.get("corpus-id", r.get("doc_id", r.get("pid", r.get("_doc_id"))))
            if qid is None or did is None:
                raise ValueError("Qrels row missing query-id/query_id and/or corpus-id/doc_id.")
            qid = str(qid)
            did = str(did)
            rel = int(r.get("score", r.get("relevance", 1)))
            out.setdefault(qid, {})[did] = rel
        return out

    def load_data(self, **kwargs):
        """
        Expected layout:
        - default config: split 'test' with qrels rows: ['query-id','corpus-id','score']
        - named subset 'corpus' with split 'corpus' (docs)
        - named subset 'queries' with split 'queries' (query texts)
        """
        path = self.metadata.dataset["path"]
        rev = self.metadata.dataset.get("revision")

        # --- Load qrels from default config 'test'
        ds_default = load_dataset(path, revision=rev)
        if "test" not in ds_default:
            raise ValueError("Expected default config to contain a 'test' split with qrels.")
        test = ds_default["test"]
        cols = set(test.column_names)
        required = {"query-id", "corpus-id", "score"}
        if not required.issubset(cols):
            raise ValueError(f"'test' split must have {required}, found {sorted(cols)}")

        # Build qrels
        qrels = {}
        for r in test:
            qid = str(r["query-id"])
            did = str(r["corpus-id"])
            rel = int(r.get("score", 1))
            qrels.setdefault(qid, {})[did] = rel

        # --- Load corpus from named subset
        try:
            corpus_rows = load_dataset(path, name="corpus", revision=rev, split="corpus")
        except Exception as e:
            raise ValueError(
                "Could not load the corpus subset (name='corpus', split='corpus'). "
                "Make sure your dataset exposes that subset.\n"
                f"Original error: {e}"
            )

        corpus = {}
        for r in corpus_rows:
            doc_id = r.get("_id") or r.get("id") or r.get("doc_id")
            if doc_id is None:
                raise ValueError("Corpus row missing an id-like column (_id/id/doc_id).")
            doc_id = str(doc_id)
            title = r.get("title")
            text = r.get("text") or r.get("contents") or r.get("passage")
            if text is None:
                raise ValueError("Corpus row missing a text-like column (text/contents/passage).")
            corpus[doc_id] = {"title": title, "text": text}

        # --- Load queries from named subset
        try:
            queries_rows = load_dataset(path, name="queries", revision=rev, split="queries")
            queries = {}
            for r in queries_rows:
                qid = r.get("_id") or r.get("id") or r.get("query_id")
                if qid is None:
                    raise ValueError("Query row missing an id-like column (_id/id/query_id).")
                qid = str(qid)
                qtext = r.get("text") or r.get("query")
                if not qtext or not qtext.strip():
                    # You can choose to drop or raise. Dropping is safer:
                    continue
                queries[qid] = qtext
        except Exception as e:
            raise ValueError(
                "Could not load queries subset (name='queries', split='queries'). "
                "You must provide query texts to evaluate retrieval.\n"
                f"Original error: {e}"
            )

        # --- Filter qrels to valid corpus & queries
        corpus_ids = set(corpus.keys())
        kept_qrels = {}
        kept_queries = {}
        dropped_qids_empty_text = 0
        dropped_qids_missing_docs = 0

        for qid, rels in qrels.items():
            if qid not in queries:
                dropped_qids_empty_text += 1
                continue
            rels_f = {did: rel for did, rel in rels.items() if did in corpus_ids}
            if not rels_f:
                dropped_qids_missing_docs += 1
                continue
            kept_qrels[qid] = rels_f
            kept_queries[qid] = queries[qid]

        log.info(
            f"ChemQuest: corpus={len(corpus)} docs, "
            f"queries={len(queries)} provided, "
            f"qrels={len(qrels)} qids "
            f"-> kept {len(kept_queries)} queries "
            f"(dropped {dropped_qids_empty_text} empty-text, "
            f"{dropped_qids_missing_docs} no-valid-docs)."
        )

        if not kept_qrels:
            raise ValueError("After filtering, no valid queries/qrels remain. Check dataset alignment.")

        # --- Assign to MTEB fields with split keys
        self.corpus = {"test": corpus}
        self.queries = {"test": kept_queries}
        self.relevant_docs = {"test": kept_qrels}

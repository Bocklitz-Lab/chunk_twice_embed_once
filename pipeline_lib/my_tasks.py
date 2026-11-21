# my_tasks.py
from __future__ import annotations
import logging
from typing import Dict, Any, Iterable
from pathlib import Path
import json
import csv

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
            "revision": "5ab7dc673653b800f368fe30851efbdb24385e0d",  # pinned ChemQuests release; see pipeline_lib/dataset_pin.py
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

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.k_values = [1, 3, 5, 10, 100]
    
    def __init__(self, data_dir: str | Path | None = None, qrels_file: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.k_values = [1, 3, 5, 10, 100]
        # keep what came from config so load_data can use it later
        self.data_dir = Path(data_dir).expanduser() if data_dir else None
        # Name of the qrels file inside data_dir (config: labeling.qrels_file, e.g. qrels_span_c<value>.jsonl
        # or qrels_lexical.jsonl). None keeps the old lookup (test.tsv, qrels.tsv, qrels.jsonl).
        self.qrels_file = qrels_file

    # ---------- helpers ----------
    def _read_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _read_tsv(self, path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                yield row

    # Your existing mappers:
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
            text = r.get("text")
            if text is None:
                text = r.get("query")
            if text is None:
                raise ValueError("Query row missing text/query column.")
            out[qid] = text  # an empty question is kept (it scores 0), never dropped
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
            # `relevance` is the label (0 or 1). The continuous lexical `score` written by
            # stage 5 is never read as a label.
            if r.get("relevance") is None:
                raise ValueError(
                    "Qrels row has no 'relevance' column; the 'score' column is never used as a label."
                )
            rel = self._parse_label(r["relevance"])
            out.setdefault(qid, {})[did] = rel
        return out

    @staticmethod
    def _parse_label(value) -> int:
        f = float(value)
        if f not in (0.0, 1.0):
            raise ValueError(f"Relevance labels must be 0 or 1, got {value!r}.")
        return int(f)

    def _finalize_split(self, corpus, queries, qrels_all, origin: str):
        """Build (queries, qrels) keeping EVERY query.

        A query without a positive label stays in the evaluation and scores 0 (nDCG, Recall).
        pytrec_eval only evaluates queries that appear in the qrels, so such a query needs at
        least one row: stage 5 writes a relevance-0 row; if a query has no row at all, a
        relevance-0 row on the first corpus id is added here (it cannot change any score).
        """
        corpus_ids = set(corpus)
        if not corpus_ids:
            raise ValueError("Empty corpus.")
        if not queries:
            raise ValueError("No queries found.")
        placeholder = sorted(corpus_ids)[0]

        kept_qrels: Dict[str, Dict[str, int]] = {}
        n_qrels_unknown_query = sum(1 for q in qrels_all if q not in queries)
        n_without_rows = 0
        for qid in queries:
            rels = {d: r for d, r in qrels_all.get(qid, {}).items() if d in corpus_ids}
            if not rels:
                rels = {placeholder: 0}
                n_without_rows += 1
            kept_qrels[qid] = rels

        n_no_positive = sum(1 for rels in kept_qrels.values() if not any(v > 0 for v in rels.values()))
        log.info(
            f"ChemQuest({origin}): corpus={len(corpus)} docs, queries={len(queries)} "
            f"-> all kept; {n_no_positive} without a positive label (score 0; "
            f"{n_without_rows} had no usable qrels row), "
            f"{n_qrels_unknown_query} qrels query ids not in queries were ignored."
        )
        if n_no_positive == len(queries):
            raise ValueError("No query has a positive label. Check dataset alignment.")
        return dict(queries), kept_qrels

    def load_data(self, data_dir: str | Path | None = None, **kwargs):
        """
        If data_dir is provided (arg or instance attribute), read from local files;
        otherwise fall back to HF Hub.
        """
        # prefer explicit arg, else instance attribute set in __init__
        if data_dir is None:
            data_dir = self.data_dir

        if data_dir is not None:
            data_dir = Path(data_dir)
            print(data_dir)
            if not data_dir.exists():
                raise FileNotFoundError(f"data_dir not found: {data_dir}")

            # --- read corpus
            corpus_path = data_dir / "corpus.jsonl"
            if not corpus_path.exists():
                raise FileNotFoundError(f"Missing {corpus_path}")
            corpus_rows = list(self._read_jsonl(corpus_path))
            corpus = self._to_corpus(corpus_rows)

            # --- read queries
            queries_path = data_dir / "queries.jsonl"
            if not queries_path.exists():
                raise FileNotFoundError(f"Missing {queries_path}")
            queries_rows = list(self._read_jsonl(queries_path))
            queries = self._to_queries(queries_rows)

            # --- read qrels
            if self.qrels_file:
                # The config names the label set. Exactly that file is used; nothing else is tried, so a
                # leftover file of another label set can never be evaluated by mistake.
                named = data_dir / self.qrels_file
                if not named.exists():
                    raise FileNotFoundError(f"Missing qrels file named in the config (labeling.qrels_file): {named}")
                qrels_rows = list(self._read_tsv(named) if named.suffix.lower() == ".tsv" else self._read_jsonl(named))
                log.info(f"ChemQuest: qrels from {named.name}")
            else:
                qrels_dir = data_dir
                tsv = qrels_dir / "test.tsv"
                alt_tsv = data_dir / "qrels.tsv"
                jsonl = qrels_dir / "qrels.jsonl"

                if tsv.exists():
                    qrels_rows = list(self._read_tsv(tsv))
                elif alt_tsv.exists():
                    qrels_rows = list(self._read_tsv(alt_tsv))
                elif jsonl.exists():
                    qrels_rows = list(self._read_jsonl(jsonl))
                else:
                    raise FileNotFoundError(
                        f"Missing qrels. Looked for {tsv}, {alt_tsv}, {jsonl}"
                    )
            qrels_all = self._to_qrels(qrels_rows)
            if self.qrels_file:
                # a labelled chunk that is not in the corpus means qrels and corpus come from different runs
                unknown = sorted({d for rels in qrels_all.values() for d in rels if d not in corpus})
                if unknown:
                    raise ValueError(
                        f"{self.qrels_file} refers to {len(unknown)} chunk ids that are not in corpus.jsonl "
                        f"(first: {unknown[:3]}); the qrels do not belong to this corpus."
                    )

            kept_queries, kept_qrels = self._finalize_split(corpus, queries, qrels_all, "local")

            self.corpus = {"test": corpus}
            self.queries = {"test": kept_queries}
            self.relevant_docs = {"test": kept_qrels}
            return  # done

    # def load_data(self, data_dir: str | Path | None = None, **kwargs):
    #     """
    #     If data_dir is provided, read from local files; otherwise fall back to HF Hub.

    #     Local layout (recommended):
    #     data_dir/
    #       corpus.jsonl            # fields: _id/id/doc_id, title?, text/contents/passage
    #       queries.jsonl           # fields: _id/id/query_id, text/query
    #       qrels/
    #         test.tsv              # tab-separated columns: query-id corpus-id score
    #     """
    #     if data_dir is not None:
    #         data_dir = Path(data_dir)
    #         print(data_dir)
    #         if not data_dir.exists():
    #             raise FileNotFoundError(f"data_dir not found: {data_dir}")

    #         # --- read corpus
    #         corpus_path = data_dir / "corpus.jsonl"
    #         if not corpus_path.exists():
    #             raise FileNotFoundError(f"Missing {corpus_path}")
    #         corpus_rows = list(self._read_jsonl(corpus_path))
    #         corpus = self._to_corpus(corpus_rows)

    #         # --- read queries
    #         queries_path = data_dir / "queries.jsonl"
    #         if not queries_path.exists():
    #             raise FileNotFoundError(f"Missing {queries_path}")
    #         queries_rows = list(self._read_jsonl(queries_path))
    #         queries = self._to_queries(queries_rows)

    #         # --- read qrels (prefer TSV at qrels/test.tsv)
    #         qrels_dir = data_dir / "qrels"
    #         tsv = qrels_dir / "test.tsv"
    #         alt_tsv = data_dir / "qrels.tsv"
    #         jsonl = qrels_dir / "test.jsonl"

    #         if tsv.exists():
    #             qrels_rows = list(self._read_tsv(tsv))
    #         elif alt_tsv.exists():
    #             qrels_rows = list(self._read_tsv(alt_tsv))
    #         elif jsonl.exists():
    #             qrels_rows = list(self._read_jsonl(jsonl))
    #         else:
    #             raise FileNotFoundError(
    #                 f"Missing qrels. Looked for {tsv}, {alt_tsv}, {jsonl}"
    #             )
    #         qrels_all = self._to_qrels(qrels_rows)

    #         # --- filter qrels to valid corpus & queries (same logic as before)
    #         corpus_ids = set(corpus.keys())
    #         kept_qrels = {}
    #         kept_queries = {}
    #         dropped_qids_empty_text = 0
    #         dropped_qids_missing_docs = 0

    #         for qid, rels in qrels_all.items():
    #             if qid not in queries:
    #                 dropped_qids_empty_text += 1
    #                 continue
    #             rels_f = {did: rel for did, rel in rels.items() if did in corpus_ids}
    #             if not rels_f:
    #                 dropped_qids_missing_docs += 1
    #                 continue
    #             kept_qrels[qid] = rels_f
    #             kept_queries[qid] = queries[qid]

    #         log.info(
    #             f"ChemQuest(local): corpus={len(corpus)} docs, "
    #             f"queries={len(queries)} provided, qrels={len(qrels_all)} qids "
    #             f"-> kept {len(kept_queries)} queries "
    #             f"(dropped {dropped_qids_empty_text} empty-text, "
    #             f"{dropped_qids_missing_docs} no-valid-docs)."
    #         )

    #         if not kept_qrels:
    #             raise ValueError("After filtering, no valid queries/qrels remain. Check dataset alignment.")

    #         self.corpus = {"test": corpus}
    #         self.queries = {"test": kept_queries}
    #         self.relevant_docs = {"test": kept_qrels}
    #         return  # done

        # ---------- fallback: HF Hub path ----------
        path = self.metadata.dataset["path"]
        rev = self.metadata.dataset.get("revision")

        ds_default = load_dataset(path, revision=rev)
        if "test" not in ds_default:
            raise ValueError("Expected default config to contain a 'test' split with qrels.")
        test = ds_default["test"]
        cols = set(test.column_names)
        required = {"query-id", "corpus-id", "relevance"}
        if not required.issubset(cols):
            raise ValueError(f"'test' split must have {required}, found {sorted(cols)}")
        qrels = self._to_qrels(test)

        try:
            corpus_rows = load_dataset(path, name="corpus", revision=rev, split="corpus")
        except Exception as e:
            raise ValueError(
                "Could not load the corpus subset (name='corpus', split='corpus'). "
                f"Original error: {e}"
            )
        corpus = self._to_corpus(corpus_rows)

        try:
            queries_rows = load_dataset(path, name="queries", revision=rev, split="queries")
        except Exception as e:
            raise ValueError(
                "Could not load queries subset (name='queries', split='queries'). "
                f"Original error: {e}"
            )
        queries = self._to_queries(queries_rows)

        kept_queries, kept_qrels = self._finalize_split(corpus, queries, qrels, "hf")

        self.corpus = {"test": corpus}
        self.queries = {"test": kept_queries}
        self.relevant_docs = {"test": kept_qrels}

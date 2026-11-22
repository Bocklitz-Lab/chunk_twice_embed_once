"""BM25 baseline over the chunks of a prepared task (rank_bm25, BM25Okapi).

Runs on exactly the corpus / queries / qrels the dense models use. The text of a chunk is the chunk text itself (no
document title). Tokens: Unicode word characters, lower-cased; no stemming and no stop-word list by default.
The index covers the whole corpus (all papers), like the dense retrievers.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    return _WORD.findall(text.lower() if lowercase else text)


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25, lowercase: bool = True):
        self.k1, self.b, self.epsilon, self.lowercase = k1, b, epsilon, lowercase
        self.timing = {"bm25_index_s": 0.0, "bm25_search_s": 0.0}

    def index(self, corpus: Dict[str, Dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        t0 = time.perf_counter()
        self.ids = sorted(corpus)                                 # deterministic corpus order
        toks = [tokenize(corpus[i]["text"], self.lowercase) for i in self.ids]
        self.bm25 = BM25Okapi(toks, k1=self.k1, b=self.b, epsilon=self.epsilon)
        # rank of every document id in DESCENDING id order: ties in the score are broken like trec_eval (id descending)
        order = sorted(range(len(self.ids)), key=lambda i: self.ids[i], reverse=True)
        self._tie_rank = np.empty(len(self.ids), dtype=np.int64)
        self._tie_rank[order] = np.arange(len(self.ids))
        self.timing["bm25_index_s"] += time.perf_counter() - t0

    def search(self, queries: Dict[str, str], top_k: int = 100) -> Dict[str, Dict[str, float]]:
        t0 = time.perf_counter()
        out: Dict[str, Dict[str, float]] = {}
        for qid, text in queries.items():
            scores = np.asarray(self.bm25.get_scores(tokenize(text, self.lowercase)), dtype=np.float64)
            order = np.lexsort((self._tie_rank, -scores))[:top_k]     # score descending, then id descending
            out[str(qid)] = {self.ids[i]: float(scores[i]) for i in order}
        self.timing["bm25_search_s"] += time.perf_counter() - t0
        return out

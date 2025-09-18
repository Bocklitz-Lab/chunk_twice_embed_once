import os, glob, importlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .utils import cfg_get

@dataclass
class ChunkRecord:
    document_name: str
    start_index: int
    end_index: int
    chunk_text: str

class SimpleChunker:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        assert chunk_size>0 and 0<=chunk_overlap<chunk_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_text(self, text: str) -> List[str]:
        out, n, step, i = [], len(text), self.chunk_size-self.chunk_overlap, 0
        while i < n:
            out.append(text[i: min(n, i+self.chunk_size)])
            if i + self.chunk_size >= n: break
            i += step
        return out

class ChunkerFactory:
    # Default registry for your modules; override via config.chunking.class_path if needed
    DEFAULT_CLASS_REGISTRY = {
        "recursive_token": "recursive:RecursiveTokenChunker",
        "fixed_token": "Fixed:FixedTokenChunker",
        "kamradt_modified": "Karmadt:KamradtModifiedChunker",
        "cluster_semantic": "Cluster:ClusterSemanticChunker",
        "llm_semantic": "LLM:LLMSemanticChunker",
    }
    @staticmethod
    def _import_class(path: str):
        module_name, class_name = path.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def __init__(self, cfg, embedding_function=None):
        self.cfg = cfg
        self.embedding_function = embedding_function

    def build(self):
        strategy = cfg_get(self.cfg, "chunking.strategy", "simple")
        if strategy == "simple":
            p = cfg_get(self.cfg, "chunking.simple", {}) or {}
            return SimpleChunker(p.get("chunk_size",800), p.get("chunk_overlap",100))

        class_path = cfg_get(self.cfg, "chunking.class_path", None)
        if class_path is None:
            if strategy not in self.DEFAULT_CLASS_REGISTRY:
                raise ValueError(f"Unknown chunking.strategy: {strategy}")
            class_path = self.DEFAULT_CLASS_REGISTRY[strategy]
        ChunkerClass = self._import_class(class_path)

        if strategy == "recursive_token":
            p = cfg_get(self.cfg, "chunking.recursive_token", {}) or {}
            return ChunkerClass(
                chunk_size=int(p.get("chunk_size",2000)),
                chunk_overlap=int(p.get("chunk_overlap",100)),
                keep_separator=bool(p.get("keep_separator",True)),
                separators=p.get("separators", ["\n\n","\n",".","?","!"," ",""]),
                is_separator_regex=bool(p.get("is_separator_regex", False)),
            )
        if strategy == "fixed_token":
            p = cfg_get(self.cfg, "chunking.fixed_token", {}) or {}
            return ChunkerClass(
                encoding_name=p.get("encoding_name","cl100k_base"),
                model_name=p.get("model_name", None),
                chunk_size=int(p.get("chunk_size",1000)),
                chunk_overlap=int(p.get("chunk_overlap",100)),
                allowed_special=set(p.get("allowed_special", [])),
                disallowed_special=p.get("disallowed_special","all"),
            )
        if strategy == "kamradt_modified":
            p = cfg_get(self.cfg, "chunking.kamradt_modified", {}) or {}
            return ChunkerClass(
                avg_chunk_size=int(p.get("avg_chunk_size",400)),
                min_chunk_size=int(p.get("min_chunk_size",50)),
                embedding_function=self.embedding_function,
            )
        if strategy == "cluster_semantic":
            p = cfg_get(self.cfg, "chunking.cluster_semantic", {}) or {}
            return ChunkerClass(
                max_chunk_size=int(p.get("max_chunk_size",400)),
                min_chunk_size=int(p.get("min_chunk_size",50)),
                embedding_function=self.embedding_function,
            )
        if strategy == "llm_semantic":
            p = cfg_get(self.cfg, "chunking.llm_semantic", {}) or {}
            return ChunkerClass(
                organisation=p.get("organisation","openai"),
                api_key=p.get("api_key", None),
                model_name=p.get("model_name","gpt-4o"),
            )
        raise ValueError(f"Unsupported strategy: {strategy}")

    @staticmethod
    def materialize_with_offsets(full_text: str, doc_name: str, chunk_texts: List[str]) -> List[ChunkRecord]:
        chunks, cursor, n = [], 0, len(full_text)
        for ch in chunk_texts:
            if not ch: continue
            pos = full_text.find(ch, cursor)
            if pos == -1: pos = full_text.find(ch)
            if pos == -1:
                start, end = cursor, min(n, cursor+len(ch))
            else:
                start, end = pos, pos+len(ch)
            chunks.append(ChunkRecord(doc_name, start, end, ch))
            cursor = end
        return chunks

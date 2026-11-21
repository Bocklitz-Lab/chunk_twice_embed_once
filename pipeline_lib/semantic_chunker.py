# MIT License
# Semantic splitter wrapper using LangChain's SemanticChunker (+ HF token-size enforcement)

import hashlib
import json
import logging
import os
import re
import tempfile
from typing import Any, List, Optional, Sequence, Dict, Tuple

log = logging.getLogger(__name__)

# Bump when the meaning of a cached segmentation changes (it is part of the cache key).
_SEGMENT_CACHE_VERSION = 1

# Document
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document  # older LC
    except Exception:
        Document = None  # type: ignore

# SemanticChunker (location varies by LC version)
_semantic_cls = None
for path in [
    "langchain_experimental.text_splitter",
    "langchain_experimental.text_splitters",
    "langchain_experimental.document_transformers",
]:
    try:
        mod = __import__(path, fromlist=["SemanticChunker"])
        _semantic_cls = getattr(mod, "SemanticChunker")
        break
    except Exception:
        pass
if _semantic_cls is None:
    raise ImportError(
        "Could not import SemanticChunker. Please install/upgrade langchain-experimental:\n"
        "  pip install -U langchain-experimental"
    )

# HF embeddings (new & old LC package locations)
_HFEmb = None
for path, name in [
    ("langchain_huggingface", "HuggingFaceEmbeddings"),   # pip install -U langchain-huggingface
    ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
    ("langchain.embeddings", "HuggingFaceEmbeddings"),  # very old LC
]:
    try:
        mod = __import__(path, fromlist=[name])
        _HFEmb = getattr(mod, name)
        break
    except Exception:
        pass
if _HFEmb is None:
    raise ImportError(
        "Could not import HuggingFaceEmbeddings. Install one of:\n"
        "  pip install -U langchain-huggingface\n"
        "  pip install -U langchain-community sentence-transformers"
    )

# Optional RecursiveTokenChunker (your project/local)
try:
    from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker
except ImportError:
    try:
        from recursive_token_chunker import RecursiveTokenChunker
    except Exception:
        RecursiveTokenChunker = None  # type: ignore

# HF tokenizer
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    raise ImportError(
        "transformers is required for HF token counting. Install:\n"
        "  pip install -U transformers"
    )


class _HFTokenTextSplitter:
    """
    Minimal tokenizer-aware splitter using a Hugging Face tokenizer.
    Splits by token count (chunk_size / chunk_overlap) and cuts each window out of the
    ORIGINAL text using the tokenizer's character offsets, so every chunk is an exact
    substring of the input (no decode round trip: that lower-cased and re-spaced the text).

    Notes:
      - add_special_tokens=False for counting.
      - Requires a fast tokenizer (offset_mapping).
    """

    def __init__(
        self,
        *,
        tokenizer,
        chunk_size: int,
        chunk_overlap: int,
        strip_whitespace: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be <= chunk_size")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_whitespace = strip_whitespace

    def split_text(self, text: str) -> List[str]:
        return [piece for piece, _, _ in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """(piece, start, end) with ``text[start:end] == piece``; positions come from the token windows."""
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )
        offsets = enc["offset_mapping"]
        if not offsets:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        out: List[Tuple[str, int, int]] = []
        for start in range(0, len(offsets), step):
            window = offsets[start : start + self.chunk_size]
            if not window:
                continue
            a, b = window[0][0], window[-1][1]
            piece = text[a:b]
            if self.strip_whitespace:
                stripped = piece.strip()
                if stripped:
                    a += len(piece) - len(piece.lstrip())
                piece = stripped
                b = a + len(piece)
            # skip a window whose characters all lie inside the previous chunk (see fixed-token chunker)
            if piece and not (out and b <= out[-1][2]):
                out.append((piece, a, b))
            if start + self.chunk_size >= len(offsets):
                break  # this window reaches the end of the segment: another one would lie inside it
        return out


class SemanticSplitter:
    """
    Semantic splitter using LangChain's SemanticChunker for breakpoint detection,
    with token-based re-chunking to enforce `chunk_size` and `chunk_overlap`
    using a Hugging Face tokenizer.

    Public API (unchanged):
      - split_text(text: str) -> List[str]
      - create_documents(texts: Sequence[str], metadatas: Optional[Sequence[dict]]) -> List[Document]

    Init parameters kept:
      model_name: str (required, no default)           # HF model id for both embeddings + tokenizer
      chunk_size: int = 4000
      chunk_overlap: int = 200
      use_recursive: bool = False
      breakpoint_threshold_type: str = "percentile"
      breakpoint_threshold_amount: float = 95.0
      buffer_size: int = 1
      min_chunk_size: Optional[int] = None
      add_start_index: bool = False
      strip_whitespace: bool = True
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        use_recursive: bool = False,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1,
        min_chunk_size: Optional[int] = None,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        # passthroughs
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        **_: Any,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be <= chunk_size")
        if Document is None:
            raise ImportError("Could not import LangChain Document class.")
        if not model_name:
            raise ValueError("SemanticSplitter needs model_name (embedding model and tokenizer; no default).")

        # --- Build HF pieces from model_name ---
        # 1) Embeddings for SemanticChunker
        if model_kwargs is None:
            model_kwargs = {"trust_remote_code": True}

        emb_kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
        }
        if encode_kwargs is not None:
            # pydantic requires dict, not None
            emb_kwargs["encode_kwargs"] = encode_kwargs

        embeddings = _HFEmb(**emb_kwargs)

        # ---- Tokenizer (trust flag) ----
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {"trust_remote_code": True}
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, **tokenizer_kwargs
        )
        # --- SemanticChunker (finds semantic breakpoints) ---
        self._semantic = _semantic_cls(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
            min_chunk_size=min_chunk_size,
        )

        # --- Enforcement splitter (HF tokenizer-based) ---
        self._use_recursive = use_recursive
        if use_recursive:
            if RecursiveTokenChunker is None:
                raise ImportError("RecursiveTokenChunker not available.")
            # Measure chunk size with the retrieval model's own tokenizer. Without
            # model_name, RecursiveTokenChunker silently falls back to bert-base-uncased.
            self._token_splitter = RecursiveTokenChunker(
                model_name=model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                # "" last: a run without any separator (paper 91: ~1,000 characters of "~/~/...")
                # is split by characters instead of staying one oversized chunk.
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
                keep_separator=True,
            )
        else:
            self._token_splitter = _HFTokenTextSplitter(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strip_whitespace=strip_whitespace,
            )

        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

        # Optional on-disk cache of the semantic segmentation. The breakpoints depend on the
        # embedding model, the document and the breakpoint parameters, but NOT on chunk_size or
        # chunk_overlap, so one entry serves every size/overlap of the same (model, document).
        self._cache_dir = os.fspath(cache_dir) if cache_dir else None
        self._encode_kwargs = dict(encode_kwargs) if encode_kwargs else None
        self._model_kwargs = dict(model_kwargs) if model_kwargs else None
        self.cache_hits = 0
        self.cache_misses = 0

        self._config = dict(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_recursive=use_recursive,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
            min_chunk_size=min_chunk_size,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
            tokenizer="hf",
            embeddings="hf",
        )

    def _cache_key(self, text: str) -> str:
        """sha256 over everything that determines the segmentation of ``text``.

        model name, document text hash and the splitter parameters (breakpoint type/amount,
        buffer size, min chunk size, number_of_chunks, sentence regex, embedding kwargs) plus the
        langchain-experimental version. Not chunk_size / chunk_overlap / strategy.
        """
        sc = self._semantic
        try:
            from importlib.metadata import version
            lc_version = version("langchain-experimental")
        except Exception:
            lc_version = "unknown"
        payload = {
            "cache_version": _SEGMENT_CACHE_VERSION,
            "model_name": self._config["model_name"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "breakpoint_threshold_type": self._config["breakpoint_threshold_type"],
            "breakpoint_threshold_amount": self._config["breakpoint_threshold_amount"],
            "buffer_size": self._config["buffer_size"],
            "min_chunk_size": self._config["min_chunk_size"],
            "number_of_chunks": getattr(sc, "number_of_chunks", None),
            "sentence_split_regex": getattr(sc, "sentence_split_regex", None),
            "encode_kwargs": self._encode_kwargs,
            "model_kwargs": self._model_kwargs,
            "langchain_experimental": lc_version,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _valid_spans(spans: Any, n: int) -> bool:
        if not isinstance(spans, list) or not spans:
            return False
        prev_end = 0
        for sp in spans:
            if not (isinstance(sp, (list, tuple)) and len(sp) == 2 and all(isinstance(v, int) for v in sp)):
                return False
            a, b = sp
            if not (prev_end <= a <= b <= n):
                return False
            prev_end = b
        return True

    def _segment_spans(self, text: str) -> List[Tuple[int, int]]:
        """Semantic segment spans of ``text``, served from the on-disk cache when enabled.

        The cache never changes a result: an entry is used only if its key matches and its spans
        are valid for this text; anything else is recomputed and rewritten.
        """
        if not self._cache_dir:
            return self._compute_segment_spans(text)
        path = os.path.join(self._cache_dir, f"{self._cache_key(text)}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                spans = json.load(f)["spans"]
            if self._valid_spans(spans, len(text)):
                self.cache_hits += 1
                return [(a, b) for a, b in spans]
            log.warning("ignoring invalid semantic cache entry %s", path)
        except FileNotFoundError:
            pass
        except Exception as e:  # unreadable / corrupt entry: recompute
            log.warning("ignoring unreadable semantic cache entry %s (%s)", path, e)

        spans = self._compute_segment_spans(text)
        self.cache_misses += 1
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"model_name": self._config["model_name"], "n_chars": len(text),
                           "spans": [list(sp) for sp in spans]}, f)
            os.replace(tmp, path)  # atomic: parallel runs never see a half-written file
        except Exception as e:
            log.warning("could not write semantic cache entry %s (%s)", path, e)
        return spans

    def _compute_segment_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        (start, end) character spans, in the ORIGINAL text, of the semantic segments.

        Same breakpoint logic as langchain_experimental's SemanticChunker.split_text
        (0.3.x), but it returns spans instead of the strings LangChain builds with
        " ".join(sentences). That re-joining replaces the original whitespace between
        sentences, so its segments are not substrings of the document.
        """
        sc = self._semantic
        needed = ("sentence_split_regex", "min_chunk_size", "number_of_chunks", "breakpoint_threshold_type",
                  "_calculate_sentence_distances", "_calculate_breakpoint_threshold", "_threshold_from_clusters")
        missing = [a for a in needed if not hasattr(sc, a)]
        if missing:
            raise RuntimeError(f"Unsupported SemanticChunker API (missing {missing}); langchain_experimental 0.3.x is required.")

        # Sentence spans, using the same regex LangChain splits on.
        spans: List[Tuple[int, int]] = []
        pos = 0
        for m in re.finditer(sc.sentence_split_regex, text):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, len(text)))
        sentences = [text[a:b] for a, b in spans]
        if sentences != re.split(sc.sentence_split_regex, text):
            raise RuntimeError("sentence_split_regex yields different pieces with finditer and split (capture group?).")

        if len(sentences) == 1:
            return spans
        if sc.breakpoint_threshold_type == "gradient" and len(sentences) == 2:
            return spans

        distances, _ = sc._calculate_sentence_distances(sentences)
        if sc.number_of_chunks is not None:
            threshold, breakpoints = sc._threshold_from_clusters(distances), distances
        else:
            threshold, breakpoints = sc._calculate_breakpoint_threshold(distances)

        groups: List[Tuple[int, int]] = []
        first = 0  # first sentence of the current group
        for i, x in enumerate(breakpoints):
            if x <= threshold:
                continue
            # min_chunk_size is measured on the " "-joined text, as LangChain does.
            if sc.min_chunk_size is not None and len(" ".join(sentences[first : i + 1])) < sc.min_chunk_size:
                continue
            groups.append((spans[first][0], spans[i][1]))
            first = i + 1
        if first < len(sentences):
            groups.append((spans[first][0], spans[-1][1]))
        return groups

    def _semantic_segments(self, text: str) -> List[str]:
        # Verbatim slices of the original text, so every chunk derived from them is too.
        return [text[a:b] for a, b in self._segment_spans(text)]

    def split_text(self, text: str) -> List[str]:
        """
        1) Split semantically into coherent segments.
        2) Within each segment, enforce token-based chunk_size/overlap with HF tokenizer.
        """
        return [piece for piece, _, _ in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """[(chunk, start, end)] with ``text[start:end] == chunk`` (end exclusive).

        Positions are native: the segment span from ``_segment_spans`` plus the position of the
        chunk inside the segment (token-window start for semantic-fixed, a search confined to
        the segment for semantic-recursive).
        """
        out: List[Tuple[str, int, int]] = []
        for a, b in self._segment_spans(text):
            seg = text[a:b]
            for piece, s, e in self._token_splitter.split_text_with_offsets(seg):
                out.append((piece, a + s, a + e))
        return out

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict]] = None,
    ):
        """
        Returns a flat list of Documents created from the input texts using
        semantic splitting + token enforcement. If `add_start_index=True`,
        includes a character 'start_index' in metadata for each chunk.
        """
        if metadatas is not None:
            if len(metadatas) == 1 and len(texts) > 1:
                metadatas = list(metadatas) * len(texts)
            elif len(metadatas) != len(texts):
                raise ValueError("metadatas length must be 1 or match texts length")

        out_docs: List[Document] = []
        for i, text in enumerate(texts):
            base_meta = metadatas[i] if metadatas else {}
            for ch, start, _end in self.split_text_with_offsets(text):
                meta = dict(base_meta)
                if self._add_start_index:
                    meta["start_index"] = start
                out_docs.append(Document(page_content=ch, metadata=meta))
        return out_docs

    @property
    def n_ambiguous(self) -> int:
        """Chunks (recursive variant only) whose text matched more than once in their window."""
        return getattr(self._token_splitter, "n_ambiguous", 0)

    @property
    def n_relaxed(self) -> int:
        return getattr(self._token_splitter, "n_relaxed", 0)

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def chunk_size(self) -> int:
        return self._config["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["chunk_overlap"]

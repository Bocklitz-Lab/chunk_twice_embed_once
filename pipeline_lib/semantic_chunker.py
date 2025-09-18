# MIT License
# Semantic splitter wrapper using LangChain's SemanticChunker (+ token-size enforcement)

from typing import Any, List, Optional, Sequence, Dict

# Embeddings + Document
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    # older LC
    from langchain.embeddings.base import Embeddings  # type: ignore

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

# TokenTextSplitter (new & old package names)
try:
    from langchain_text_splitters import TokenTextSplitter as LCTokenTextSplitter
except ImportError:
    from langchain.text_splitter import TokenTextSplitter as LCTokenTextSplitter  # type: ignore

# RecursiveTokenChunker
try:
    from pipeline_lib.recursive_token_chunker import RecursiveTokenChunker
except ImportError:
    try:
        from recursive_token_chunker import RecursiveTokenChunker
    except Exception:
        RecursiveTokenChunker = None  # type: ignore

class SemanticSplitter:
    """
    Semantic splitter using LangChain's SemanticChunker for breakpoint detection,
    with optional token-based re-chunking to enforce model-friendly `chunk_size`
    and `chunk_overlap`.

    API:
      - split_text(text: str) -> List[str]
      - create_documents(texts: Sequence[str], metadatas: Optional[Sequence[dict]]) -> List[Document]

    Key params:
      embeddings: Embeddings (required)  # e.g., from langchain_openai, sentence-transformers, etc.
      chunk_size: int = 4000             # token-based max size per final chunk (post semantic)
      chunk_overlap: int = 200           # token overlap (applied during token enforcement)
      breakpoint_threshold_type: str = "percentile"  # per LC SemanticChunker
      breakpoint_threshold_amount: float = 95.0
      buffer_size: int = 1               # sentence buffer around breakpoints
      min_chunk_size: Optional[int] = None  # semantic-side minimum tokens (LC may treat as chars in some versions)

    Tokenization knobs (forwarded to TokenTextSplitter):
      encoding_name: str = "cl100k_base"
      model_name: Optional[str] = None
      allowed_special = set()
      disallowed_special = "all"
      strip_whitespace: bool = True
      add_start_index: bool = False  # for create_documents: adds character start offsets
    """

    def __init__(
        self,
        *,
        embeddings: "Embeddings",
        # token enforcement
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        use_recursive: bool = False,
        # semantic knobs
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1,
        min_chunk_size: Optional[int] = None,
        # tokenization knobs
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        allowed_special=None,
        disallowed_special="all",
        strip_whitespace: bool = True,
        add_start_index: bool = False,
        # compatibility / extras
        **kwargs: Any,
    ) -> None:
        if allowed_special is None:
            allowed_special = set()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be <= chunk_size")

        # --- SemanticChunker (finds semantic breakpoints) ---
        # Different LC versions have slightly different constructor signatures;
        # we pass the common ones and ignore unknown **kwargs here.
        self._semantic = _semantic_cls(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
            min_chunk_size=min_chunk_size,
        )

        # --- Token splitter (enforce final token size / overlap) ---
        # --- Enforcement splitter ---
        self._use_recursive = use_recursive
        if use_recursive:
            if RecursiveTokenChunker is None:
                raise ImportError("RecursiveTokenChunker not available.")
            self._token_splitter = RecursiveTokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "?", "!", " "],
                keep_separator=True,
            )
        else:
            self._token_splitter = LCTokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name=encoding_name,
                model_name=model_name,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
                add_start_index=False,  # we compute offsets ourselves in create_documents if requested
                strip_whitespace=strip_whitespace,
            )


        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

        self._config = dict(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_recursive=use_recursive,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            buffer_size=buffer_size,
            min_chunk_size=min_chunk_size,
            encoding_name=encoding_name,
            model_name=model_name,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

    def _semantic_segments(self, text: str) -> List[str]:
        # LangChain SemanticChunker typically exposes split_text
        # (older variants may expose a transform that returns Documents).
        if hasattr(self._semantic, "split_text"):
            return self._semantic.split_text(text)
        # Fallback: try transform_documents API if split_text is unavailable
        if hasattr(self._semantic, "create_documents"):
            docs = self._semantic.create_documents([text])
            return [d.page_content for d in docs]
        if hasattr(self._semantic, "transform_documents"):
            docs = self._semantic.transform_documents([Document(page_content=text)])  # type: ignore
            return [d.page_content for d in docs]
        raise RuntimeError("Unsupported SemanticChunker API in this LangChain version.")

    def split_text(self, text: str) -> List[str]:
        """
        1) Split semantically into coherent segments.
        2) Within each segment, enforce token-based chunk_size/overlap.
        """
        segments = self._semantic_segments(text)

        chunks: List[str] = []
        for seg in segments:
            # Enforce token-size limits per segment
            sub_chunks = self._token_splitter.split_text(seg)
            chunks.extend(sub_chunks)
        return chunks

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
        if Document is None:
            raise ImportError("Could not import LangChain Document class.")

        if metadatas is not None:
            if len(metadatas) == 1 and len(texts) > 1:
                metadatas = list(metadatas) * len(texts)
            elif len(metadatas) != len(texts):
                raise ValueError("metadatas length must be 1 or match texts length")

        out_docs: List[Document] = []
        for i, text in enumerate(texts):
            base_meta = metadatas[i] if metadatas else {}
            # Do the same 2-stage split as split_text
            segments = self._semantic_segments(text)

            # Build final chunks and compute start indices if requested
            cursor = 0
            for seg in segments:
                sub_chunks = self._token_splitter.split_text(seg)
                for ch in sub_chunks:
                    start_idx = None
                    if self._add_start_index:
                        # Find the next occurrence of chunk from current cursor
                        pos = text.find(ch, cursor)
                        if pos == -1:
                            # Fallback: search from 0 (may duplicate if text repeats)
                            pos = text.find(ch)
                        if pos != -1:
                            start_idx = pos
                            cursor = pos + len(ch)

                    meta = dict(base_meta)
                    if start_idx is not None:
                        meta["start_index"] = start_idx
                    out_docs.append(Document(page_content=ch, metadata=meta))

        return out_docs

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def chunk_size(self) -> int:
        return self._config["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["chunk_overlap"]

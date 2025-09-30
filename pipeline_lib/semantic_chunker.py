# MIT License
# Semantic splitter wrapper using LangChain's SemanticChunker (+ HF token-size enforcement)

from typing import Any, List, Optional, Sequence, Dict

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
    Splits by token count (chunk_size / chunk_overlap) and decodes back to text.

    Notes:
      - add_special_tokens=False for counting; decode with skip_special_tokens=True.
      - Decoding may slightly alter whitespace; start_index is best-effort via substring search.
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
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: List[str] = []
        for start in range(0, len(ids), step):
            window_ids = ids[start : start + self.chunk_size]
            if not window_ids:
                continue
            piece = self.tokenizer.decode(window_ids, skip_special_tokens=True)
            if self.strip_whitespace:
                piece = piece.strip()
            if piece:
                chunks.append(piece)
        return chunks


class SemanticSplitter:
    """
    Semantic splitter using LangChain's SemanticChunker for breakpoint detection,
    with token-based re-chunking to enforce `chunk_size` and `chunk_overlap`
    using a Hugging Face tokenizer.

    Public API (unchanged):
      - split_text(text: str) -> List[str]
      - create_documents(texts: Sequence[str], metadatas: Optional[Sequence[dict]]) -> List[Document]

    Init parameters kept:
      model_name: str                                  # HF model id for both embeddings + tokenizer
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
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
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
            # If your RecursiveTokenChunker supports tokenizer, pass it here similarly.
            self._token_splitter = RecursiveTokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "?", "!", " "],
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

    def _semantic_segments(self, text: str) -> List[str]:
        # Prefer split_text; fallback to create_documents/transform_documents
        if hasattr(self._semantic, "split_text"):
            return self._semantic.split_text(text)
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
        2) Within each segment, enforce token-based chunk_size/overlap with HF tokenizer.
        """
        segments = self._semantic_segments(text)
        chunks: List[str] = []
        for seg in segments:
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
        if metadatas is not None:
            if len(metadatas) == 1 and len(texts) > 1:
                metadatas = list(metadatas) * len(texts)
            elif len(metadatas) != len(texts):
                raise ValueError("metadatas length must be 1 or match texts length")

        out_docs: List[Document] = []
        for i, text in enumerate(texts):
            base_meta = metadatas[i] if metadatas else {}
            segments = self._semantic_segments(text)

            cursor = 0
            for seg in segments:
                sub_chunks = self._token_splitter.split_text(seg)
                for ch in sub_chunks:
                    start_idx = None
                    if self._add_start_index:
                        pos = text.find(ch, cursor)
                        if pos == -1:
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

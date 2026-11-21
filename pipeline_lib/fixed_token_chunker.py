from typing import Any, List, Optional, Sequence, Type, TypeVar, Tuple, Dict

# LangChain Document import (support both v0.1+ and v0.2+ package names)
try:
    from langchain_core.documents import Document  # v0.2+
except Exception:
    try:
        from langchain.schema import Document  # v0.1
    except Exception:
        Document = None  # type: ignore

# Prefer Hugging Face tokenizers
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
except ImportError as e:
    raise ImportError(
        "transformers is required for FixedTokenChunker. Install with: pip install transformers"
    ) from e


TS = TypeVar("TS", bound="FixedTokenChunker")


class _HFTokenTextSplitter:
    """
    Minimal, self-contained token-based splitter using a Hugging Face tokenizer.
    Chunks by token count with overlap; uses offsets to map back to char spans.
    """

    _TOKENIZER_CACHE: Dict[str, PreTrainedTokenizerBase] = {}

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        model_name: Optional[str],
        add_start_index: bool,
        strip_whitespace: bool,
    ):
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be <= chunk_size ({chunk_size})"
            )

        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.add_start_index = bool(add_start_index)
        self.strip_whitespace = bool(strip_whitespace)

        tok_name = model_name or "gpt2"  # default to a widely available HF tokenizer
        self.tokenizer = self._get_tokenizer(tok_name)

    @classmethod
    def _get_tokenizer(cls, name: str) -> PreTrainedTokenizerBase:
        if name not in cls._TOKENIZER_CACHE:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            cls._TOKENIZER_CACHE[name] = tok
        return cls._TOKENIZER_CACHE[name]

    def _tokenize_with_offsets(self, text: str):
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )
        offsets = enc.get("offset_mapping")
        if offsets is None and hasattr(enc, "encodings") and enc.encodings:
            offsets = enc.encodings[0].offsets
        if offsets is None:
            offsets = [(0, len(text))] * len(enc["input_ids"])
        input_ids = enc["input_ids"]
        return input_ids, offsets

    def split_text(self, text: str) -> List[str]:
        return [piece for piece, _, _ in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunks as (piece, start, end) with ``text[start:end] == piece`` (end exclusive).

        Positions come straight from the tokenizer's character offsets of the window's first and
        last token; with ``strip_whitespace`` the start is moved past the stripped leading
        whitespace. Nothing is searched for.
        """
        if not text:
            return []
        input_ids, offsets = self._tokenize_with_offsets(text)
        n_tokens = len(input_ids)
        if n_tokens == 0:
            if not text.strip():
                return []
            if not self.strip_whitespace:
                return [(text, 0, len(text))]
            start = len(text) - len(text.lstrip())
            piece = text.strip()
            return [(piece, start, start + len(piece))]

        out: List[Tuple[str, int, int]] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start_token = 0

        while start_token < n_tokens:
            end_token = min(n_tokens, start_token + self.chunk_size)
            start_char = offsets[start_token][0]
            end_char = offsets[end_token - 1][1]
            piece = text[start_char:end_char]
            if self.strip_whitespace:
                stripped = piece.strip()
                if stripped:
                    start_char += len(piece) - len(piece.lstrip())
                piece = stripped
                end_char = start_char + len(piece)
            if piece:
                out.append((piece, start_char, end_char))
            if end_token == n_tokens:
                break
            start_token += step

        return out

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ):
        if Document is None:
            docs = []
            for i, t in enumerate(texts):
                md_base = (metadatas[i] if metadatas and i < len(metadatas) else {}) or {}
                chunks, starts = self._split_with_starts(t)
                for chunk, start in zip(chunks, starts):
                    md = dict(md_base)
                    if self.add_start_index:
                        md["start_index"] = start
                    docs.append({"page_content": chunk, "metadata": md})
            return docs

        docs: List[Document] = []
        for i, t in enumerate(texts):
            md_base = (metadatas[i] if metadatas and i < len(metadatas) else {}) or {}
            chunks, starts = self._split_with_starts(t)
            for chunk, start in zip(chunks, starts):
                md = dict(md_base)
                if self.add_start_index:
                    md["start_index"] = start
                docs.append(Document(page_content=chunk, metadata=md))
        return docs

    def _split_with_starts(self, text: str) -> Tuple[List[str], List[int]]:
        items = self.split_text_with_offsets(text)
        return [p for p, _, _ in items], [a for _, a, _ in items]


class FixedTokenChunker:
    """
    Token-based chunker using a Hugging Face tokenizer (no OpenAI/tiktoken).
    Params:
      - model_name: HF model id whose tokenizer will be used (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
      - chunk_size, chunk_overlap
      - add_start_index, strip_whitespace
    Methods:
      - split_text(text) -> List[str]
      - create_documents(texts, metadatas=None) -> List[Document or dict]
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be <= chunk_size ({chunk_size})"
            )

        self._splitter = _HFTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

        self._config = dict(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

    @classmethod
    def from_tiktoken_encoder(
        cls: Type[TS],
        *,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> TS:
        """
        Factory retained for compatibility in call sites, but now purely HF-based.
        Prefer passing your HF model name via model_name (e.g., 'sentence-transformers/all-MiniLM-L6-v2').
        """
        return cls(model_name=model_name, **kwargs)

    # --- Main API ---

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)

    def split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """[(chunk, start, end)] with ``text[start:end] == chunk``; positions are native, not searched."""
        return self._splitter.split_text_with_offsets(text)

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ):
        return self._splitter.create_documents(texts, metadatas)

    # Convenience passthroughs
    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def chunk_size(self) -> int:
        return self._config["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["chunk_overlap"]

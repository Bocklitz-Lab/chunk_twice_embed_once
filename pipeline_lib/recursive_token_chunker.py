# MIT License
# Wrapper to use LangChain's recursive splitter with token-based length (Hugging Face)

import logging
from typing import Any, List, Optional, Sequence, Tuple

try:
    # LangChain v0.2+ splitters package
    from langchain_text_splitters import RecursiveCharacterTextSplitter as LCRecursiveSplitter
except ImportError:
    # Fallback for older installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LCRecursiveSplitter

from .utils import Language

log = logging.getLogger(__name__)


class RecursiveTokenChunker:
    """
    A thin wrapper around LangChain's RecursiveCharacterTextSplitter configured
    to split based on **token length** using a Hugging Face tokenizer.

    Surface/API unchanged:

      - split_text(text) -> List[str]
      - create_documents(texts, metadatas=None) -> List[Document]

    Parameters:
      - chunk_size: int = 4000
      - chunk_overlap: int = 200
      - separators: Optional[List[str]] = None
      - keep_separator: bool = True
      - is_separator_regex: bool = False
      - **kwargs (supported):
          model_name: Optional[str] = None  # HF model id (e.g., "sentence-transformers/all-MiniLM-L6-v2")
          add_start_index: bool = False
          strip_whitespace: bool = True
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be <= chunk_size ({chunk_size})"
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        # Explicitly reject removed/unsupported kwargs
        for removed in ("encoding_name", "allowed_special", "disallowed_special"):
            if removed in kwargs:
                raise ValueError(f"Unsupported argument: {removed}. This class no longer accepts it.")

        # Supported kwargs
        model_name = kwargs.pop("model_name", None)
        add_start_index = kwargs.pop("add_start_index", False)
        strip_whitespace = kwargs.pop("strip_whitespace", True)

        # Warn on any unknown kwargs to avoid silent misconfigurations
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(f"Unknown argument(s): {unknown}")

        # Default separators (match your original order) if none provided
        if separators is None:
            separators = ["\n\n", "\n", ".", "?", "!", " ", ""]

        # Build a length_function using a Hugging Face tokenizer
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise ImportError(
                "transformers is required for Hugging Face tokenization. "
                "Install with `pip install transformers`."
            ) from e

        if not model_name:
            # No silent default (it used to be bert-base-uncased): chunk size is measured with the
            # tokenizer of the embedding model that will encode the chunks.
            raise ValueError("RecursiveTokenChunker needs model_name (the HF tokenizer of the embedding model).")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            raise ValueError(
                f"Failed to load Hugging Face tokenizer for model_name='{model_name}'."
            ) from e

        self._tokenizer = tokenizer
        self.n_ambiguous = 0   # chunks whose text matched more than once in the allowed window
        self.n_relaxed = 0     # chunks that needed the wider fallback search (see _locate)

        def _length_fn(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        self._length_fn = _length_fn

        self._splitter = LCRecursiveSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_length_fn,
            separators=separators,
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

        self._config = dict(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators),
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
            model_name=model_name,   # HF model id
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

    # --- Main API ---

    def split_text(self, text: str) -> List[str]:
        """Token-length-based recursive split using a Hugging Face tokenizer."""
        return self._splitter.split_text(text)

    def split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """[(chunk, start, end)] with ``text[start:end] == chunk`` (end exclusive).

        LangChain returns strings only. Each chunk is therefore located by a search that is
        confined to where it can possibly be (see ``_locate``), never a plain forward ``find``.
        """
        return self._locate(text, self._splitter.split_text(text))

    def _max_overlap_chars(self, prev_text: str) -> int:
        """Upper bound (in characters) for how much of ``prev_text`` the next chunk can repeat.

        The overlap is a suffix of the previous chunk of at most ``chunk_overlap`` tokens.
        At overlap 0 there is none. A small slack covers differences between summing the
        token counts of separator pieces (LangChain) and tokenising the joined text.
        """
        overlap = self._config["chunk_overlap"]
        if overlap <= 0 or not prev_text:
            return 0
        k = overlap + 2 + overlap // 8
        enc = self._tokenizer(
            prev_text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )
        offs = enc["offset_mapping"]
        if len(offs) <= k:
            return len(prev_text)
        return len(prev_text) - offs[len(offs) - k][0]

    def _locate(self, text: str, chunks: List[str]) -> List[Tuple[str, int, int]]:
        """Exact positions of LangChain's chunks inside ``text``.

        Chunk i starts at or after ``prev_end - max_overlap_chars`` and (apart from
        whitespace that is stripped) not after ``prev_end``, and it ends after the previous chunk
        (each chunk adds new text). Candidates are the matches of its text in that window. LangChain never repeats more than ``chunk_overlap`` tokens, so
        candidates whose overlap with the previous chunk is longer are discarded, and the first
        remaining match is taken. A text that still matches more than once is counted in
        ``n_ambiguous`` and logged (first match wins). If the window finds nothing (bound too
        tight) the search is widened to "after the previous chunk's start" and counted in
        ``n_relaxed``.
        """
        overlap = self._config["chunk_overlap"]
        out: List[Tuple[str, int, int]] = []
        prev_start, prev_end, prev_text = -1, 0, ""
        for ch in chunks:
            if not ch:
                continue
            lo = max(prev_start + 1, prev_end - self._max_overlap_chars(prev_text))
            hi = prev_end
            while hi < len(text) and text[hi].isspace():
                hi += 1          # whitespace between chunks was stripped
            cands: List[int] = []
            p = text.find(ch, lo)
            while p != -1 and p <= hi:
                if p + len(ch) > prev_end:   # every chunk adds new text: its end moves forward
                    cands.append(p)
                p = text.find(ch, p + 1)

            if not cands:
                idx = text.find(ch, prev_start + 1)
                if idx == -1:
                    raise ValueError(
                        "RecursiveTokenChunker: chunk is not a substring of the text after the "
                        f"previous chunk (prev_start={prev_start}); chunk starts {ch[:60]!r}"
                    )
                self.n_relaxed += 1
                log.warning("recursive chunk located outside the expected window (start=%d, window=[%d,%d])", idx, lo, hi)
            else:
                pool = [c for c in cands
                        if c >= prev_end or self._length_fn(text[c:prev_end]) <= overlap] or cands
                idx = pool[0]
                if len(pool) > 1:
                    self.n_ambiguous += 1
                    log.warning("recursive chunk text occurs %d times in its window [%d,%d]; taking the first (start=%d)",
                                len(pool), lo, hi, idx)
            out.append((ch, idx, idx + len(ch)))
            prev_start, prev_end, prev_text = idx, idx + len(ch), ch
        return out

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ):
        """Create LangChain Documents (with 'start_index' if add_start_index=True)."""
        return self._splitter.create_documents(texts, metadatas)

    # --- Helpers ---

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        return LCRecursiveSplitter.get_separators_for_language(language)

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def chunk_size(self) -> int:
        return self._config["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["chunk_overlap"]

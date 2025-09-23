# MIT License
# Wrapper to use LangChain's recursive splitter with token-based length (Hugging Face)

from typing import Any, List, Optional, Sequence

try:
    # LangChain v0.2+ splitters package
    from langchain_text_splitters import RecursiveCharacterTextSplitter as LCRecursiveSplitter
except ImportError:
    # Fallback for older installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LCRecursiveSplitter

from .utils import Language


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

        if model_name is None:
            model_name = "bert-base-uncased"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            raise ValueError(
                f"Failed to load Hugging Face tokenizer for model_name='{model_name}'."
            ) from e

        def _length_fn(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

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

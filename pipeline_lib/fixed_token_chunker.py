from typing import Any, AbstractSet, Collection, Iterable, List, Literal, Optional, Sequence, Union, Type, TypeVar

# LangChain imports (support both v0.1+ and v0.2+ package names)
try:
    from langchain_text_splitters import TokenTextSplitter as LCTokenTextSplitter
except ImportError:
    # Fallback for older installs
    from langchain.text_splitter import TokenTextSplitter as LCTokenTextSplitter


TS = TypeVar("TS", bound="FixedTokenChunker")


class FixedTokenChunker:
    """
    A thin wrapper around LangChain's TokenTextSplitter that keeps (roughly) the same
    constructor surface and behavior as in your example:
      - Token-based chunks using tiktoken encoder
      - Supports chunk_size, chunk_overlap
      - Supports encoding_name/model_name
      - Supports allowed_special/disallowed_special
      - Supports add_start_index (via LangChain's TextSplitter base)
      - Supports strip_whitespace
      - keep_separator is accepted for API compatibility (no-op in token mode)

    Methods:
      - split_text(text) -> List[str]
      - create_documents(texts, metadatas=None) -> List[Document]
        (includes 'start_index' in metadata if add_start_index=True)
    """

    def __init__(
        self,
        *,
        # tokenization controls
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        # chunking controls
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        # formatting / metadata
        keep_separator: bool = False,     # accepted for compatibility; not used by token splitter
        add_start_index: bool = False,    # handled by LC TextSplitter
        strip_whitespace: bool = True,    # handled by LC TextSplitter
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be <= chunk_size ({chunk_size})"
            )

        # Note: TokenTextSplitter does not use keep_separator (char-splitting concern).
        # We accept it to mirror the original API.
        self.keep_separator = keep_separator

        # Instantiate the LC token-based splitter
        self._splitter = LCTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

        # Expose config if you want to introspect later
        self._config = dict(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

    @classmethod
    def from_tiktoken_encoder(
        cls: Type[TS],
        *,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> TS:
        """
        Factory for parity with your original API.
        All kwargs are forwarded (e.g., chunk_size, chunk_overlap, add_start_index, strip_whitespace).
        """
        return cls(
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            **kwargs,
        )

    # --- Main API ---

    def split_text(self, text: str) -> List[str]:
        """Token-based split using LangChain's TokenTextSplitter."""
        return self._splitter.split_text(text)

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ):
        """
        Wraps LangChain's create_documents:
        - If add_start_index=True, each resulting Document's metadata includes 'start_index'
          (the character start of the chunk in the original string).
        - strip_whitespace is applied if enabled.
        """
        return self._splitter.create_documents(texts, metadatas)

    # Convenience passthroughs (optional)
    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def chunk_size(self) -> int:
        return self._config["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["chunk_overlap"]




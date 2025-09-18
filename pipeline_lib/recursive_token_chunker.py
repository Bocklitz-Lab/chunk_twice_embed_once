
# # This script is adapted from the LangChain package, developed by LangChain AI.
# # Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
# # License: MIT License

# from typing import Any, List, Optional
# from .base_chunker import BaseChunker
# from .utils import Language
# from .fixed_token_chunker import TextSplitter
# import re

# def _split_text_with_regex(
#     text: str, separator: str, keep_separator: bool
# ) -> List[str]:
#     # Now that we have the separator, split the text
#     if separator:
#         if keep_separator:
#             # The parentheses in the pattern keep the delimiters in the result.
#             _splits = re.split(f"({separator})", text)
#             splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
#             if len(_splits) % 2 == 0:
#                 splits += _splits[-1:]
#             splits = [_splits[0]] + splits
#         else:
#             splits = re.split(separator, text)
#     else:
#         splits = list(text)
#     return [s for s in splits if s != ""]

# class RecursiveTokenChunker(TextSplitter):
#     """Splitting text by recursively look at characters.

#     Recursively tries to split by different characters to find one
#     that works.
#     """

#     def __init__(
#         self,
#         chunk_size: int = 4000,
#         chunk_overlap: int = 200,
#         separators: Optional[List[str]] = None,
#         keep_separator: bool = True,
#         is_separator_regex: bool = False,
#         **kwargs: Any,
#     ) -> None:
#         """Create a new TextSplitter."""
#         super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator, **kwargs)
#         self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
#         self._is_separator_regex = is_separator_regex

#     def _split_text(self, text: str, separators: List[str]) -> List[str]:
#         """Split incoming text and return chunks."""
#         final_chunks = []
#         # Get appropriate separator to use
#         separator = separators[-1]
#         new_separators = []
#         for i, _s in enumerate(separators):
#             _separator = _s if self._is_separator_regex else re.escape(_s)
#             if _s == "":
#                 separator = _s
#                 break
#             if re.search(_separator, text):
#                 separator = _s
#                 new_separators = separators[i + 1 :]
#                 break

#         _separator = separator if self._is_separator_regex else re.escape(separator)
#         splits = _split_text_with_regex(text, _separator, self._keep_separator)

#         # Now go merging things, recursively splitting longer texts.
#         _good_splits = []
#         _separator = "" if self._keep_separator else separator
#         for s in splits:
#             if self._length_function(s) < self._chunk_size:
#                 _good_splits.append(s)
#             else:
#                 if _good_splits:
#                     merged_text = self._merge_splits(_good_splits, _separator)
#                     final_chunks.extend(merged_text)
#                     _good_splits = []
#                 if not new_separators:
#                     final_chunks.append(s)
#                 else:
#                     other_info = self._split_text(s, new_separators)
#                     final_chunks.extend(other_info)
#         if _good_splits:
#             merged_text = self._merge_splits(_good_splits, _separator)
#             final_chunks.extend(merged_text)
#         return final_chunks

#     def split_text(self, text: str) -> List[str]:
#         return self._split_text(text, self._separators)

#     # @classmethod
#     # def from_language(
#     #     cls, language: Language, **kwargs: Any
#     # ) -> RecursiveCharacterTextSplitter:
#     #     separators = cls.get_separators_for_language(language)
#     #     return cls(separators=separators, is_separator_regex=True, **kwargs)

#     @staticmethod
#     def get_separators_for_language(language: Language) -> List[str]:
#         if language == Language.CPP:
#             return [
#                 # Split along class definitions
#                 "\nclass ",
#                 # Split along function definitions
#                 "\nvoid ",
#                 "\nint ",
#                 "\nfloat ",
#                 "\ndouble ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\nswitch ",
#                 "\ncase ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.GO:
#             return [
#                 # Split along function definitions
#                 "\nfunc ",
#                 "\nvar ",
#                 "\nconst ",
#                 "\ntype ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nswitch ",
#                 "\ncase ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.JAVA:
#             return [
#                 # Split along class definitions
#                 "\nclass ",
#                 # Split along method definitions
#                 "\npublic ",
#                 "\nprotected ",
#                 "\nprivate ",
#                 "\nstatic ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\nswitch ",
#                 "\ncase ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.KOTLIN:
#             return [
#                 # Split along class definitions
#                 "\nclass ",
#                 # Split along method definitions
#                 "\npublic ",
#                 "\nprotected ",
#                 "\nprivate ",
#                 "\ninternal ",
#                 "\ncompanion ",
#                 "\nfun ",
#                 "\nval ",
#                 "\nvar ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\nwhen ",
#                 "\ncase ",
#                 "\nelse ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.JS:
#             return [
#                 # Split along function definitions
#                 "\nfunction ",
#                 "\nconst ",
#                 "\nlet ",
#                 "\nvar ",
#                 "\nclass ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\nswitch ",
#                 "\ncase ",
#                 "\ndefault ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.TS:
#             return [
#                 "\nenum ",
#                 "\ninterface ",
#                 "\nnamespace ",
#                 "\ntype ",
#                 # Split along class definitions
#                 "\nclass ",
#                 # Split along function definitions
#                 "\nfunction ",
#                 "\nconst ",
#                 "\nlet ",
#                 "\nvar ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\nswitch ",
#                 "\ncase ",
#                 "\ndefault ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.PHP:
#             return [
#                 # Split along function definitions
#                 "\nfunction ",
#                 # Split along class definitions
#                 "\nclass ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nforeach ",
#                 "\nwhile ",
#                 "\ndo ",
#                 "\nswitch ",
#                 "\ncase ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.PROTO:
#             return [
#                 # Split along message definitions
#                 "\nmessage ",
#                 # Split along service definitions
#                 "\nservice ",
#                 # Split along enum definitions
#                 "\nenum ",
#                 # Split along option definitions
#                 "\noption ",
#                 # Split along import statements
#                 "\nimport ",
#                 # Split along syntax declarations
#                 "\nsyntax ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.PYTHON:
#             return [
#                 # First, try to split along class definitions
#                 "\nclass ",
#                 "\ndef ",
#                 "\n\tdef ",
#                 # Now split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.RST:
#             return [
#                 # Split along section titles
#                 "\n=+\n",
#                 "\n-+\n",
#                 "\n\\*+\n",
#                 # Split along directive markers
#                 "\n\n.. *\n\n",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.RUBY:
#             return [
#                 # Split along method definitions
#                 "\ndef ",
#                 "\nclass ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nunless ",
#                 "\nwhile ",
#                 "\nfor ",
#                 "\ndo ",
#                 "\nbegin ",
#                 "\nrescue ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.RUST:
#             return [
#                 # Split along function definitions
#                 "\nfn ",
#                 "\nconst ",
#                 "\nlet ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nwhile ",
#                 "\nfor ",
#                 "\nloop ",
#                 "\nmatch ",
#                 "\nconst ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.SCALA:
#             return [
#                 # Split along class definitions
#                 "\nclass ",
#                 "\nobject ",
#                 # Split along method definitions
#                 "\ndef ",
#                 "\nval ",
#                 "\nvar ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\nmatch ",
#                 "\ncase ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.SWIFT:
#             return [
#                 # Split along function definitions
#                 "\nfunc ",
#                 # Split along class definitions
#                 "\nclass ",
#                 "\nstruct ",
#                 "\nenum ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\ndo ",
#                 "\nswitch ",
#                 "\ncase ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.MARKDOWN:
#             return [
#                 # First, try to split along Markdown headings (starting with level 2)
#                 "\n#{1,6} ",
#                 # Note the alternative syntax for headings (below) is not handled here
#                 # Heading level 2
#                 # ---------------
#                 # End of code block
#                 "```\n",
#                 # Horizontal lines
#                 "\n\\*\\*\\*+\n",
#                 "\n---+\n",
#                 "\n___+\n",
#                 # Note that this splitter doesn't handle horizontal lines defined
#                 # by *three or more* of ***, ---, or ___, but this is not handled
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.LATEX:
#             return [
#                 # First, try to split along Latex sections
#                 "\n\\\\chapter{",
#                 "\n\\\\section{",
#                 "\n\\\\subsection{",
#                 "\n\\\\subsubsection{",
#                 # Now split by environments
#                 "\n\\\\begin{enumerate}",
#                 "\n\\\\begin{itemize}",
#                 "\n\\\\begin{description}",
#                 "\n\\\\begin{list}",
#                 "\n\\\\begin{quote}",
#                 "\n\\\\begin{quotation}",
#                 "\n\\\\begin{verse}",
#                 "\n\\\\begin{verbatim}",
#                 # Now split by math environments
#                 "\n\\\begin{align}",
#                 "$$",
#                 "$",
#                 # Now split by the normal type of lines
#                 " ",
#                 "",
#             ]
#         elif language == Language.HTML:
#             return [
#                 # First, try to split along HTML tags
#                 "<body",
#                 "<div",
#                 "<p",
#                 "<br",
#                 "<li",
#                 "<h1",
#                 "<h2",
#                 "<h3",
#                 "<h4",
#                 "<h5",
#                 "<h6",
#                 "<span",
#                 "<table",
#                 "<tr",
#                 "<td",
#                 "<th",
#                 "<ul",
#                 "<ol",
#                 "<header",
#                 "<footer",
#                 "<nav",
#                 # Head
#                 "<head",
#                 "<style",
#                 "<script",
#                 "<meta",
#                 "<title",
#                 "",
#             ]
#         elif language == Language.CSHARP:
#             return [
#                 "\ninterface ",
#                 "\nenum ",
#                 "\nimplements ",
#                 "\ndelegate ",
#                 "\nevent ",
#                 # Split along class definitions
#                 "\nclass ",
#                 "\nabstract ",
#                 # Split along method definitions
#                 "\npublic ",
#                 "\nprotected ",
#                 "\nprivate ",
#                 "\nstatic ",
#                 "\nreturn ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\ncontinue ",
#                 "\nfor ",
#                 "\nforeach ",
#                 "\nwhile ",
#                 "\nswitch ",
#                 "\nbreak ",
#                 "\ncase ",
#                 "\nelse ",
#                 # Split by exceptions
#                 "\ntry ",
#                 "\nthrow ",
#                 "\nfinally ",
#                 "\ncatch ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.SOL:
#             return [
#                 # Split along compiler information definitions
#                 "\npragma ",
#                 "\nusing ",
#                 # Split along contract definitions
#                 "\ncontract ",
#                 "\ninterface ",
#                 "\nlibrary ",
#                 # Split along method definitions
#                 "\nconstructor ",
#                 "\ntype ",
#                 "\nfunction ",
#                 "\nevent ",
#                 "\nmodifier ",
#                 "\nerror ",
#                 "\nstruct ",
#                 "\nenum ",
#                 # Split along control flow statements
#                 "\nif ",
#                 "\nfor ",
#                 "\nwhile ",
#                 "\ndo while ",
#                 "\nassembly ",
#                 # Split by the normal type of lines
#                 "\n\n",
#                 "\n",
#                 " ",
#                 "",
#             ]
#         elif language == Language.COBOL:
#             return [
#                 # Split along divisions
#                 "\nIDENTIFICATION DIVISION.",
#                 "\nENVIRONMENT DIVISION.",
#                 "\nDATA DIVISION.",
#                 "\nPROCEDURE DIVISION.",
#                 # Split along sections within DATA DIVISION
#                 "\nWORKING-STORAGE SECTION.",
#                 "\nLINKAGE SECTION.",
#                 "\nFILE SECTION.",
#                 # Split along sections within PROCEDURE DIVISION
#                 "\nINPUT-OUTPUT SECTION.",
#                 # Split along paragraphs and common statements
#                 "\nOPEN ",
#                 "\nCLOSE ",
#                 "\nREAD ",
#                 "\nWRITE ",
#                 "\nIF ",
#                 "\nELSE ",
#                 "\nMOVE ",
#                 "\nPERFORM ",
#                 "\nUNTIL ",
#                 "\nVARYING ",
#                 "\nACCEPT ",
#                 "\nDISPLAY ",
#                 "\nSTOP RUN.",
#                 # Split by the normal type of lines
#                 "\n",
#                 " ",
#                 "",
#             ]

#         else:
#             raise ValueError(
#                 f"Language {language} is not supported! "
#                 f"Please choose from {list(Language)}"
#             )







































# MIT License
# Wrapper to use LangChain's recursive splitter with token-based length

from typing import Any, List, Optional, Sequence

try:
    # LangChain v0.2+ splitters package
    from langchain_text_splitters import RecursiveCharacterTextSplitter as LCRecursiveSplitter
except ImportError:
    # Fallback for older installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LCRecursiveSplitter

# Keep your local imports so the surface matches your existing code
from .utils import Language


class RecursiveTokenChunker:
    """
    A thin wrapper around LangChain's RecursiveCharacterTextSplitter configured
    to split based on **token length** (tiktoken) rather than character length.

    Constructor matches the original API you provided, and methods return the same types:

      - split_text(text) -> List[str]
      - create_documents(texts, metadatas=None) -> List[Document]  (optional passthrough)

    Parameters (same surface as your previous class):
      - chunk_size: int = 4000
      - chunk_overlap: int = 200
      - separators: Optional[List[str]] = None
      - keep_separator: bool = True
      - is_separator_regex: bool = False
      - **kwargs: tokenization and formatting options forwarded to LC:
          encoding_name: str = "cl100k_base"
          model_name: Optional[str] = None
          allowed_special: Union["all", AbstractSet[str]] = set()
          disallowed_special: Union["all", Collection[str]] = "all"
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

        # Extract tokenization / formatting kwargs with sensible defaults
        encoding_name = kwargs.pop("encoding_name", "cl100k_base")
        model_name = kwargs.pop("model_name", None)
        allowed_special = kwargs.pop("allowed_special", set())
        disallowed_special = kwargs.pop("disallowed_special", "all")
        add_start_index = kwargs.pop("add_start_index", False)
        strip_whitespace = kwargs.pop("strip_whitespace", True)

        # Default separators (match your original order) if none provided
        if separators is None:
            separators = ["\n\n", "\n", ".", "?", "!", " ", ""]

        # Build a LangChain recursive splitter that counts tokens
        # NOTE: from_tiktoken_encoder configures length_function to token counting.
        self._splitter = LCRecursiveSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

        # Stash config for introspection/debugging
        self._config = dict(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators),
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
            encoding_name=encoding_name,
            model_name=model_name,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )

    # --- Main API ---

    def split_text(self, text: str) -> List[str]:
        """Token-length-based recursive split using LangChain."""
        return self._splitter.split_text(text)

    # Optional passthrough to match common usage in your codebase
    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ):
        """Create LangChain Documents (with 'start_index' if add_start_index=True)."""
        return self._splitter.create_documents(texts, metadatas)

    # --- Compatibility helpers / parity with your previous class ---

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        """
        Delegate to LangChain's built-in language-aware separators to avoid
        duplicating a large mapping here.
        """
        return LCRecursiveSplitter.get_separators_for_language(language)

    # A couple of convenience properties for parity
    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def chunk_size(self) -> int:
        return self._config["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._config["chunk_overlap"]

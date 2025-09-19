# section_hier_chunker.py
from __future__ import annotations
from typing import Any, AbstractSet, Collection, Iterable, List, Literal, Optional, Sequence, Tuple, Type, Union
import re

# LangChain imports (support both v0.1+ and v0.2+ package names)
try:
    from langchain_text_splitters import TokenTextSplitter as LCTokenTextSplitter
except ImportError:
    from langchain.text_splitter import TokenTextSplitter as LCTokenTextSplitter

try:
    import tiktoken
except Exception:
    tiktoken = None


class HierarchicalSectionChunker:
    """
    Section-aware hierarchical chunker for scientific papers.

    Strategy:
      1) Detect section/subsection boundaries using:
         - LaTeX commands: \\section{...}, \\subsection{...}, \\subsubsection{...}
         - Numbered headings: '1 Introduction', '2.1 Experimental', etc.
         - Common chemistry headers: Experimental, Materials and Methods, Results, Results and Discussion, ...
         - (Optional) generic heading heuristics (short line, no trailing period, title case or ALLCAPS, blank line around).
      2) Build contiguous section spans (heading + content until next heading).
      3) Optionally merge short adjacent sections (< min_section_tokens) to avoid tiny orphan chunks.
      4) Within each (possibly merged) section, do token-based chunking with overlap via LangChain's TokenTextSplitter.

    Notes:
      - Chunks are substrings of the original text (we keep the original heading text) so your
        locate_chunk_offsets() will align well.
      - We include the heading line only once at the start of the section (no synthetic prefixes).
    """

    def __init__(
        self,
        *,
        # tokenization controls (passed to TokenTextSplitter)
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        keep_separator: bool = False,  # accepted for parity; not used by token splitter

        # section detection controls
        use_latex: bool = True,
        use_numbered_headings: bool = True,
        use_common_chem_headings: bool = True,
        use_generic_heading_heuristics: bool = True,

        # list of regexes (strings) for custom headings; evaluated case-insensitively on full lines
        custom_heading_regexes: Optional[List[str]] = None,

        # merging small sections
        join_short_sections: bool = True,
        min_section_tokens: int = 200,       # sections shorter than this will be merged forward/backward
        prefer_merge_forward: bool = True,   # merge with next section if possible

        # option to drop trailing sections like references if desired
        drop_tail_sections_matching: Optional[List[str]] = None,  # e.g., ["references", "bibliography", "acknowledg"]
        drop_from_first_match: bool = False,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be <= chunk_size ({chunk_size})"
            )

        self.keep_separator = keep_separator

        # splitter used *inside* sections
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

        # save config
        self._config = dict(
            encoding_name=encoding_name,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
            keep_separator=keep_separator,
            use_latex=use_latex,
            use_numbered_headings=use_numbered_headings,
            use_common_chem_headings=use_common_chem_headings,
            use_generic_heading_heuristics=use_generic_heading_heuristics,
            join_short_sections=join_short_sections,
            min_section_tokens=min_section_tokens,
            prefer_merge_forward=prefer_merge_forward,
            drop_tail_sections_matching=drop_tail_sections_matching or [],
            drop_from_first_match=drop_from_first_match, 
        )

        # prepare token encoder for counting sections (optional)
        self._encoder = None
        if tiktoken is not None:
            try:
                if model_name:
                    self._encoder = tiktoken.encoding_for_model(model_name)
                else:
                    self._encoder = tiktoken.get_encoding(encoding_name or "cl100k_base")
            except Exception:
                try:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._encoder = None

        # compile heading detectors
        self._latex_re = re.compile(
            r"""\\(?P<lvl>section|subsection|subsubsection)\*?\{(?P<title>[^}]+)\}""",
            flags=re.IGNORECASE
        )

        # numbered headings like "1", "1.2", "1.2.3", optionally followed by '-', ')' or '.'
        num_prefix = r"""(?:(?:\d+)(?:\.\d+){0,3}\s*[-\).]?\s*)"""

        # common chemistry headings (very useful in this domain)
        chem_heads = [
            # Front matter
            r"abstract",
            r"graphical\s+abstract",
            r"highlights",
            r"keywords?",

            # Introduction / background
            r"introduction",
            r"background",
            r"literature\s+review",
            r"related\s+work",
            r"theory",
            r"theoretical\s+(background|methods|framework|analysis|considerations?)",
            r"computational\s+details?",
            r"model(\s+systems?)?",

            # Methods / experimental
            r"(materials\s+and\s+methods)",
            r"methods?",
            r"methodology",
            r"experimental(\s+(section|details|setup|design|procedures?))?",
            r"experimental\s+methods?",
            r"experimental\s+section",
            r"experimental\s+procedures?",
            r"procedures?",
            r"apparatus",
            r"instruments?",
            r"instrumentation",
            r"chemicals\s+and\s+reagents?",
            r"data\s+collection",
            r"sample\s+preparation",
            r"synthesis",
            r"fabrication",
            r"characterization",
            r"measurements?",

            # Results / findings
            r"results",
            r"observations?",
            r"findings?",
            r"experimental\s+results",
            r"numerical\s+results",
            r"simulation\s+results",
            r"case\s+study",
            r"case\s+studies",
            r"examples?",

            # Analysis / discussion
            r"discussion",
            r"analysis",
            r"results\s+and\s+discussion",
            r"discussion\s+and\s+conclusions?",
            r"modeling\s+and\s+analysis",
            r"interpretation",

            # Conclusions / outlook
            r"conclusion(s)?",
            r"summary",
            r"general\s+discussion",
            r"outlook",
            r"future\s+work",
            r"perspectives?",

            # Supporting sections
            r"supporting\s+(information|materials)",
            r"supplementary\s+(information|materials|data|figures|tables)?",
            r"appendix",
            r"appendices",
            r"annex",
            r"additional\s+(information|materials|files|data)?",

            # Acknowledgments / ethics
            r"acknowledg(e)?ments?",
            r"conflicts?\s+of\s+interest",
            r"declarations?",
            r"funding",
            r"author\s+contributions?",
            r"contribut(ion|ions)\s+of\s+authors?",
            r"ethics",
            r"ethical\s+approval",
            r"data\s+availability",

            # References / end matter
            r"references?",
            r"bibliograph(y|ies)",
            r"works\s+cited",
            r"citations?",
            r"further\s+reading",

            # Journal-specific extras
            r"nomenclature",
            r"abbreviations",
            r"glossary",
            r"list\s+of\s+figures",
            r"list\s+of\s+tables",
            r"toc\s+graphic",
            r"table\s+of\s+contents\s+graphic",
        ]

        chem_body = r"(?:%s)" % "|".join(chem_heads)
        self._chem_re = re.compile(
            rf"^\s*(?:{num_prefix})?{chem_body}\s*:?\s*$",
            flags=re.IGNORECASE | re.MULTILINE
        )

        # numbered headings (generic, not only chemistry)
        self._numbered_re = re.compile(
            rf"^\s*(?:{num_prefix})?[A-Z][^\n]{{0,120}}\s*$",
            flags=re.MULTILINE
        )

        # heuristics: short, no trailing period, looks like a title, surrounded by blank line(s)
        self._generic_re = re.compile(
            r"^[^\n\.]{1,120}$"  # one line, short, no trailing period
        )

        # user-specified custom regexes
        self._custom_res = [
            re.compile(rx, flags=re.IGNORECASE | re.MULTILINE) for rx in (custom_heading_regexes or [])
        ]

    # ---------- Public API ----------

    def split_text(self, text: str) -> List[str]:
        text = text or ""
        if not text.strip():
            return []

        # Step 1: find candidate section starts (list of (start_idx, end_idx) ranges covering the heading line/command)
        headings = self._find_headings(text)

        # If nothing detected, just token-chunk the whole text
        if not headings:
            return self._splitter.split_text(text)

        # Remove duplicates / sort
        headings = self._dedupe_and_sort(headings)

        # Step 2: build contiguous sections: [ (start_idx, next_start_idx), ... ]
        spans = self._build_section_spans(text, headings)

        # Optionally drop tail sections like References, Bibliography, etc.
        drop_labels = self._config["drop_tail_sections_matching"]
        if drop_labels:
            spans = self._drop_tail_sections(text, spans, drop_labels)

        # Step 3: merge tiny sections if requested
        if self._config["join_short_sections"]:
            spans = self._merge_short_sections(text, spans, self._config["min_section_tokens"])

        # Step 4: inside each section, run token splitter
        chunks: List[str] = []
        for (s, e) in spans:
            section_text = text[s:e]
            # ensure we don't accidentally pass empty strings
            if section_text.strip():
                chunks.extend(self._splitter.split_text(section_text))

        return chunks

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ):
        # For parity with LC splitter: delegate at paragraph granularity
        docs = []
        metadatas = metadatas or [{} for _ in texts]
        for t, m in zip(texts, metadatas):
            for chunk in self.split_text(t):
                docs.extend(self._splitter.create_documents([chunk], [m]))
        return docs

    # ---------- Internals ----------

    def _find_headings(self, text: str) -> List[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = []

        # 1) LaTeX \section{...}
        if self._config["use_latex"]:
            for m in self._latex_re.finditer(text):
                candidates.append((m.start(), m.end()))

        # 2) Line-by-line checks
        line_starts = self._line_starts(text)
        lines = text.splitlines()

        def line_span(i: int) -> Tuple[int, int]:
            start = line_starts[i]
            # end at start of next line or end of text
            end = line_starts[i + 1] if i + 1 < len(line_starts) else len(text)
            # strip trailing '\r' if present
            while end > start and text[end - 1] in ("\n", "\r"):
                end -= 1
            return (start, end)

        # helper: check blank prev/next lines
        def is_blank_line(line: str) -> bool:
            return len(line.strip()) == 0

        for i, line in enumerate(lines):
            start_i, end_i = line_span(i)
            l = line.strip()

            if not l:
                continue

            matched = False

            # custom regexes (full text, but they are compiled as MULTILINE so they can match this line)
            for rx in self._custom_res:
                if rx.match(line):
                    candidates.append((start_i, end_i))
                    matched = True
                    break
            if matched:
                continue

            # common chemistry headings
            if self._config["use_common_chem_headings"]:
                if self._chem_re.match(line):
                    candidates.append((start_i, end_i))
                    continue

            # numbered headings (generic)
            if self._config["use_numbered_headings"]:
                # Must be reasonably short, and often preceded or followed by a blank line
                if self._numbered_re.match(line):
                    # apply a small heuristic to avoid catching normal sentences:
                    #  - either previous or next line is blank
                    prev_blank = is_blank_line(lines[i - 1]) if i > 0 else True
                    next_blank = is_blank_line(lines[i + 1]) if i + 1 < len(lines) else True
                    if prev_blank or next_blank:
                        candidates.append((start_i, end_i))
                        continue

            # generic heuristics
            if self._config["use_generic_heading_heuristics"]:
                # short, no final period, surrounded by blank line(s), looks like a title
                if self._generic_re.match(l):
                    prev_blank = is_blank_line(lines[i - 1]) if i > 0 else True
                    next_blank = is_blank_line(lines[i + 1]) if i + 1 < len(lines) else True
                    if prev_blank or next_blank:
                        # Title-ish if either many capitals or Most Words TitleCased
                        if self._looks_like_title(l):
                            candidates.append((start_i, end_i))
                            continue

        return candidates

    def _dedupe_and_sort(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        spans = sorted(spans, key=lambda x: x[0])
        deduped: List[Tuple[int, int]] = []
        last_end = -1
        for s, e in spans:
            if s >= last_end:
                deduped.append((s, e))
                last_end = e
            else:
                # overlapping headings: keep the earliest start / longest end
                if deduped and s == deduped[-1][0] and e > deduped[-1][1]:
                    deduped[-1] = (s, e)
                    last_end = e
        return deduped

    def _build_section_spans(self, text: str, headings: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Cover entire doc: from first heading to next, etc. If the first heading isn't at pos 0,
        # also include the preamble as a section (e.g., title/author/abstract when not marked)
        spans: List[Tuple[int, int]] = []
        n = len(text)

        if headings and headings[0][0] > 0:
            spans.append((0, headings[0][0]))

        for i, (hs, he) in enumerate(headings):
            start = hs
            end = headings[i + 1][0] if i + 1 < len(headings) else n
            if start < end:
                spans.append((start, end))

        # If no headings, fallback handled before.
        return spans

    def _drop_tail_sections(
        self,
        text: str,
        spans: List[Tuple[int, int]],
        labels: List[str],
    ) -> List[Tuple[int, int]]:
        """
        Drop tail sections based on heading labels.

        Modes:
        - Default: remove only the contiguous trailing run of matches.
        - Aggressive (drop_from_first_match=True in config): remove everything
            from the first match (scanning backward) to the end.
        """
        if not spans:
            return spans

        lowered_labels = [lbl.lower() for lbl in labels]
        drop_from_first_match = bool(self._config.get("drop_from_first_match", False))

        def is_tail(span: Tuple[int, int]) -> bool:
            s, e = span
            # extract the first line of this span
            first_newline = text.find("\n", s, e)
            line = text[s:e] if first_newline == -1 else text[s:first_newline]
            l = line.strip().lower()
            return any(lbl in l for lbl in lowered_labels)

        if drop_from_first_match:
            # find the *last* match scanning backwards and drop from there
            for i in range(len(spans) - 1, -1, -1):
                if is_tail(spans[i]):
                    return spans[:i]  # cut everything from here onward
            return spans

        # default behavior: drop only trailing run of matches
        end_idx = len(spans)
        while end_idx > 0 and is_tail(spans[end_idx - 1]):
            end_idx -= 1
        return spans[:end_idx]


    def _merge_short_sections(
        self, text: str, spans: List[Tuple[int, int]], min_tokens: int
    ) -> List[Tuple[int, int]]:
        if not spans or min_tokens <= 0:
            return spans

        counts = [self._count_tokens(text[s:e]) for (s, e) in spans]
        merged: List[Tuple[int, int]] = []
        i = 0
        n = len(spans)

        while i < n:
            s, e = spans[i]
            tok = counts[i]
            if tok >= min_tokens or i == n - 1:
                merged.append((s, e))
                i += 1
                continue

            # too short; try to merge with neighbor
            if self._config["prefer_merge_forward"] and i + 1 < n:
                # merge with next
                ns, ne = spans[i + 1]
                merged.append((s, ne))
                i += 2
            elif not self._config["prefer_merge_forward"] and merged:
                # merge with previous
                ps, pe = merged.pop()
                merged.append((ps, e))
                i += 1
            else:
                # fallback: merge forward if possible
                if i + 1 < n:
                    ns, ne = spans[i + 1]
                    merged.append((s, ne))
                    i += 2
                else:
                    merged.append((s, e))
                    i += 1

        return merged

    def _line_starts(self, text: str) -> List[int]:
        # positions in text where each line starts
        starts = [0]
        idx = 0
        while True:
            j = text.find("\n", idx)
            if j == -1:
                break
            starts.append(j + 1)
            idx = j + 1
        return starts

    def _looks_like_title(self, line: str) -> bool:
        # simple heuristic: many capitals OR most words title-cased
        words = [w for w in re.split(r"\s+", line.strip()) if w]
        if not words:
            return False

        # many capital letters (relative to letters)
        letters = [c for c in line if c.isalpha()]
        if letters:
            cap_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if cap_ratio >= 0.6:
                return True

        # Title Case heuristic: majority of words start with uppercase
        titleish = sum(1 for w in words if re.match(r"^[A-Z][a-zA-Z\-µα-ωΑ-Ω0-9]*$", w)) / len(words)
        return titleish >= 0.6

    def _count_tokens(self, text: str) -> int:
        if self._encoder is None:
            # approximate if tiktoken isn't available
            # (token ~= 0.75 * words for English; here we just use words count as a proxy)
            return max(1, len(re.findall(r"\w+", text)))
        try:
            return len(self._encoder.encode(text))
        except Exception:
            return max(1, len(re.findall(r"\w+", text)))

    # Convenience passthroughs
    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def chunk_size(self) -> int:
        return self._splitter._chunk_size if hasattr(self._splitter, "_chunk_size") else self._config["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self._splitter._chunk_overlap if hasattr(self._splitter, "_chunk_overlap") else self._config["chunk_overlap"]

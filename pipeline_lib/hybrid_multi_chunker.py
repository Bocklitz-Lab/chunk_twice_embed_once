# hybrid_multi_chunker.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

# LangChain token splitter (v0.1+ and v0.2+ compatibility)
try:
    from langchain_text_splitters import TokenTextSplitter as LCTokenTextSplitter
except ImportError:
    from langchain.text_splitter import TokenTextSplitter as LCTokenTextSplitter

# Reuse your sectioner (import path flexible)
try:
    from pipeline_lib.hierarchical_section_chunker import HierarchicalSectionChunker
except Exception:
    try:
        from hierarchical_section_chunker import HierarchicalSectionChunker
    except Exception:
        HierarchicalSectionChunker = None  # type: ignore

# --- NEW: Hugging Face tokenizer support (no OpenAI/tiktoken) ---
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore


# ---------- Regex helpers / constants ----------

# Front matter / end matter we may want to suppress at section granularity
FRONT_MATTER_TITLES_RE = re.compile(
    r"""^\s*(graphical\s+abstract|keywords?|corresponding\s+author|author\s+information|orcid|
            author\s+contributions?|funding(\s+sources?)?|acknowledg(e)?ments?|abbreviations|
            nomenclature|glossary)\s*:?\s*$""",
    flags=re.IGNORECASE | re.VERBOSE,
)

# Real body titles (we want to prefer these as section titles)
REAL_BODY_TITLES_RE = re.compile(
    r"""^\s*((\d+(\.\d+){0,3}\s*[-\).]?\s*)?
            (abstract|introduction|background|methods?|materials\s+and\s+methods|
             experimental(\s+(section|details|setup|procedures?))?|results(\s+and\s+discussion)?|
             discussion|conclusions?|summary|appendix|supplementary(\s+information|materials)?)
         )\s*:?\s*$""",
    flags=re.IGNORECASE | re.VERBOSE,
)

# Generic heading-ish line (numbered heading or Title Case single line)
HEADING_LINE_RE = re.compile(
    r"""^\s*(?:(?:\d+)(?:\.\d+){0,3}\s*[-\).]?\s*)?[A-Z][^\n]{0,120}$""",
    flags=re.MULTILINE,
)

# Markdown ATX heading like "#", "##", "### ..." (strip for display, treat as heading)
MD_HEADING_RE = re.compile(r"^\s{0,3}(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

# Image-only markdown like ![Alt](src)
IMAGE_ONLY_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$", flags=re.MULTILINE)

# Author / affiliation / email detectors (skip as section titles)
AUTHOR_LINE_RE = re.compile(
    r"""^
        (?:
          (?:[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)   # Name-like tokens
          (?:\s*[A-Z]\.)*                        # optional initials
          (?:[,\s]|$)
        ){1,6}
        (?:\d+|\*|,)?$
    """,
    re.VERBOSE,
)

AFFILIATION_HINT_RE = re.compile(
    r"(department|university|institute|laboratory|school|faculty|campus|college|centre|center)",
    re.IGNORECASE,
)

EMAIL_LINE_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


# ------------------ NEW: HF-based splitter ------------------

class HFTokenTextSplitter:
    """
    Minimal token-based text splitter using a Hugging Face tokenizer.
    Mirrors the subset of the LangChain TokenTextSplitter interface we rely on.
    """
    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer,
        strip_whitespace: bool = True,
        add_start_index: bool = False,  # accepted for parity; unused here
        **_: Any,
    ) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)
        self.tokenizer = tokenizer
        self.strip_whitespace = bool(strip_whitespace)

    # --- internal helpers ---

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _last_n_tokens_text(self, text: str, n: int) -> str:
        if n <= 0:
            return ""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        tail = ids[-n:] if len(ids) > n else ids
        # Decode w/o special tokens; HF decodes whitespace reasonably
        return self.tokenizer.decode(tail, skip_special_tokens=True)

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        # Split into tokens softly by growing from whitespace-aware segments
        # We iterate over segments preserving spaces using a capture group.
        segments = re.split(r"(\s+)", text)
        out: List[str] = []
        current: str = ""

        def finalize_current():
            nonlocal current
            c = current.strip() if self.strip_whitespace else current
            if c:
                out.append(c)
            current = ""

        for seg in segments:
            if seg == "":
                continue
            candidate = (current + seg) if current else seg
            if self._token_len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    finalize_current()
                    # Start new chunk with overlap tail
                    if self.chunk_overlap > 0 and out:
                        tail_text = self._last_n_tokens_text(out[-1], self.chunk_overlap)
                        current = tail_text
                    else:
                        current = ""
                    # Try to add seg into new chunk (may still be too big -> split hard)
                    if self._token_len(seg) <= self.chunk_size:
                        current += (seg if not current else seg)
                    else:
                        # Hard split very long segment by tokens
                        ids = self.tokenizer.encode(seg, add_special_tokens=False)
                        start = 0
                        while start < len(ids):
                            end = min(start + self.chunk_size, len(ids))
                            piece = self.tokenizer.decode(ids[start:end], skip_special_tokens=True)
                            piece = piece.strip() if self.strip_whitespace else piece
                            if piece:
                                out.append(piece)
                            start = end
                        # seed next current with overlap
                        if self.chunk_overlap > 0 and out:
                            tail_text = self._last_n_tokens_text(out[-1], self.chunk_overlap)
                            current = tail_text
                        else:
                            current = ""
                else:
                    # current is empty; seg itself exceeds chunk size -> hard split
                    ids = self.tokenizer.encode(seg, add_special_tokens=False)
                    start = 0
                    while start < len(ids):
                        end = min(start + self.chunk_size, len(ids))
                        piece = self.tokenizer.decode(ids[start:end], skip_special_tokens=True)
                        piece = piece.strip() if self.strip_whitespace else piece
                        if piece:
                            out.append(piece)
                        start = end
                    if self.chunk_overlap > 0 and out:
                        tail_text = self._last_n_tokens_text(out[-1], self.chunk_overlap)
                        current = tail_text
                    else:
                        current = ""

        if current:
            finalize_current()
        return out

    # light parity with LangChain splitter
    def create_documents(self, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None):
        metadatas = metadatas or [{} for _ in texts]
        docs = []
        for t, meta in zip(texts, metadatas):
            for chunk in self.split_text(t):
                docs.append({"page_content": chunk, "metadata": meta})
        return docs


class HybridMultiGranularityChunker:
    """
    Multi-level chunker: section -> paragraph -> sentence (configurable).

    - Uses HierarchicalSectionChunker for section detection (with pass-through params).
    - Token-based splitting within each level with level-specific sizes/overlaps.
    - Returns a list[dict] with granular metadata & parent-scope character offsets
      (exact chunk start/end are computed by your runner when compute_offsets=True).
    """

    def __init__(
        self,
        *,
        # tokenization defaults
        encoding_name: str = "cl100k_base",  # kept for API compatibility; unused if HF tokenizer is used
        model_name: Optional[str] = None,     # now treated as HF model id if provided
        strip_whitespace: bool = True,

        # define levels & per-level sizes/overlaps (in tokens)
        levels: Sequence[str] = ("section", "paragraph", "sentence"),
        chunk_sizes: Optional[Dict[str, int]] = None,
        chunk_overlaps: Optional[Dict[str, int]] = None,

        # if no token counter, use approx chars per token (kept for future use)
        approx_chars_per_token: int = 4,

        # optional: configure the internal sectioner
        sectioner: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.levels = [lvl.lower() for lvl in levels]
        self.strip_whitespace = strip_whitespace
        self.approx_chars_per_token = max(1, int(approx_chars_per_token))

        # sane defaults per level
        default_sizes = {"section": 1200, "paragraph": 500, "sentence": 200}
        default_overlaps = {"section": 150, "paragraph": 60, "sentence": 40}

        self.chunk_sizes = {**default_sizes, **(chunk_sizes or {})}
        self.chunk_overlaps = {**default_overlaps, **(chunk_overlaps or {})}

        # Try to prepare a HF tokenizer if model_name and transformers are available
        self._hf_tokenizer = None
        if model_name and AutoTokenizer is not None:
            try:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception:
                self._hf_tokenizer = None  # fallback to LC splitter

        # build a TokenTextSplitter per level
        self._splitters: Dict[str, Any] = {}
        for lvl in set(self.levels):
            size = int(self.chunk_sizes.get(lvl, default_sizes.get(lvl, 500)))
            overlap = int(self.chunk_overlaps.get(lvl, default_overlaps.get(lvl, 50)))
            if overlap > size:
                overlap = max(0, size // 4)  # safety fallback

            if self._hf_tokenizer is not None:
                # Use HF-based token counting
                self._splitters[lvl] = HFTokenTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                    tokenizer=self._hf_tokenizer,
                    add_start_index=False,
                    strip_whitespace=strip_whitespace,
                )
            else:
                # Fallback: LangChain TokenTextSplitter (character-based fallback;
                # API kept identical to your previous code)
                self._splitters[lvl] = LCTokenTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                    encoding_name=encoding_name,  # kept for compatibility; may be ignored by LC
                    model_name=None,              # avoid OpenAI/tiktoken path
                    allowed_special=set(),
                    disallowed_special="all",
                    add_start_index=False,
                    strip_whitespace=strip_whitespace,
                )

        # sectioner
        self._sectioner = None
        if "section" in self.levels:
            if HierarchicalSectionChunker is None:
                raise ImportError("HierarchicalSectionChunker not available for hybrid mode.")
            self._sectioner = HierarchicalSectionChunker(**(sectioner or {}))

    # ------------- Public API -------------

    def split_text(self, text: str) -> List[dict]:
        """
        Returns list of dict chunks:
          {
            "text": ...,
            "granularity": "section" | "paragraph" | "sentence",
            "section_title": optional,
            "section_index": optional,
            "parent_section_start": int|None,
            "parent_section_end": int|None,
            "parent_par_start": int|None,
            "parent_par_end": int|None,
          }
        Note: start/end offsets for the chunk itself are computed by the runner
              if compute_offsets=True. We attach only parent-scope boundaries here.
        """
        text = text or ""
        if not text.strip():
            return []

        results: List[dict] = []

        if "section" in self.levels:
            sections = self._get_sections(text)  # list of (s, e, title, idx)
        else:
            # single "pseudo-section": whole doc
            sections = [(0, len(text), None, 0)]

        # iterate sections
        for sec_start, sec_end, sec_title, sec_idx in sections:
            sec_text = text[sec_start:sec_end]
            title_for_check = (sec_title or "").strip()
            is_front_matter = bool(FRONT_MATTER_TITLES_RE.match(title_for_check))

            # SECTION-LEVEL chunks (suppress for front-matter-like sections)
            if "section" in self.levels and not is_front_matter:
                for s_chunk in self._split_level(sec_text, level="section"):
                    results.append({
                        "text": s_chunk,
                        "granularity": "section",
                        "section_title": sec_title,
                        "section_index": sec_idx,
                        "parent_section_start": sec_start,
                        "parent_section_end": sec_end,
                        "parent_par_start": None,
                        "parent_par_end": None,
                    })

            # derive paragraphs within the section
            if "paragraph" in self.levels or "sentence" in self.levels:
                paragraphs = self._split_paragraphs(sec_text, base_offset=sec_start)
            else:
                paragraphs = []

            # PARAGRAPH-LEVEL chunks
            if "paragraph" in self.levels:
                for par_start, par_end in paragraphs:
                    par_text = text[par_start:par_end]
                    # optionally drop image-only paragraphs from paragraph granularity
                    if IMAGE_ONLY_RE.fullmatch(par_text.strip() or ""):
                        continue
                    for p_chunk in self._split_level(par_text, level="paragraph"):
                        results.append({
                            "text": p_chunk,
                            "granularity": "paragraph",
                            "section_title": sec_title,
                            "section_index": sec_idx,
                            "parent_section_start": sec_start,
                            "parent_section_end": sec_end,
                            "parent_par_start": par_start,
                            "parent_par_end": par_end,
                        })

            # SENTENCE-LEVEL chunks
            if "sentence" in self.levels:
                for par_start, par_end in paragraphs:
                    par_text = text[par_start:par_end]

                    # skip headings at sentence level to avoid "1." / "Introduction" / "### 2." splits
                    if self._is_heading_paragraph(par_text):
                        continue
                    # skip pure image/figure lines
                    if IMAGE_ONLY_RE.fullmatch(par_text.strip() or ""):
                        continue

                    sentences = self._split_sentences(par_text, base_offset=par_start)
                    for s_start, s_end in sentences:
                        s_text = text[s_start:s_end]
                        # apply token splitter in case sentence is still long
                        for sen_chunk in self._split_level(s_text, level="sentence"):
                            results.append({
                                "text": sen_chunk,
                                "granularity": "sentence",
                                "section_title": sec_title,
                                "section_index": sec_idx,
                                "parent_section_start": sec_start,
                                "parent_section_end": sec_end,
                                "parent_par_start": par_start,
                                "parent_par_end": par_end,
                            })

        return results

    def create_documents(self, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None):
        """
        Optional parity method: returns LC Document-like objects if your pipeline expects it.
        """
        docs = []
        metadatas = metadatas or [{} for _ in texts]
        # choose an arbitrary splitter (sentence level) for doc object creation
        chosen_splitter = self._splitters.get("sentence") or next(iter(self._splitters.values()))
        for t, meta in zip(texts, metadatas):
            for item in self.split_text(t):
                m = {**meta, **{k: v for k, v in item.items() if k != "text"}}
                # both LC and our HF splitter implement create_documents
                docs.extend(chosen_splitter.create_documents([item["text"]], [m]))
        return docs

    # ------------- Internals -------------

    def _get_sections(self, text: str) -> List[Tuple[int, int, Optional[str], int]]:
        """
        Use the internal HierarchicalSectionChunker to create *true* sections, but we need
        their spans & titles. We call its private routines to get spans, then compute a
        better section title via _best_section_title().
        """
        sec = self._sectioner
        if not sec:
            return [(0, len(text), None, 0)]

        # Use sectioner's detection/merge logic
        headings = sec._find_headings(text)  # type: ignore[attr-defined]
        if not headings:
            return [(0, len(text), None, 0)]

        headings = sec._dedupe_and_sort(headings)  # type: ignore[attr-defined]
        spans = sec._build_section_spans(text, headings)  # type: ignore[attr-defined]

        # drop tail sections if configured
        drop_labels = sec._config.get("drop_tail_sections_matching", [])  # type: ignore[attr-defined]
        if drop_labels:
            spans = sec._drop_tail_sections(text, spans, drop_labels)  # type: ignore[attr-defined]

        # merge short if requested
        if sec._config.get("join_short_sections", True):  # type: ignore[attr-defined]
            spans = sec._merge_short_sections(text, spans, sec._config.get("min_section_tokens", 200))  # type: ignore[attr-defined]

        out: List[Tuple[int, int, Optional[str], int]] = []
        for idx, (s, e) in enumerate(spans):
            title = self._best_section_title(text, (s, e))
            # strip markdown hashes if any slipped through (e.g., "### Title" -> "Title")
            if title:
                md = MD_HEADING_RE.match(title)
                if md:
                    title = md.group("title").strip()
            out.append((s, e, title, idx))
        return out

    def _best_section_title(self, full_text: str, span: Tuple[int, int]) -> Optional[str]:
        s, e = span
        snippet = full_text[s:e]

        # prefer LaTeX \section{...}
        m = re.search(
            r"""\\(?P<lvl>section|subsection|subsubsection)\*?\{(?P<title>[^}]+)\}""",
            snippet, flags=re.IGNORECASE
        )
        if m:
            t = m.group("title").strip()
            if t:
                return t

        # Walk lines; choose the first good candidate
        for raw in snippet.splitlines():
            l = raw.strip()
            if not l:
                continue

            # Normalize Markdown ATX headings ("### Title" -> "Title")
            md = MD_HEADING_RE.match(l)
            if md:
                l = md.group("title").strip()

            # Skip obvious non-titles
            if FRONT_MATTER_TITLES_RE.match(l):
                continue
            if EMAIL_LINE_RE.search(l):
                continue
            if AUTHOR_LINE_RE.match(l):
                continue
            if AFFILIATION_HINT_RE.search(l):
                continue

            # Prefer known body titles
            if REAL_BODY_TITLES_RE.match(l):
                return l

            # Heuristic acceptance:
            # - allow fairly long titles (<= 200 chars)
            # - avoid lines that end with full stop (likely a sentence)
            if len(l) <= 200 and not l.endswith("."):
                return l

        return None

    def _split_paragraphs(self, text: str, base_offset: int = 0) -> List[Tuple[int, int]]:
        """
        Paragraphs split on >=1 blank lines. Keeps indices relative to original text via base_offset.
        Trims edges if strip_whitespace is True.
        """
        starts: List[int] = []
        ends: List[int] = []
        idx = 0
        n = len(text)

        # Split on 2+ newlines
        for m in re.finditer(r"(?:\r?\n){2,}", text):
            ends.append(m.start())
            starts.append(idx)
            idx = m.end()

        # last paragraph
        if idx < n:
            starts.append(idx)
            ends.append(n)

        out: List[Tuple[int, int]] = []
        for s, e in zip(starts, ends):
            seg = text[s:e]
            if self.strip_whitespace:
                left = len(seg) - len(seg.lstrip())
                right = len(seg) - len(seg.rstrip())
                s2 = s + left
                e2 = e - right
            else:
                s2, e2 = s, e
            if s2 < e2:
                out.append((base_offset + s2, base_offset + e2))
        return out

    def _is_heading_paragraph(self, text: str) -> bool:
        """
        Recognize heading-only paragraphs to avoid sentence-splitting them.
        """
        t = (text or "").strip()
        if not t:
            return False
        if FRONT_MATTER_TITLES_RE.match(t):
            return True
        if REAL_BODY_TITLES_RE.match(t):
            return True
        if MD_HEADING_RE.match(t):
            return True
        if HEADING_LINE_RE.match(t):
            return True
        return False

    def _split_sentences(self, text: str, base_offset: int = 0) -> List[Tuple[int, int]]:
        """
        Lightweight rule-based sentence splitter that’s chemistry-friendly:
        - Respects common abbreviations and units
        - Protects initials (Z.Q., C.N.N., H. D.)
        - Avoids splitting on figure refs, months, etc.
        """
        # Protect tokens by temporarily marking the period with a special token
        marker = "§§DOT§§"
        protected_patterns = [
            # Latin & refs
            r"e\.g\.", r"i\.e\.", r"cf\.", r"vs\.", r"et al\.", r"al\.", r"ca\.", r"approx\.",
            r"Fig\.", r"Figs\.", r"Eq\.", r"Eqs\.", r"Ref\.", r"Refs\.",
            # Titles
            r"Dr\.", r"Mr\.", r"Ms\.", r"Prof\.",
            # Time/units/abbr
            r"min\.", r"sec\.", r"hr\.", r"hrs\.", r"vol\.", r"conc\.", r"temp\.", r"wt\.", r"mol\.", r"No\.", r"Nos\.",
            # Months
            r"Jan\.", r"Feb\.", r"Mar\.", r"Apr\.", r"Jun\.", r"Jul\.", r"Aug\.", r"Sep\.", r"Oct\.", r"Nov\.", r"Dec\.",
            # Initials & sequences of initials
            r"(?:(?<!\w)(?:[A-Z]\.){2,})",  # Z.Q. | C.N.N.
            r"(?<!\w)[A-Z]\.\s[A-Z]\.",     # H. D.
        ]

        prot_text = text
        for pat in protected_patterns:
            prot_text = re.sub(pat, lambda m: m.group(0).replace(".", marker), prot_text)

        # Split on period/question/exclamation followed by whitespace + capital/( or start of latex \
        pieces = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(\\])", prot_text)

        # Restore dots
        pieces = [p.replace(marker, ".") for p in pieces]

        # Build spans with offsets
        out: List[Tuple[int, int]] = []
        cursor = 0
        for p in pieces:
            seg = p.strip() if self.strip_whitespace else p
            if not seg:
                cursor += len(p)
                continue
            i = text.find(seg, cursor)
            if i == -1:
                # fallback: align by length window
                i = cursor
            j = i + len(seg)
            out.append((base_offset + i, base_offset + j))
            cursor = j
        return out

    def _split_level(self, span_text: str, level: str) -> List[str]:
        """
        Token-based splitting with the prebuilt splitter for `level`.
        """
        splitter = self._splitters[level]
        if not span_text or not span_text.strip():
            return []
        return splitter.split_text(span_text)

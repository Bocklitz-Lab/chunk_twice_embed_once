"""Span-based relevance labels (qrels) for the ChemQuests retrieval tasks.

A question q has an evidence span S = [s, e) in its source document. A chunk is [a, b) in the
same character coordinates (the chunker writes exact positions). Given a cover level c:

  Step 1  Every chunk of q's document that by itself holds >= c of the characters of S is relevant.
  Step 2  If no chunk does, chunks are added greedily by NEW (not yet covered) span characters
          until the added chunks together cover >= c of S. A chunk that adds no new character is
          never added. Ties are broken by chunk order in the document.
  Step 3  If c cannot be reached even with all overlapping chunks, every chunk that overlaps S
          is relevant.
  Step 0  No chunk overlaps S: no label (the question is kept and scores 0).

All comparisons are integer comparisons. c is an exact fraction (see ``parse_cover_level``):
"a chunk holds >= c of S" means ``overlap_chars * D >= N * len(S)`` for c = N / D. No float is
ever compared.

c is a parameter. It has no default in this module: it comes from the config
(``labeling.cover_level``), from ``--param`` / ``EXTRA_PARAMS``, or from the command line.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Sequence, Tuple

METHODS = ("span", "lexical")
LEXICAL_QRELS_FILE = "qrels_lexical.jsonl"


# --------------------------------------------------------------------------------------
# cover level
# --------------------------------------------------------------------------------------
def parse_cover_level(value: Any) -> Fraction:
    """Return c as an exact Fraction. Raise ValueError unless 0 < c <= 1.

    A float or a string is read as the decimal number it is written as (``str(float)`` is the
    shortest representation that round-trips), so ``0.8`` becomes exactly 4/5, not the binary
    float nearest to it. A Fraction must be a finite decimal, because c also appears in file names.
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"cover level must be a number in (0, 1], got {value!r}")
    try:
        if isinstance(value, Fraction):
            frac = value
        elif isinstance(value, (int, float, str, Decimal)):
            frac = Fraction(Decimal(str(value).strip()))
        else:
            raise ValueError(f"cover level must be a number in (0, 1], got {value!r}")
    except (InvalidOperation, ValueError, OverflowError, ZeroDivisionError) as e:
        raise ValueError(f"cover level must be a number in (0, 1], got {value!r}") from e
    if not (0 < frac <= 1):
        raise ValueError(f"cover level must lie in (0, 1], got {value!r}")
    den = frac.denominator
    for p in (2, 5):
        while den % p == 0:
            den //= p
    if den != 1:
        raise ValueError(f"cover level must be a finite decimal number (it is used in file names), got {value!r}")
    return frac


def cover_tag(c: Any) -> str:
    """File-name tag of c, injective and exact: percent with 3 digits, 'p' for a decimal point.

    Examples (for percentages): 80 -> c080, 100 -> c100, 87.5 -> c087p5.
    """
    frac = parse_cover_level(c)
    percent = frac * 100
    k = 0
    while (percent * 10 ** k).denominator != 1:
        k += 1
    scaled = int(percent * 10 ** k)
    whole, rest = divmod(scaled, 10 ** k)
    return f"c{whole:03d}" + (f"p{rest:0{k}d}" if k else "")


def qrels_filename(method: str, cover_level: Any = None) -> str:
    """Name of the qrels file for a label method (and c, for span labels)."""
    if method == "lexical":
        return LEXICAL_QRELS_FILE
    if method == "span":
        if cover_level is None:
            raise ValueError("labeling.cover_level is required for method 'span'")
        return f"qrels_span_{cover_tag(cover_level)}.jsonl"
    raise ValueError(f"labeling.method must be one of {METHODS}, got {method!r}")


def label_stats_filename(cover_level: Any) -> str:
    return f"label_stats_{cover_tag(cover_level)}.json"


def resolve_labeling(labeling: Dict[str, Any] | None) -> Dict[str, Any]:
    """Validate a ``labeling`` block and fill in ``qrels_file`` (derived from method and c unless set).

    Returns a new dict with ``method``, ``cover_level`` (a number as written; None only for the
    lexical method when it is not given) and ``qrels_file``.
    """
    block = dict(labeling or {})
    method = block.get("method")
    if method not in METHODS:
        raise ValueError(f"labeling.method must be one of {METHODS}, got {method!r}")
    c = block.get("cover_level")
    if c is not None or method == "span":
        parse_cover_level(c)  # raises unless 0 < c <= 1
    if not block.get("qrels_file"):
        block["qrels_file"] = qrels_filename(method, c)
    block.setdefault("cover_level", c)
    return block


# --------------------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------------------
Overlap = Tuple[Any, int, int, int]  # (key, lo, hi, order) with lo < hi, absolute coordinates inside S


def clip_to_span(s: int, e: int, chunks: Iterable[Tuple[Any, int, int, int]]) -> List[Overlap]:
    """Overlap of every chunk (key, a, b, order), [a, b), with S = [s, e); chunks without overlap are dropped."""
    out: List[Overlap] = []
    for key, a, b, order in chunks:
        lo, hi = max(a, s), min(b, e)
        if b > a and hi > lo:
            out.append((key, lo, hi, order))
    return out


def label_overlaps(s: int, e: int, overlaps: Iterable[Overlap], c: Any) -> Dict[str, Any]:
    """Apply the rule to the chunks that overlap S = [s, e).

    ``overlaps``: (key, lo, hi, order); [lo, hi) is the part of the chunk inside S.
    Returns dict(ids, step, coverage, covered, span_len, reason, n_overlap): ``ids`` are the relevant
    chunk keys in chunk order; the result does not depend on the order of the input list.
    """
    frac = parse_cover_level(c)
    num, den = frac.numerator, frac.denominator
    span_len = e - s
    if span_len <= 0:
        return _result([], 0, 0, span_len, "empty_span", 0)
    ov = sorted(overlaps, key=lambda x: x[3])  # chunk order; ties in Step 2 go to the lower order
    if not ov:
        return _result([], 0, 0, span_len, "no_overlapping_chunk", 0)

    # ---- Step 1: chunks that hold >= c of S by themselves
    full = [x for x in ov if den * (x[2] - x[1]) >= num * span_len]
    if full:
        covered = _union_len(span_len, s, full)
        return _result([x[0] for x in full], 1, covered, span_len, "", len(ov))

    # ---- can c be reached at all?
    all_covered = _union_len(span_len, s, ov)
    if den * all_covered < num * span_len:
        return _result([x[0] for x in ov], 3, all_covered, span_len, "coverage_unreachable", len(ov))

    # ---- Step 2: greedy by new (uncovered) characters
    cov = bytearray(span_len)
    chosen: List[Overlap] = []
    rest = list(ov)
    n_cov = 0
    while rest:
        best, best_new = None, 0
        for x in rest:  # ascending chunk order, so a strict ">" keeps the lower order on ties
            new = (x[2] - x[1]) - cov.count(1, x[1] - s, x[2] - s)
            if new > best_new:
                best, best_new = x, new
        if best is None:  # nothing adds a new character
            break
        chosen.append(best)
        rest.remove(best)
        cov[best[1] - s:best[2] - s] = b"\x01" * (best[2] - best[1])
        n_cov += best_new
        if den * n_cov >= num * span_len:
            break
    chosen.sort(key=lambda x: x[3])
    return _result([x[0] for x in chosen], 2, n_cov, span_len, "", len(ov))


def label_span(s: int, e: int, chunks: Iterable[Tuple[Any, int, int, int]], c: Any) -> Dict[str, Any]:
    """Rule applied to raw chunks (key, a, b, order) of the question's document."""
    return label_overlaps(s, e, clip_to_span(s, e, chunks), c)


def _union_len(span_len: int, s: int, ov: Sequence[Overlap]) -> int:
    cov = bytearray(span_len)
    for _, lo, hi, _ in ov:
        cov[lo - s:hi - s] = b"\x01" * (hi - lo)
    return cov.count(1)


def _result(ids, step, covered, span_len, reason, n_overlap) -> Dict[str, Any]:
    return dict(
        ids=ids,
        step=step,
        coverage=(covered / span_len) if span_len > 0 else 0.0,
        covered=covered,
        span_len=span_len,
        reason=reason,
        n_overlap=n_overlap,
    )


# --------------------------------------------------------------------------------------
# overlap table (independent of c)
# --------------------------------------------------------------------------------------
def overlap_rows(query_id: int, doc_id: str, s: int, e: int, chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per chunk of the document that overlaps S = [s, e).

    ``chunks``: dicts with chunk_id, chunk_index, start, end. Any cover level can be derived from
    these rows (``label_from_rows``): they hold the overlap interval, not only its length.
    """
    span_len = e - s
    rows = []
    for ch in sorted(chunks, key=lambda x: x["chunk_index"]):
        lo, hi = max(ch["start"], s), min(ch["end"], e)
        if ch["end"] > ch["start"] and hi > lo:
            n = hi - lo
            rows.append({
                "query_id": str(query_id),
                "doc_id": str(doc_id),
                "chunk_id": ch["chunk_id"],
                "chunk_index": ch["chunk_index"],
                "chunk_start": ch["start"],
                "chunk_end": ch["end"],
                "span_start": s,
                "span_end": e,
                "overlap_start": lo,
                "overlap_end": hi,
                "overlap_chars": n,
                "span_len": span_len,
                "fraction": n / span_len,
                "contains_span": n == span_len,
            })
    return rows


def label_from_rows(rows: Sequence[Dict[str, Any]], c: Any, span_start: int | None = None, span_end: int | None = None) -> Dict[str, Any]:
    """Apply the rule to the overlap-table rows of ONE question (``span_start``/``span_end`` default to the rows' own)."""
    if not rows:
        if span_start is None or span_end is None:
            return label_overlaps(0, 0, [], c)
        return label_overlaps(span_start, span_end, [], c)
    s = rows[0]["span_start"] if span_start is None else span_start
    e = rows[0]["span_end"] if span_end is None else span_end
    ov = [(r["chunk_id"], r["overlap_start"], r["overlap_end"], r["chunk_index"]) for r in rows]
    return label_overlaps(s, e, ov, c)

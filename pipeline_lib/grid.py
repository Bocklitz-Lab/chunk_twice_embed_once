"""The Phase 2 configuration grid (the same constraints the Makefile applies)."""
from __future__ import annotations

from typing import Iterable, List, Tuple

CHUNKERS = ["recursive_token", "fixed_token", "semantic_recursive", "semantic_fixed", "hierarchical_section"]
SIZES = [128, 384, 192, 256, 320, 448, 512]
OVERLAPS = [0, 16, 32, 48, 64]
TOKEN_LIMIT = 512        # L <= TOKEN_LIMIT
TOTAL_LIMIT = 512        # L + O <= TOTAL_LIMIT
MIN_STRIDE = 96          # L - O >= MIN_STRIDE
MAX_OVERLAP_ABS = 64     # O <= min(MAX_OVERLAP_ABS, MAX_OVERLAP_PCT % of L)
MAX_OVERLAP_PCT = 30


def is_valid(size: int, overlap: int) -> bool:
    if size > TOKEN_LIMIT:
        return False
    max_o = min(size * MAX_OVERLAP_PCT // 100, MAX_OVERLAP_ABS)
    return overlap <= max_o and size - overlap >= MIN_STRIDE and size + overlap <= TOTAL_LIMIT


def grid(sizes: Iterable[int] = SIZES, overlaps: Iterable[int] = OVERLAPS) -> List[Tuple[int, int]]:
    """Valid (size, overlap) pairs, sizes ascending then overlaps ascending."""
    return [(s, o) for s in sorted(sizes) for o in sorted(overlaps) if is_valid(s, o)]

import pytest

pytest.importorskip("tqdm")

from stages.chunking import locate_chunk_offsets


def _windows(text: str, size: int, step: int):
    """Sliding character windows (overlap = size - step). Returns (chunks, true_starts)."""
    starts = list(range(0, max(1, len(text) - size + 1), step))
    return [text[s:s + size] for s in starts], starts


def _assert_exact(text, chunks, offsets):
    for chunk, (s, e) in zip(chunks, offsets):
        if s is not None:
            assert text[s:e] == chunk


def test_no_overlap_positions_are_exact():
    text = "".join(f"word{i} " for i in range(200))
    chunks, starts = _windows(text, size=50, step=50)
    offsets = locate_chunk_offsets(text, chunks)
    assert [s for s, _ in offsets] == starts
    _assert_exact(text, chunks, offsets)


def test_overlap_positions_are_exact():
    # Regression: the previous implementation searched forward from the END of the previous
    # chunk, so overlapping chunks got wrong positions.
    text = "".join(f"word{i} " for i in range(200))
    chunks, starts = _windows(text, size=60, step=40)  # 20-character overlap
    offsets = locate_chunk_offsets(text, chunks)
    assert [s for s, _ in offsets] == starts
    assert [e for _, e in offsets] == [s + 60 for s in starts]
    _assert_exact(text, chunks, offsets)


def test_repeated_chunk_text_is_located_in_document_order():
    block = "alpha beta gamma delta epsilon. "
    text = block * 3
    chunks = [block, block, block]
    offsets = locate_chunk_offsets(text, chunks)
    assert offsets == [(0, len(block)), (len(block), 2 * len(block)), (2 * len(block), 3 * len(block))]


def test_non_verbatim_chunk_gets_no_position_and_later_chunks_still_located():
    text = "First sentence here.\n\nSecond sentence here.\n\nThird sentence here."
    chunks = [
        "First sentence here.",
        "second sentence here",        # lower-cased / re-joined: not a substring
        "Third sentence here.",
    ]
    offsets = locate_chunk_offsets(text, chunks)
    assert offsets[0] == (0, 20)
    assert offsets[1] == (None, None)
    assert offsets[2] == (text.index("Third"), len(text))


def test_empty_chunk_gets_no_position():
    assert locate_chunk_offsets("abc", ["", "abc"]) == [(None, None), (0, 3)]

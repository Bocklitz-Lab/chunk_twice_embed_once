"""The span labelling rule: the 34 constructed cases of review_samples/chunker_audit/rule_tests.md, ported
with c = 0.90, plus cover-level handling and a differential test against the scratch reference rule.

Cases 1-31 test the rule. Cases 32-34 of the audit test the OFFSET RECOVERY of the old stage 4
(a search in the paper text). The pipeline no longer searches (the chunkers write native positions),
but the three cases are ported as they were so that all 34 exist: 32 and 34 run against the legacy
``locate_chunk_offsets`` that is still in ``stages/chunking.py``; 33 runs against a copy of the
repeat-aware search of the audit (a reference function in this file, not pipeline code).
"""
import importlib.util
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_lib.span_labels import (  # noqa: E402
    cover_tag,
    label_span,
    parse_cover_level,
    qrels_filename,
    resolve_labeling,
)

C = "0.90"  # the cover level of the audit cases; the rule itself has no default


def ids(s, e, chunks, c=C):
    r = label_span(s, e, chunks, c)
    return r["ids"], r["step"]


def grid(n, w, ov=0):
    """Contiguous chunks of width w over a document of n characters: c0, c1, ... (key, a, b, order)."""
    out, a, k = [], 0, 0
    while a < n:
        b = min(n, a + w)
        out.append((f"c{k}", a, b, k))
        k += 1
        if b == n:
            break
        a += w - ov
    return out


G = grid(1000, 100)  # c0=[0,100) ... c9=[900,1000)
G32 = grid(1000, 100, ov=30)  # starts 0,70,140,210,...


def test_01_span_fully_inside_one_chunk():
    assert ids(120, 160, G) == (["c1"], 1)


def test_02_span_split_51_49():
    assert ids(149, 249, G) == (["c1", "c2"], 2)


def test_03_span_split_95_5_only_the_big_chunk():
    assert ids(105, 205, G) == (["c1"], 1)


def test_04_span_over_three_chunks_needs_all():
    assert ids(70, 220, G) == (["c0", "c1", "c2"], 2)


def test_05_tiny_third_chunk_not_needed():
    assert ids(60, 210, G) == (["c0", "c1"], 2)


def test_06_span_inside_the_overlap_of_two_chunks():
    assert ids(75, 95, G32) == (["c0", "c1"], 1)


def test_07_chunk_adding_no_new_coverage_is_never_added():
    r = label_span(100, 230, [("big", 100, 190, 0), ("sub", 120, 180, 1), ("tail", 190, 230, 2)], C)
    assert (r["ids"], r["step"]) == (["big", "tail"], 2)


def test_08_greedy_order_is_by_new_characters_not_raw_overlap():
    r = label_span(0, 100, [("A", 0, 60, 0), ("B", 20, 100, 1), ("C", 40, 60, 2)], C)
    assert (r["ids"], r["step"]) == (["A", "B"], 2)


def test_09_gap_inside_span_makes_cover_unreachable_step_3():
    assert ids(50, 350, [("L", 0, 100, 0), ("R", 300, 400, 1)]) == (["L", "R"], 3)


def test_10_gap_inside_span_but_still_above_cover():
    assert ids(0, 200, [("L", 0, 95, 0), ("R", 105, 300, 1)]) == (["L", "R"], 2)


def test_11_span_without_overlapping_chunk():
    assert ids(500, 520, [("a", 0, 100, 0), ("b", 600, 700, 1)]) == ([], 0)


def test_12_empty_chunk_list():
    assert ids(500, 520, []) == ([], 0)


def test_13_span_at_the_very_start():
    assert ids(0, 30, G) == (["c0"], 1)


def test_14_span_at_the_very_end():
    assert ids(970, 1000, G) == (["c9"], 1)


def test_15_span_is_the_whole_document_step_2():
    assert ids(0, 1000, G)[1] == 2


def test_16_whole_document_span_needs_nine_chunks():
    assert len(label_span(0, 1000, G, C)["ids"]) == 9


def _duplicate_passage_doc():
    para = "Sodium hydroxide (0.1 M) was added dropwise to the solution under stirring. "
    doc = "Intro. " * 20 + para + "Middle text. " * 40 + para + "End. " * 20
    p1 = doc.find(para)
    p2 = doc.find(para, p1 + 1)
    chunks, w, a, k = [], 120, 0, 0
    while a < len(doc):
        chunks.append((f"c{k}", a, min(len(doc), a + w), k))
        a += w
        k += 1
    return p1, p2, chunks


def test_17_duplicate_passage_first_occurrence_labels_first_occurrence_chunk():
    p1, _, chunks = _duplicate_passage_doc()
    lab1 = label_span(p1 + 5, p1 + 60, chunks, C)
    want = [c[0] for c in chunks if c[1] <= p1 + 5 < c[2]]
    assert lab1["ids"][:1] == want


def test_18_duplicate_passage_label_sets_are_disjoint():
    p1, p2, chunks = _duplicate_passage_doc()
    lab1 = label_span(p1 + 5, p1 + 60, chunks, C)
    lab2 = label_span(p2 + 5, p2 + 60, chunks, C)
    assert set(lab2["ids"]) & set(lab1["ids"]) == set()


def test_19_identical_spans_give_identical_labels():
    a, b = label_span(120, 170, G, C), label_span(120, 170, G, C)
    assert (a["ids"], a["step"]) == (b["ids"], b["step"]) == (["c1"], 1)


def test_20_result_does_not_depend_on_chunk_list_order():
    shuffled = G[:]
    random.Random(1).shuffle(shuffled)
    a, b = label_span(120, 170, G, C), label_span(120, 170, shuffled, C)
    assert (a["ids"], a["step"]) == (b["ids"], b["step"])


def test_21_chunk_ends_exactly_where_span_starts():
    assert ids(100, 150, [("a", 0, 100, 0)]) == ([], 0)


def test_22_chunk_starts_exactly_where_span_ends():
    assert ids(100, 150, [("a", 150, 200, 0)]) == ([], 0)


def test_23_one_shared_character_is_an_overlap_step_3():
    assert label_span(100, 200, [("a", 99, 120, 0)], C)["step"] == 3


def test_24_span_of_one_character_inside_a_chunk():
    assert ids(100, 101, [("a", 100, 101, 0)]) == (["a"], 1)


def test_25_span_of_one_character_on_the_end_boundary():
    assert ids(100, 101, [("a", 0, 100, 0)]) == ([], 0)


def test_26_exactly_the_cover_level_counts():
    assert ids(0, 10, [("a", 0, 9, 0)]) == (["a"], 1)


def test_27_just_under_the_cover_level_goes_to_step_2():
    assert ids(0, 10, [("a", 0, 8, 0), ("b", 8, 9, 1)]) == (["a", "b"], 2)


def test_28_nine_of_eleven_characters_is_not_enough():
    assert label_span(0, 11, [("a", 0, 9, 0)], C)["step"] == 3


def test_29_no_float_rounding_at_the_boundary():
    # the audit looked for L where 0.9 * L < exact 0.9 L in floating point; the rule uses integers
    # whatever float arithmetic does. For every multiple of 10, a chunk holding exactly 90 % counts.
    for length in range(10, 2001, 10):
        r = label_span(0, length, [("a", 0, length * 9 // 10, 0)], C)
        assert (r["ids"], r["step"]) == (["a"], 1), length
        r = label_span(0, length, [("a", 0, length * 9 // 10 - 1, 0), ("b", length * 9 // 10 - 1, length * 9 // 10, 1)], C)
        assert r["step"] == 2, length


def test_30_zero_length_span():
    assert ids(5, 5, G) == ([], 0)


def test_31_zero_length_chunk_is_ignored():
    assert ids(10, 20, [("z", 15, 15, 0), ("a", 0, 100, 1)]) == (["a"], 1)


# ---- audit cases 32-34: offset recovery (see the module docstring)
PARA2 = "The sample was heated at 80 C for 2 h and then cooled to room temperature. "
DOC32 = "Start of paper. " + PARA2 + "Some other sentences follow here. " + PARA2 + "End."
C0 = DOC32[: DOC32.find("Some other") + len("Some other sentences follow here. ")].strip()
C1 = PARA2.strip()


def _locate_repeat_aware(full, chunks, overlap_frac=0.0):
    """Reference copy of the audit's repeat-aware search (previous-end lower bound)."""
    out, prev = [], None
    for t in chunks:
        if prev is None:
            lb2 = 0
        else:
            ps, pe, plen = prev
            slack = 0 if overlap_frac == 0 else int(1.5 * overlap_frac * plen) + 20
            lb2 = max(ps + 1, pe - slack)
        i = full.find(t, lb2)
        out.append(i)
        prev = (i, i + len(t), len(t))
    return out


def test_32_legacy_locate_chunk_offsets_picks_a_wrong_earlier_copy():
    pytest.importorskip("tqdm")
    from stages.chunking import locate_chunk_offsets

    head = locate_chunk_offsets(DOC32, [C0, C1])
    true_c1 = DOC32.rfind(C1)
    assert head[1][0] != true_c1  # the known limitation of the search (text matches at both places)


def test_33_repeat_aware_search_places_the_chunk_correctly():
    assert _locate_repeat_aware(DOC32, [C0, C1])[1] == DOC32.rfind(C1)


def test_34_legacy_locate_chunk_offsets_exact_for_unique_text():
    pytest.importorskip("tqdm")
    from stages.chunking import locate_chunk_offsets

    txt = " ".join(f"word{i}" for i in range(200))
    ch = [txt[0:300], txt[250:550], txt[500:800]]
    assert [(s, e) for s, e in locate_chunk_offsets(txt, ch)] == [(0, 300), (250, 550), (500, 800)]


# ---- cover level: exact fractions, validation, file names
@pytest.mark.parametrize("bad", [0, 0.0, -0.5, 1.0000001, 1.5, 2, "abc", None, float("nan"), float("inf"), True, [0.5], Fraction(1, 3)])
def test_cover_level_outside_0_1_raises(bad):
    with pytest.raises(ValueError):
        parse_cover_level(bad)


@pytest.mark.parametrize("good, frac", [(0.8, Fraction(4, 5)), ("0.8", Fraction(4, 5)), (1, Fraction(1)), (1.0, Fraction(1)),
                                        (0.875, Fraction(7, 8)), (0.01, Fraction(1, 100)), (Fraction(3, 4), Fraction(3, 4))])
def test_cover_level_is_an_exact_fraction(good, frac):
    assert parse_cover_level(good) == frac


def test_label_span_rejects_a_bad_cover_level():
    with pytest.raises(ValueError):
        label_span(0, 10, G, 1.2)
    with pytest.raises(ValueError):
        label_span(0, 10, G, 0)


@pytest.mark.parametrize("c, tag", [(0.8, "c080"), (1.0, "c100"), (0.5, "c050"), (0.875, "c087p5"), (0.01, "c001"), (0.995, "c099p5")])
def test_cover_tag(c, tag):
    assert cover_tag(c) == tag


def test_cover_tags_are_unique_per_value():
    vals = [i / 1000 for i in range(1, 1001)]
    tags = {cover_tag(v) for v in vals}
    assert len(tags) == len(vals)


def test_qrels_filename_and_resolve_labeling():
    assert qrels_filename("span", 0.8) == "qrels_span_c080.jsonl"
    assert qrels_filename("lexical") == "qrels_lexical.jsonl"
    assert resolve_labeling({"method": "span", "cover_level": 1.0, "qrels_file": None})["qrels_file"] == "qrels_span_c100.jsonl"
    assert resolve_labeling({"method": "span", "cover_level": 1.0, "qrels_file": "mine.jsonl"})["qrels_file"] == "mine.jsonl"
    assert resolve_labeling({"method": "lexical", "cover_level": 0.5})["qrels_file"] == "qrels_lexical.jsonl"
    with pytest.raises(ValueError):
        resolve_labeling({"method": "span", "cover_level": None})
    with pytest.raises(ValueError):
        resolve_labeling({"method": "span", "cover_level": 0})
    with pytest.raises(ValueError):
        resolve_labeling({"method": "bm25", "cover_level": 0.5})


def test_step_counts_move_in_the_expected_direction_with_c():
    """A single span of 100 chars over three 40-char chunks (40/40/20): the step depends on c."""
    chunks = [("a", 0, 40, 0), ("b", 40, 80, 1), ("c", 80, 120, 2)]
    steps = {c: label_span(0, 100, chunks, c)["step"] for c in (0.3, 0.4, 0.5, 0.8, 1.0)}
    assert steps == {0.3: 1, 0.4: 1, 0.5: 2, 0.8: 2, 1.0: 2}
    # a gap makes 100 % unreachable: step 3 only at the highest c
    gap = [("a", 0, 40, 0), ("b", 50, 100, 1)]
    steps = {c: label_span(0, 100, gap, c)["step"] for c in (0.5, 0.8, 0.9, 1.0)}
    assert steps == {0.5: 1, 0.8: 2, 0.9: 2, 1.0: 3}


def test_exact_boundary_chunk_holding_exactly_c_counts():
    for c, num in ((0.8, 80), (0.5, 50), (1.0, 100), (0.25, 25)):
        assert label_span(0, 100, [("a", 0, num, 0)], c)["step"] == 1
        if num > 1:
            assert label_span(0, 100, [("a", 0, num - 1, 0)], c)["step"] != 1


# ---- differential test against the scratch reference implementation of the audit (when present)
REF = ROOT / "review_samples" / "chunker_audit" / "scripts" / "rule.py"


@pytest.mark.skipif(not REF.exists(), reason="reference rule.py (git-ignored review_samples) not available")
def test_identical_to_the_reference_implementation_on_random_cases():
    pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("ref_rule", REF)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rng = random.Random(12345)
    for _ in range(20000):
        n = rng.randint(1, 400)
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n)
        chunks, a = [], 0
        k = rng.randint(1, 12)
        for order in range(k):
            lo = rng.randint(0, n)
            hi = rng.randint(lo, min(n, lo + rng.randint(0, 150)))
            chunks.append((f"c{order}", lo, hi, order))
        a = ref.label_span(s, e, chunks)
        b = label_span(s, e, chunks, C)
        assert (a["ids"], a["step"], a["reason"], a["n_overlap"]) == (b["ids"], b["step"], b["reason"], b["n_overlap"]), (s, e, chunks)
        assert a["coverage"] == b["coverage"], (s, e, chunks)

import pytest

from pipeline_lib.hierarchical_section_chunker import HierarchicalSectionChunker


def _chunker():
    # model_name=None: whitespace tokens, so the test runs offline.
    return HierarchicalSectionChunker(
        model_name=None,
        chunk_size=40,
        chunk_overlap=0,
        use_latex=False,
        use_numbered_headings=False,
        use_common_chem_headings=False,
        use_generic_heading_heuristics=False,
        custom_heading_regexes=[
            r"^\s*corresponding\s+author\s*$",
            r"^\s*introduction\s*$",
            r"^\s*results\s*$",
            r"^\s*references?\s*$",
        ],
        join_short_sections=False,
        drop_tail_sections_matching=["references", "corresponding author"],
        drop_from_first_match=True,
    )


def _doc(with_references: bool) -> str:
    # "Corresponding Author" sits in the author block (about 3% into the text), as in docs 9 and 23.
    front = "A Study of Furfural\n\nJane Doe, John Roe\n\nCorresponding Author\nJane Doe, Department of Chemistry\n\n"
    intro = "Introduction\n\n" + " ".join(f"Intro sentence {i} about furfural." for i in range(60)) + "\n\n"
    results = "Results\n\n" + " ".join(f"Result sentence {i} about yields." for i in range(60)) + "\n\n"
    refs = ("References\n\n" + " ".join(f"[{i}] Some reference." for i in range(20))) if with_references else ""
    return front + intro + results + refs


def test_front_matter_heading_does_not_drop_the_paper_body():
    # Regression: the only tail-style heading is in the front matter. The chunker used to cut there
    # and index nothing but the first ~3% of the document.
    text = _doc(with_references=False)
    assert text.index("Corresponding Author") < 0.30 * len(text)
    joined = "\n".join(_chunker().split_text(text))
    assert "Intro sentence 0 " in joined
    assert "Result sentence 59 " in joined


def test_real_end_matter_is_still_dropped():
    text = _doc(with_references=True)
    joined = "\n".join(_chunker().split_text(text))
    assert "Result sentence 59 " in joined
    assert "Some reference" not in joined

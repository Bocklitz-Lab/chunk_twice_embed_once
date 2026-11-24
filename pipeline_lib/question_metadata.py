"""Official per-question labels of ChemQuests: question type and chemistry subfields (``qa/question_metadata.jsonl`` of the dataset repository).

The file is one JSON object per question::

    {"question_id": "q0000", "corpus_id": "0", "question": "...", "question_type": "Synthetic and Experimental",
     "chemistry_subfields": ["Agriculture and Food Chemistry", "Chemical Engineering and Industrial Chemistry"]}

It is NOT part of the pinned benchmark release (``dataset_pin.CHEMQUESTS_REVISION``) and has its own pin
(``dataset_pin.QUESTION_METADATA_REVISION`` / ``QUESTION_METADATA_SHA256``); the download stage writes it next to ``qa.jsonl`` as
``question_metadata.jsonl``.

``chemistry_subfields`` belongs to the QUESTION. It is not the paper-level ``fields_of_study`` of ``metadata.jsonl`` (up to 10 per paper), and the
entries of one question are a set, not a ranking. A question carries one or two subfields.

``load`` is the only reader. It FAILS LOUDLY (``QuestionMetadataError``, no fallback, nothing normalised away) unless

  * the file exists (otherwise the error names the command that downloads it),
  * it has 952 rows and every ``question_id`` is present once,
  * ``question_id`` ``q%04d`` is the row index of ``qa.jsonl`` for every row (query id = row index of qa.jsonl),
  * ``corpus_id`` and ``question`` equal those of ``qa.jsonl`` exactly, row by row (``qa/qa.csv`` of the dataset has other question texts: do not use it),
  * ``question_type`` is one of the four types,
  * ``chemistry_subfields`` is a non-empty list of strings without repeats, and the file holds exactly the 17 subfield names of ``SUBFIELDS``.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

N_QUESTIONS = 952
QUESTION_TYPES: Tuple[str, ...] = ("Conceptual", "Mechanistic", "Applied", "Synthetic and Experimental")
# the 17 subfields of the paper, as spelled in the pinned file (sorted); a different spelling is an error, never mapped
SUBFIELDS: Tuple[str, ...] = (
    "Agriculture and Food Chemistry", "Analytical Chemistry", "Biological and Medicinal Chemistry", "Catalysis", "Chemical Education",
    "Chemical Engineering and Industrial Chemistry", "Earth, Space, and Environmental Chemistry", "Energy Chemistry", "Inorganic Chemistry",
    "Materials Chemistry", "Materials Science", "Nanoscience", "Organic Chemistry", "Organometallic Chemistry",
    "Physical Chemistry", "Polymer Science", "Theoretical and Computational Chemistry",
)
DOWNLOAD_HINT = ("python -m stages.download_chemquests --config configs/stages/stage3.yaml   "
                 "(or `make stage3`; downloads qa/question_metadata.jsonl of the pinned revision next to qa.jsonl, needs an HF token)")


class QuestionMetadataError(ValueError):
    """The question metadata is missing or does not match qa.jsonl."""


@dataclass(frozen=True)
class QuestionMetadata:
    """Labels of the questions, aligned with the rows of ``qa.jsonl`` (index = query id)."""
    question_type: Tuple[str, ...]
    subfields: Tuple[Tuple[str, ...], ...]       # per question: the set of its subfields, in file order (order carries no meaning)
    subfield_names: Tuple[str, ...]              # sorted names that occur in the file
    path: str

    def __len__(self) -> int:
        return len(self.question_type)

    def type_counts(self) -> Dict[str, int]:
        c = Counter(self.question_type)
        return {t: c.get(t, 0) for t in QUESTION_TYPES}

    def subfield_counts(self) -> Dict[str, int]:
        """Questions per subfield (multi-label: a question counts in every subfield it carries)."""
        c = Counter(s for subs in self.subfields for s in subs)
        return {s: c[s] for s in self.subfield_names}

    def subfield_fractions(self) -> Dict[str, float]:
        """Questions per subfield with weight 1/len(subfields) per question: the values add up to the number of questions."""
        out = {s: 0.0 for s in self.subfield_names}
        for subs in self.subfields:
            for s in subs:
                out[s] += 1.0 / len(subs)
        return out

    def n_labels_counts(self) -> Dict[int, int]:
        return dict(sorted(Counter(len(s) for s in self.subfields).items()))


def _read_jsonl(path: Path, what: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise QuestionMetadataError(f"{what} line {n}: not valid JSON ({e})") from e
    return rows


def load(path: Any, qa_jsonl: Any, *, n_questions: int = N_QUESTIONS, subfields: Sequence[str] = SUBFIELDS,
         question_types: Sequence[str] = QUESTION_TYPES) -> QuestionMetadata:
    """Read and check the metadata file against ``qa_jsonl`` (see the module docstring). The keyword arguments exist for tests on small fixtures."""
    path, qa_path = Path(path), Path(qa_jsonl)
    if not path.is_file():
        raise QuestionMetadataError(f"question metadata not found: {path}\nThe question type and the chemistry subfields come only from this file; "
                                    f"nothing is guessed from the question text. Download it with:\n  {DOWNLOAD_HINT}")
    if not qa_path.is_file():
        raise QuestionMetadataError(f"qa.jsonl not found: {qa_path}")
    rows, qa = _read_jsonl(path, str(path)), _read_jsonl(qa_path, str(qa_path))
    if len(rows) != n_questions:
        raise QuestionMetadataError(f"{path}: {len(rows)} rows, expected {n_questions}")
    if len(qa) != len(rows):
        raise QuestionMetadataError(f"{path}: {len(rows)} rows but {qa_path} has {len(qa)}")
    ids = [r.get("question_id") for r in rows]
    dup = sorted(k for k, v in Counter(ids).items() if v > 1)
    if dup:
        raise QuestionMetadataError(f"{path}: duplicated question_id {dup[:10]}")
    want = [f"q{i:04d}" for i in range(len(qa))]
    missing = sorted(set(want) - set(ids))
    if missing:
        raise QuestionMetadataError(f"{path}: question_id missing {missing[:10]} ({len(missing)} in all)")
    shifted = [(i, ids[i]) for i in range(len(rows)) if ids[i] != want[i]]
    if shifted:
        raise QuestionMetadataError(f"{path}: question_id is not the row index of qa.jsonl in {len(shifted)} rows, e.g. row {shifted[0][0]} has "
                                    f"{shifted[0][1]!r}, expected {want[shifted[0][0]]!r}")
    bad_corpus = [i for i, (r, q) in enumerate(zip(rows, qa)) if str(r.get("corpus_id")) != str(q.get("corpus_id"))]
    if bad_corpus:
        i = bad_corpus[0]
        raise QuestionMetadataError(f"{path}: corpus_id differs from qa.jsonl in {len(bad_corpus)} rows, e.g. row {i}: {rows[i].get('corpus_id')!r} vs "
                                    f"{qa[i].get('corpus_id')!r}")
    bad_text = [i for i, (r, q) in enumerate(zip(rows, qa)) if r.get("question") != q.get("question")]
    if bad_text:
        shown = "\n".join(f"  row {i}: metadata {rows[i].get('question')!r}\n         qa.jsonl {qa[i].get('question')!r}" for i in bad_text[:10])
        raise QuestionMetadataError(f"{path}: question text differs from qa.jsonl in {len(bad_text)} rows (exact comparison, not normalised); first ones:\n{shown}")
    types = []
    for i, r in enumerate(rows):
        t = r.get("question_type")
        if t not in question_types:
            raise QuestionMetadataError(f"{path}: row {i} ({r['question_id']}): question_type {t!r} is missing or not one of {list(question_types)}")
        types.append(t)
    subs: List[Tuple[str, ...]] = []
    for i, r in enumerate(rows):
        s = r.get("chemistry_subfields")
        if not isinstance(s, list) or not s:
            raise QuestionMetadataError(f"{path}: row {i} ({r['question_id']}): chemistry_subfields must be a non-empty list, got {s!r}")
        if not all(isinstance(x, str) and x for x in s) or len(set(s)) != len(s):
            raise QuestionMetadataError(f"{path}: row {i} ({r['question_id']}): chemistry_subfields must be distinct non-empty strings, got {s!r}")
        subs.append(tuple(s))
    found = sorted({x for s in subs for x in s})
    if len(found) != len(subfields):
        raise QuestionMetadataError(f"{path}: {len(found)} distinct subfields, expected {len(subfields)} (the paper's list); found: {found}")
    outside = sorted(set(found) - set(subfields))
    if outside:
        raise QuestionMetadataError(f"{path}: subfield names not in the expected list of {len(subfields)}: {outside}; expected names not found: "
                                    f"{sorted(set(subfields) - set(found))} (names are not mapped or merged)")
    return QuestionMetadata(question_type=tuple(types), subfields=tuple(subs), subfield_names=tuple(found), path=str(path))

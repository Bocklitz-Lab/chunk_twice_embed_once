"""Hand check of the automatic features (question text: ``question_features_check.csv``; evidence span: ``span_features_check.csv``, the same two rules). The two automatic question features ``auto_has_number`` and ``auto_has_formula`` (``pipeline_lib/question_features.py``).

The features are rule-based, so before any feature breakdown is reported as final, a person must check the rules on a seeded sample of questions.

  draw   writes ``question_features_check.csv``: a seeded sample of 50 questions (``params.analysis.question_features_check``) in which, FOR EACH feature,
         25 questions are flagged and 25 are not flagged; one question can count for both features (overlap allowed), so the sheet has 50 rows. The sheet
         holds only the question text and EMPTY columns ``manual_has_number`` and ``manual_has_formula`` (fill in 1 or 0, also yes / no); it does not show
         the automatic flags, so the person is not guided by them. Fill in: 1 if the question contains a number or a number with a unit (manual_has_number),
         1 if it contains a chemical formula, i.e. a molecule, ion or material written with element symbols such as H2O, NaCl, Fe(II), TiO2; not an
         abbreviation such as UV, NMR, PVC, and not an element name or a word (manual_has_formula).
  score  reads the filled sheet and reports, per feature, the counts (TP, FP, FN, TN), precision, recall and agreement of the automatic flag with the manual
         one. The automatic flags are recomputed from the question text with the current rules. With ``--qa`` it also reports precision and recall
         WEIGHTED to all 952 questions (the sample takes 25 flagged and 25 not flagged questions, so the raw recall over-represents the unflagged ones).

  python -m tools.question_features_check draw  --qa artifacts/chemquests_hf_db/qa.jsonl --out-dir results/validation
  python -m tools.question_features_check score --sheet results/validation/question_features_check.csv --qa artifacts/chemquests_hf_db/qa.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_lib import config_params  # noqa: E402
from pipeline_lib import question_features as QF  # noqa: E402

FEATURES = ("number", "formula")
SHEET_COLUMNS = ["query_id", "question", "manual_has_number", "manual_has_formula"]
TRUE, FALSE = {"1", "yes", "y", "true"}, {"0", "no", "n", "false"}
nan = float("nan")


def read_questions(qa: Any) -> List[str]:
    return [json.loads(l)["question"] for l in open(qa, encoding="utf-8") if l.strip()]


SPAN_SHEET_COLUMNS = ["query_id", "span_text", "manual_span_has_number", "manual_span_has_formula"]


def flags(text: str, kind: str = "question") -> Dict[str, bool]:
    """Automatic flags of a question text (kind ``question``) or of an evidence span (kind ``span``; same rules)."""
    if kind == "span":
        f = QF.span_features(text)
        return {"number": f.auto_span_has_number, "formula": f.auto_span_has_formula}
    f = QF.question_features(text)
    return {"number": f.auto_has_number, "formula": f.auto_has_formula}


def read_spans(qa: Any) -> List[str]:
    rows = [json.loads(l) for l in open(qa, encoding="utf-8") if l.strip()]
    return QF.span_texts(qa, rows)


def draw_sample(questions: Sequence[str], n: int, seed: int, kind: str = "question") -> List[int]:
    """Query ids of ``n`` questions (n even) with n/2 flagged and n/2 not flagged for EACH feature.

    Four kinds of question (both flags, number only, formula only, neither): with a questions of both and a of neither, and b = n/2 - a of each single
    kind, both features have n/2 flagged and n/2 not flagged. ``a`` is as close to n/4 as the data allow. Seeded; the result is in a shuffled order."""
    if n % 2:
        raise ValueError("n must be even")
    half = n // 2
    kinds: Dict[tuple, List[int]] = {}
    for i, q in enumerate(questions):
        f = flags(q, kind)
        kinds.setdefault((f["number"], f["formula"]), []).append(i)
    both, only_n, only_f, neither = (kinds.get(k, []) for k in ((True, True), (True, False), (False, True), (False, False)))
    for a in sorted(range(half + 1), key=lambda a: (abs(a - half / 2), a)):
        b = half - a
        if a <= len(both) and a <= len(neither) and b <= len(only_n) and b <= len(only_f):
            break
    else:
        raise ValueError(f"cannot draw {half} flagged and {half} not flagged questions per feature: both={len(both)}, number only={len(only_n)}, "
                         f"formula only={len(only_f)}, neither={len(neither)}")
    rng = random.Random(seed)
    ids = rng.sample(both, a) + rng.sample(neither, a) + rng.sample(only_n, b) + rng.sample(only_f, b)
    rng.shuffle(ids)
    return ids


def write_sheet(path: Any, questions: Sequence[str], ids: Sequence[int], kind: str = "question") -> Path:
    """The sheet to fill in. The automatic flags are NOT in it (hidden). kind ``span``: the evidence spans instead of the questions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SPAN_SHEET_COLUMNS if kind == "span" else SHEET_COLUMNS)
        w.writeheader()
        for i in ids:
            if kind == "span":
                w.writerow({"query_id": i, "span_text": questions[i], "manual_span_has_number": "", "manual_span_has_formula": ""})
            else:
                w.writerow({"query_id": i, "question": questions[i], "manual_has_number": "", "manual_has_formula": ""})
    return path


def parse_answer(value: str, where: str) -> bool:
    v = str(value).strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError(f"{where}: expected 1 / 0 (or yes / no), got {value!r}" if v else f"{where}: empty; the sheet is not filled in yet")


def _ratio(a: float, b: float) -> float:
    return a / b if b else nan


def score(sheet: Any, qa: Optional[Any] = None, kind: str = "question") -> List[Dict[str, Any]]:
    """One row per feature: counts, precision, recall, agreement (raw, on the sample) and, with ``qa``, precision and recall weighted to all questions."""
    rows = list(csv.DictReader(open(sheet, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"{sheet}: no rows")
    texts = (read_spans if kind == "span" else read_questions)
    population = [flags(q, kind) for q in texts(qa)] if qa else None
    text_col, prefix = ("span_text", "manual_span_has_") if kind == "span" else ("question", "manual_has_")
    out = []
    for feat in FEATURES:
        col = f"{prefix}{feat}"
        pairs = []
        for r in rows:
            pairs.append((flags(r[text_col], kind)[feat], parse_answer(r.get(col, ""), f"{sheet} query {r.get('query_id')} {col}")))
        tp = sum(a and m for a, m in pairs); fp = sum(a and not m for a, m in pairs)
        fn = sum((not a) and m for a, m in pairs); tn = sum((not a) and (not m) for a, m in pairs)
        res = {"feature": f"auto_{'span_' if kind == 'span' else ''}has_{feat}", "feature_source": "rule_based", "n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn), "agreement": _ratio(tp + tn, len(pairs)),
               "weighted_precision": nan, "weighted_recall": nan}
        if population is not None:
            n_flag = sum(p[feat] for p in population)
            n_not = len(population) - n_flag
            s_flag, s_not = tp + fp, fn + tn
            if s_flag and s_not:                                        # stratum weights: population size / sample size of each automatic stratum
                wf, wn = n_flag / s_flag, n_not / s_not
                res["weighted_precision"] = _ratio(wf * tp, wf * (tp + fp))
                res["weighted_recall"] = _ratio(wf * tp, wf * tp + wn * fn)
        out.append(res)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="write the sheet to fill in")
    d.add_argument("--qa", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--n", type=int, help="default: params.analysis.question_features_check.n_questions")
    d.add_argument("--seed", type=int, help="default: params.analysis.question_features_check.seed")
    d.add_argument("--main-config", default=str(config_params.DEFAULT_MAIN_CONFIG))
    s = sub.add_parser("score", help="precision, recall and agreement from the filled sheet")
    s.add_argument("--sheet", required=True)
    s.add_argument("--qa", help="qa.jsonl: also give precision and recall weighted to all questions")
    s.add_argument("--kind", choices=["question", "span"], default="question", help="which sheet: the questions (question_features_check.csv) or the evidence spans (span_features_check.csv)")
    s.add_argument("--out", help="CSV of the scores (default: question_features_check_scores.csv next to the sheet)")
    args = ap.parse_args(argv)
    if args.cmd == "draw":
        cfg = config_params.section("analysis", args.main_config).get("question_features_check") or {}
        n = args.n or int(cfg["n_questions"])
        seed = args.seed if args.seed is not None else int(cfg["seed"])
        questions = read_questions(args.qa)
        ids = draw_sample(questions, n, seed)
        path = write_sheet(Path(args.out_dir) / "question_features_check.csv", questions, ids)
        print(f"{len(ids)} questions (seed {seed}) -> {path}; fill in manual_has_number and manual_has_formula, then run `score`")
        spans = read_spans(args.qa)
        n_sp = int(cfg.get("n_spans", n))
        sids = draw_sample(spans, n_sp, seed, "span")
        spath = write_sheet(Path(args.out_dir) / "span_features_check.csv", spans, sids, "span")
        print(f"{len(sids)} evidence spans (seed {seed}) -> {spath}; fill in manual_span_has_number and manual_span_has_formula, then run `score --kind span`")
        return 0
    res = score(args.sheet, args.qa, args.kind)
    out = Path(args.out) if args.out else Path(args.sheet).with_name(("span" if args.kind == "span" else "question") + "_features_check_scores.csv")
    cols = list(res[0])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in res:
        print(f"{r['feature']}: n={r['n']} TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']} precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"agreement={r['agreement']:.3f} (weighted to all questions: precision={r['weighted_precision']:.3f} recall={r['weighted_recall']:.3f})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

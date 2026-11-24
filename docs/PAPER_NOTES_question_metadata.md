# Paper notes: question type, subfields and automatic question features

For the authors. This note replaces section AA of the paper-change list (Reviewer comment 5 of the Discover Chemistry review: breakdowns by the 17 subfields and by
question category). Everything below is produced by `tools/subfield_breakdown.py` (`make analysis`) from the official question labels.

## What the paper text must now say

1. **Question types are dataset labels, not heuristics.** Every question carries one of four official types: Conceptual, Mechanistic, Applied, Synthetic and Experimental
   (file `qa/question_metadata.jsonl` of `Bocklitz-Lab/ChemQuests`). The types are exclusive. Nothing is inferred from the question text.
   Counts in the 952 questions (checked on the pinned file): 227 / 238 / 243 / 244, i.e. 23.8 / 25.0 / 25.5 / 25.6 %. The dataset card gives 23.7 / 24.9 / 25.5 / 25.7 %
   (and calls the last type "Experimental/Synthetic"): the shares differ by at most 0.1 percentage point from the counts. Please state the counts, or ask the dataset maintainers which numbers the card was built from.
2. **Subfields are per question, not per paper.** Each question has one or two of the 17 subfields (226 questions have one, 726 have two); the two names are a set, not a ranking. The paper-level
   `fields_of_study` of `metadata.jsonl` (up to 10 per paper) is not used. Every one of the 952 questions has at least one subfield (the "unlabelled" group no longer exists).
3. **Two counting rules, both reported.** `multi_label`: a question counts fully in each of its subfields (groups overlap, the sizes add up to 1,678, not 952).
   `fractional`: weight 1/(number of subfields) in each, weighted means and a weighted bootstrap, so the weights add up to 952. Say which one a table uses; the two are in the same CSV (`label_rule`).
   Question counts per subfield (multi-label): Theoretical and Computational Chemistry 181, Materials Chemistry 176, Physical Chemistry 162, Inorganic Chemistry 156, Biological and Medicinal Chemistry 115, Organic Chemistry 114,
   Analytical Chemistry 104, Earth, Space, and Environmental Chemistry 90, Nanoscience 89, Energy Chemistry 79, Catalysis 70, Chemical Engineering and Industrial Chemistry 69, Materials Science 69, Polymer Science 62,
   Agriculture and Food Chemistry 56, Chemical Education 46, Organometallic Chemistry 40.
4. **Pinned revision.** The benchmark data stay at release `5ab7dc673653b800f368fe30851efbdb24385e0d` (`qa.jsonl` SHA-256 `e66ced49...`). The label file is not in that release; it was added in
   dataset commit `790d2bc6829ccd7e1e480a6a4610149650b79e9f` (SHA-256 of the file `4e1f83fb...`), the only commit that holds it. That commit differs from the release only by this file: `qa.jsonl`, `metadata.jsonl` and all
   `full_text/<id>.txt` have the same git blob ids, so no span, chunk or label moves. `question_id` `q%04d` equals the row index of `qa.jsonl`, and `corpus_id` and the question text of all 952 rows equal `qa.jsonl`.
   Cite both revisions.
5. **Two automatic features, rule-based, not annotations and not question types.** `auto_has_number` (the question contains a number or a number with a unit) and `auto_has_formula` (it contains a chemical formula) answer
   Reviewer 2 comment 4 (texts with formulas and numerical evidence), for which the dataset has no label. They are computed from the question text with the rules in `pipeline_lib/question_features.py` (describe them in the supplement:
   what counts, the exclusion lists for abbreviations such as UV, IR, NMR, PVC, and the known errors, e.g. a label such as "compound 1" counts as a number, "S3 state" counts as a formula). They were checked by hand on 50 questions
   (25 flagged and 25 not flagged for each feature, seed 20261121, `results/validation/question_features_check.csv`); **give precision, recall and agreement from `tools/question_features_check.py score` once the sheet is filled in.**
   Until then no feature breakdown is final. Never call them "question types" or "categories", and do not put them in the same table column as the official type.
6. **Statistics.** Intervals are paper-level bootstrap percentile intervals (papers of the group are resampled, 10,000 resamples). Differences between groups and between runs are paired over the papers; p-values are Holm-corrected
   within the 4 types, within the 6 type pairs, within the 17 subfields of one counting rule and within the groups of one feature. A group or cell with fewer than 30 questions has numbers and intervals but no test.
   The interaction "does dense - BM25 differ between question types" is in `breakdown_question_type_paired.csv` (`row_kind = contrast`).

## Earlier statements that are now wrong

* "`qa.jsonl` has no type field, the code tags the questions by rules" (question word, yes/no, number or unit, chemical formula). The dataset has an official `question_type`; the rule-based tags (`tag:qword:*`, `tag:yes_no`, ...) are removed.
* "A paper counts in every subfield listed in its `fields_of_study`." Subfields are the labels of the question; the paper-level list is not used.
* "Unlabelled: 1 paper, 8 questions." All 952 questions have at least one subfield; there is no unlabelled group.
* The pilot counts of the heuristic tags ("752 what, 158 how, ..."), and "210 questions with a number and 164 with a formula". With the new module the counts on the 952 questions are **65 with `auto_has_number` and 102 with
  `auto_has_formula`** (the old "number or unit" rule counted any digit, including those in formulas, and its formula rule treated acronyms as formulas). These two counts come from the automatic rules and must be checked by hand
  (see item 5) before they are quoted.
* "Breakdown by tag" / `breakdown_tag.csv`: the file is now `breakdown_question_type.csv` (official types) plus `breakdown_question_features.csv` (automatic features).

## Open points to confirm

* The dataset card's type shares (23.7 / 24.9 / 25.5 / 25.7 %) do not reproduce exactly from the counts in the file (23.8 / 25.0 / 25.5 / 25.6 %).
* Whether "Synthetic and Experimental" (file) or "Experimental/Synthetic" (card) is the name used in the paper.
* The reported unit of a breakdown cell: questions or papers (a group's questions come from 40-100 papers; the largest share of one paper is in `breakdown_groups.csv`).

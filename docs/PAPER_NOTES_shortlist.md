# Paper notes: the Phase 2 model list (shortlist rule)

Written for: the authors, to rewrite Methods, Results and the supplement. Decision of the authors: Phase 2 uses the shortlist of the **prompted** Phase 1
(model-card prompts, instructor-xl instruction masked, split `test`), selected by a rule that is stated in advance (config `params.phase1.selection`).

## Methods text (one paragraph, for `main:122`)

> Phase 1 screened 41 embedding models on the test split of ChemNQRetrieval (27 queries) and ChemHotpotQARetrieval (18 queries) with Geom@10, using the model
> card's query and document prompts. Because such small query sets make the rank of a model uncertain, the Phase 2 model list was fixed by a rule stated before the
> results were examined: in each of 10,000 query-level bootstrap resamples the 41 models were ranked again on each task, and a model was retained if it was in the
> top 20 % (the top eight of 41) of at least one task in at least 50 % of the resamples; the eight best models of each task by the observed Geom@10 were always
> retained, and the union over the two tasks was taken. [Result: 13 models.]

## Why it replaces the fixed top 20 %

* With prompts, the Phase 1 numbers of the published run change (all values +0.02 to +0.07; the BGE models were evaluated without their query instruction before).
* The test splits have 27 and 18 queries. In the query-level bootstrap the size of the union of the two top-eight sets is 13.7 [12, 16], and about eight models have an
  inclusion probability near 0.5 on one task (`shortlist_selection__validation_prompts.csv`). A hard cut at rank 8 therefore decides on differences that are smaller than the
  sampling error (Reviewer 2: no post-hoc selection).
* The rule keeps the published definition of "top" (top 20 %, union over tasks) and only adds the inclusion probability, so it can only widen the list, never narrow it.
* ChemHotpotQARetrieval has 180 train and 8 dev queries on the same corpus; they are not used (pooling them is an optional, cheap way to reduce the instability: the per-split
  scores are in `ChemHotpotQARetrieval.json`).

## Outcome (prompted Phase 1, 10,000 resamples, seed 20261121)

| model | rank Hotpot | P(top 8) Hotpot | rank NQ | P(top 8) NQ | published rule (top 8 per task) | P ≥ 0.4 | **P ≥ 0.5 (primary)** | P ≥ 0.6 |
|---|---|---|---|---|---|---|---|---|
| nomic-ai/nomic-embed-text-v1 | 1 | 0.97 | 7 | 0.57 | yes | yes | **yes** | yes |
| nomic-ai/nomic-embed-text-v2-moe | 19 | 0.06 | 1 | 0.96 | yes | yes | **yes** | yes |
| BAAI/bge-large-en-v1.5 | 4 | 0.95 | 13 | 0.10 | yes | yes | **yes** | yes |
| nomic-ai/nomic-embed-text-v1.5 | 2 | 0.95 | 6 | 0.76 | yes | yes | **yes** | yes |
| intfloat/e5-large-v2 | 15 | 0.10 | 2 | 0.95 | yes | yes | **yes** | yes |
| intfloat/multilingual-e5-large | 21 | 0.00 | 3 | 0.88 | yes | yes | **yes** | yes |
| intfloat/multilingual-e5-base | 18 | 0.00 | 5 | 0.87 | yes | yes | **yes** | yes |
| intfloat/multilingual-e5-small | 3 | 0.79 | 4 | 0.77 | yes | yes | **yes** | yes |
| BAAI/bge-base-en | 6 | 0.69 | 20 | 0.03 | yes | yes | **yes** | yes |
| BAAI/bge-small-en-v1.5 | 5 | 0.65 | 27 | 0.00 | yes | yes | **yes** | yes |
| BAAI/bge-large-en | 7 | 0.59 | 18 | 0.03 | yes | yes | **yes** | yes |
| BAAI/bge-m3 | 9 | 0.48 | 16 | 0.08 | - | yes | **-** | - |
| BAAI/bge-base-en-v1.5 | 11 | 0.46 | 12 | 0.16 | - | yes | **-** | - |
| intfloat/e5-small | 8 | 0.45 | 14 | 0.10 | yes | yes | **yes** | yes |
| intfloat/e5-base-v2 | 17 | 0.06 | 9 | 0.40 | - | yes | **-** | - |
| intfloat/e5-base | 12 | 0.26 | 8 | 0.40 | yes | yes | **yes** | yes |
| BAAI/bge-small-en | 10 | 0.38 | 25 | 0.01 | - | - | **-** | - |
| intfloat/e5-small-v2 | 16 | 0.02 | 10 | 0.35 | - | - | **-** | - |
| intfloat/e5-large | 12 | 0.29 | 11 | 0.31 | - | - | **-** | - |

Models selected: published rule 13, P ≥ 0.4 16, P ≥ 0.5 13, P ≥ 0.6 13 (of 41). Models with a maximum inclusion probability below 0.2 that are not observed-top are omitted from this table (full table: `shortlist_selection__validation_prompts.csv`).

**The primary rule (P >= 0.5) selects the same 13 models as the observed top-eight union.** At 0.5 nothing is added: the models next to the cut
(bge-m3 0.48, bge-base-en-v1.5 0.46, e5-base-v2 0.40) stay just below the threshold on their better task. The threshold was NOT changed to make any model enter or leave. A threshold of 0.4 would add
bge-m3, bge-base-en-v1.5 and e5-base-v2 (16 models). The authors decide whether to use 0.4 (a change of the pre-stated rule, to be reported as such) or to keep 0.5 and test
these three models in the transfer set (the config does this for bge-m3 and bge-base-en-v1.5; e5-base-v2 is not in the transfer set yet).

## Models that leave and enter, compared with the published shortlist (no prompts)

* leave (published 13, not in the new 13): `BAAI/bge-base-en-v1.5`, `BAAI/bge-m3`, `hkunlp/instructor-xl`, `intfloat/e5-large`.
* enter: `BAAI/bge-base-en`, `BAAI/bge-large-en`, `intfloat/multilingual-e5-large`, `nomic-ai/nomic-embed-text-v2-moe`.
* the 9 that stay: `BAAI/bge-large-en-v1.5`, `BAAI/bge-small-en-v1.5`, `intfloat/e5-base`, `intfloat/e5-large-v2`, `intfloat/e5-small`, `intfloat/multilingual-e5-base`,
  `intfloat/multilingual-e5-small`, `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v1.5`.
* the four that leave are in the transfer set (group `f`, role `non_shortlisted`), so the paper can show their Phase 2 performance at C1-C6.
* the published no-prompt Phase 1 and its shortlist go to the supplement for comparison (the files are `artifacts/embedding_models_screening/validation/` and
  `passed_per_dataset.csv`; `python -m stages.screen_top_k_models` with `rule: top_fraction` reproduces the published 13: tested in `tests/test_shortlist.py`).

## Places in the paper that name the number or the list

| place | text | action |
|---|---|---|
| `main_springer.tex:54` | Figure 1 caption: "... shortlist of 13 candidate embedding models" | name the rule; the number stays 13 only if the primary rule is kept |
| `main_springer.tex:70` | Introduction: "practical shortlist of strong ... candidates" | add "selected by a bootstrap inclusion rule" |
| `main_springer.tex:122` | "retained the top 20 % ... top eight models per dataset ... union" | replace by the Methods paragraph above |
| `main_springer.tex:126`, `:198-201` | "13-model shortlist, with only three models appearing in both top-eight sets" | new numbers; "three in both top-eight sets" stays true (same three) |
| `main_springer.tex:170-195` | Table 1 (top five per task, published values) | replace by the prompted values (all change); nomic-v1.5 is rank 2 (Hotpot) and 6 (NQ), not "top tier of both" |
| `main_springer.tex:212, 216` and Fig. 7 | models of the heatmap | new list; `tools/model_strategy.py` replaces the maxima |
| `main_springer.tex:260-273` | limitations / future work "sanity checks with additional non-shortlisted models" | now done (transfer set); say so |
| Supplementary Table S2 (`supplementary_materials.tex:343`, `tab:supp_phase1_shortlist_models`) | the final shortlist | replace; add the inclusion-probability column |
| Supplementary `:404` | "The resulting 13-model shortlist" | rule and number |
| Supplementary Algorithm S2 (`:509`) | "save per-task retained models and the union shortlist" | add the bootstrap step |
| Supplementary Figures S2-S6 (41-model profiles, redundancy, PCA) | Phase 1 without prompts | redraw from the prompted run (or label them "published run, no prompts") |
| Supplementary Table S1 | model table | add prompt, pooling (with the override), dimension, maximum length |

## Decisions of the authors (2026-11-25) and supplement material

* Threshold kept at **P >= 0.5** (13 models, the same as the observed top-eight union).
* Selection table in git: `results/v2-revision/shortlist_selection.csv` (all 41 models: Geom@10 and rank per task, inclusion probability per task, the four rules, published vs new
  shortlist, stays / leaves / enters) and the supplement table `results/v2-revision/table_shortlist_supplement.tex` (label `tab:supp_shortlist_selection`);
  regenerate with `python -m tools.shortlist_table`.
* SPECTER pooling: the paper (Cohan et al. 2020, arXiv 2004.07180, Sect. 3) is the source of the CLS override. The supplement reports both values
  (`results/v2-revision/table_specter_pooling.csv`): ChemNQ Geom@10 0.169 (mean) -> 0.343 (CLS); ChemHotpot 0.380 (mean) -> 0.206 (CLS). The shortlist does not change.
* Phase 1 stays on the `test` split. The new strategy is called `fixed_token_nomatter`. Reranking uses both first stages (dense e5-large-v2 and BM25).

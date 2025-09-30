# README — Composite Retrieval Loss for Chunking Sweeps

This module scans experiment folders for MTEB-style results (e.g., `ChemQuest.json`), parses run metadata from folder names (model, chunker, chunk size `c`, overlap `o`), and computes a **single scalar loss** per run that balances retrieval quality, coverage, robustness, and cost. It’s designed to help you quickly pick the best `(L, O)` (chunk size and overlap) for a fixed embedding model and chunker (e.g., **E5 + recursive token** chunking).

---

## What this tool does

* Recursively finds result files (default: `ChemQuest.json`).
* Extracts key retrieval metrics (nDCG/MAP/MRR/Recall/Precision, NAUC).
* Computes a **composite loss**:

  $$
  \mathcal{L}=0.45(1-\text{nDCG@10}) + 0.20(1-\text{MAP@10}) + 0.10(1-\text{MRR@10}) + 0.10(1-\text{Recall@20}) + 0.05\,d_p + 0.05\,L_{\text{rob}} + 0.05\,L_{\text{cost}}
  $$

  with

  * $d_p = 1 - \min\!\left(1,\frac{\text{P@10}}{\text{P@1}+\epsilon}\right)$ (precision decay),
  * $L_{\text{rob}} = 1 - \min\!\left(1,\frac{\max(0,\text{NAUC}_{\text{nDCG@10,max}} - 0.5|\text{NAUC}_{\text{nDCG@10,std}}|)}{0.2}\right)$,
  * $L_{\text{cost}} = \max(0, \text{cost\_norm}-1)$ (optional, from chunk counts).
* Ranks runs by **lower is better**.
* Prints a pretty table or writes a CSV with loss **and** the raw metrics so you can debug what’s driving the score.

---

## Why these metrics?

Think “Does the retriever bring the **right evidence**, **not just one item**, **without junk/duplicates**, **consistently**, and **without blowing up cost**?”

* **nDCG@10 (45%)** — *Are the most relevant chunks near the top?*
  RAG readers usually consume a small top-k; nDCG emphasizes ranking and graded relevance in that head. We weight it highest because head quality dominates downstream answer quality.

* **MAP@10 (20%)** — *How strong is quality across the whole top-10?*
  Complements nDCG with precision averaged over hits; smooths variance and avoids over-optimizing only rank-1.

* **MRR@10 (10%)** — *How quickly do we hit the first solid chunk?*
  Guarantees at least one highly relevant chunk appears early (critical for faithfulness and stability).

* **Recall@20 (10%)** — *Did we retrieve enough of the relevant material at all?*
  Checks **coverage** slightly deeper than the reader’s typical top-k (often 4–8). This supports reranking/Fusion and multi-evidence questions. If your reader only ever uses k≤6, you can swap to Recall@10.

* **Precision decay (P@1 vs P@10, 5%)** — *Does quality collapse as we go down?*
  Penalizes tails full of near-duplicates (often due to large overlap/small stride) or noisy results.

* **Robustness via NAUC (5%)** — *Is performance stable to cutoff changes?*
  Uses the NAUC (normalized AUC) terms around nDCG@10 to reward curves that hold up when k shifts a bit (better generalization).

* **Cost/latency (5%, optional)** — *How expensive is this config vs a baseline?*
  Small strides/large overlaps inflate index size, memory, and retrieval time. We convert **relative chunk count** into a mild penalty.

> These terms together reflect what good RAG retrieval needs: **strong head**, **breadth for multi-facts**, **non-redundancy**, **stability**, and **reasonable cost**.

---

## Why these coefficients?

* **0.45 (nDCG@10)**: Head ranking quality is the top driver of downstream RAG performance.
* **0.20 (MAP@10)**: Stabilizes head performance and avoids single-rank myopia.
* **0.10 (MRR@10)**: Ensures an early strong hit (fast evidence).
* **0.10 (Recall@20)**: Enforces coverage for multi-evidence/longer answers.
* **0.05 (Precision decay)**: Just enough to discourage overlap-induced duplication without dominating quality.
* **0.05 (Robustness)**: Rewards settings that generalize across cutoffs.
* **0.05 (Cost)**: A tie-breaker toward cheaper configs; keep small so we don’t sacrifice quality for cost.

These are **sane defaults**. If your product constraints differ (e.g., ultra-low latency or tiny memory), increase the **cost** weight; if you never rerank, reduce the **recall** weight; if you use a reader with tiny context, increase the **MRR** and decrease the **recall** weight.

---

## Installation

Python 3.9+ recommended.

```bash
pip install -r requirements.txt  # (none strictly required; stdlib only)
```

The script is self-contained and relies on Python’s standard library.

---

## File/folder expectations

* Results: MTEB-style JSON files (default filename: `ChemQuest.json`) with `scores.test` containing metrics, e.g.:

  ```json
  {
    "scores": {
      "test": [
        { "ndcg_at_10": 0.15993, "map_at_10": 0.08414, "mrr_at_10": 0.305819, "recall_at_20": 0.28613, "precision_at_1": 0.17017, "precision_at_10": 0.13466, "nauc_ndcg_at_10_max": 0.149043, "nauc_ndcg_at_10_std": 0.042103, ... }
      ]
    }
  }
  ```

* Folder names encode `(chunker, c, o)`; examples:

  ```
  all_MiniLM_L6_v2_recursive_token_c128_o32/
  e5_small_recursive_token_c320_o16/
  ```

  The parser looks for one of:

  ```
  fixed_token | recursive_token | semantic_fixed | semantic_recursive | hierarchical_section | hybrid_multi
  ```

  and extracts `_c{int}` and `_o{int}` if present.

---

## Usage

**Basic (no cost term):**

```bash
python compare_runs.py --root path/to/experiments --pattern ChemQuest.json
```

**Write CSV:**

```bash
python compare_runs.py --root path/to/experiments --csv out/loss_table.csv
```

**Add cost normalization (optional):**

```bash
python compare_runs.py \
  --root path/to/experiments \
  --doc-token-counts corpus_token_counts.txt \
  --baseline-c 256 --baseline-o 32 \
  --csv out/loss_table.csv
```

* `corpus_token_counts.txt`: one integer per line (tokenized doc length).
* Cost term is off unless all three (`--doc-token-counts`, `--baseline-c`, `--baseline-o`) are provided.

---

## Output

### Console table

* Sorted by **Loss** ascending (best first).
* Shows loss components and raw metrics so you can spot why a run wins/loses.

Columns:

* `Model`, `Chunker`, `c`, `o`
* `Loss` (total)
* `nDCG@10`, `MAP@10`, `MRR@10`, `R@20`
* `P@1`, `P@10`
* `Decay` (precision decay), `Rob` (robustness penalty), `CostN` (cost_norm)
* `Rel Path`

### CSV

* Same info plus individual component losses:

  * `loss_ndcg`, `loss_map`, `loss_mrr`, `loss_rec`, `loss_decay`, `loss_rob`, `loss_cost`
* `cost_norm` is the relative chunk count vs baseline (1.0 = same cost).

---

## Cost normalization details (optional)

For a doc of length $T$, chunk size $L$, and overlap $O$, approximate chunk count:

$$
\#\text{chunks}(T;L,O) \approx 1 + \max\!\left(0,\left\lceil\frac{T-L}{L-O}\right\rceil\right)
$$

We compute

$$
\text{cost\_norm} = \frac{\sum_d \#\text{chunks}(T_d;L,O)}{\sum_d \#\text{chunks}(T_d;L_0,O_0)}
$$

and penalize only growth beyond baseline:

$$
L_{\text{cost}} = \max(0, \text{cost\_norm} - 1).
$$

This lightly prefers cheaper configs **when quality is similar**.

---

## Interpreting results

* **Loss down ⇢ better.** If two configs are close (<0.01), prefer the one with **smaller overlap** (cheaper, less redundancy).
* High **Decay** ⇒ reduce overlap or increase stride.
* Low **Recall@20** but decent **nDCG@10** ⇒ consider a slightly **larger chunk size**.
* Poor **Robustness** ⇒ performance is brittle to k; try nearby `c/o` or review index params.

---

## Adapting the loss (advanced)

You can tune weights to product needs:

* Tiny reader window ⇒ increase **MRR** weight, reduce **Recall**.
* Heavy latency constraints ⇒ increase **Cost** weight.
* Reranking downstream ⇒ keep **Recall@20** weight or even up-weight it slightly.

Weights live inside `compute_loss_from_json(...)`. Adjust and re-run.

---

## FAQ

**Q: Why Recall@20 and not @10?**
A: It checks **coverage** slightly deeper than the head so reranking/fusion has enough items to elevate. If your reader only ever sees k≤6 and you don’t rerank, @10 is fine—update the code to use `recall_at_10`.

**Q: My results don’t have NAUC fields.**
A: The script sets the robustness penalty to zero (no extra penalty). You can remove the term or add your own stability proxy.

**Q: Different task names?**
A: The script only cares that metrics live under `scores.test[...]`. Task name doesn’t matter.

---

## License

MIT (or match your project’s license).

---

## Acknowledgements

Built for evaluating chunk size/overlap for fixed embedding + chunker setups (e.g., E5 + recursive token chunking) using MTEB-style retrieval metrics.

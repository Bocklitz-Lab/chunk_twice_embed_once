# Chunk Twice, Embed Once: A Study of Chunking and Embedding Strategies for Chemical RAG

This repository contains code and data preparation pipelines for reproducing the experiments in the paper:

> **Chunk Twice, Embed Once: A Systematic Study of Chunking and Embedding Strategies for Retrieval-Augmented Generation in Chemistry**  
> *[Authors, Year]*

The repo provides:
- **FSU-ChemRxivQuest**: a benchmark dataset for chemical retrieval built from ChemRxiv papers and QA pairs.
- **Stage 1 (Chunking Study)**: a systematic comparison of 25 chunking configurations, introducing new span-aware evaluation metrics.
- **Stage 2 (Embedding Study)**: evaluation of 48 embedding models across 3 chemistry retrieval tasks.

---

## 🌐 Repository Structure

```

chunk-twice-embed-once/
├─ configs/         # YAML configs for chunking, embeddings, and eval
├─ data/            # datasets (processed, external benchmarks)
├─ src/ragchem/     # main package (chunking, embedding, retrieval, eval, viz)
├─ experiments/     # logs and intermediate results
├─ results/         # final tables and figures
├─ tests/           # unit tests
├─ README.md
├─ requirements.txt
├─ Makefile
└─ ...

````

---

## 📊 Benchmarks

- **FSU-ChemRxivQuest** (released with this repo)
- **ChemHotpotQARetrieval** (MTEB-style benchmark)
- **ChemNQRetrieval** (MTEB-style benchmark)

All datasets follow the [MTEB retrieval format](https://github.com/embeddings-benchmark/mteb):
- `corpus.jsonl`: document chunks
- `queries.jsonl`: questions
- `qrels/test.jsonl`: relevance labels

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-org-or-username>/chunk-twice-embed-once.git
cd chunk-twice-embed-once
pip install -r requirements.txt
pip install -e .
````

---

## 🚀 Quickstart

Prepare the dataset (if not using the preprocessed version):

```bash
make prepare-data
```

Run **Stage 1 (chunking study)**:

```bash
make chunking-eval
```

Run **Stage 2 (embedding study)**:

```bash
make embedding-eval
```

Reproduce figures and tables:

```bash
make figs
```

---

## 📈 Key Findings (from the paper)

* **Chunking**: Recursive Tokenization with a 100-token window and no overlap (**RT100-0**) is a strong default, balancing precision, recall, and efficiency.
* **Overlap**: Adding overlap (e.g. RT100-60) can boost recall, but at the cost of precision and index size.
* **Embeddings**: General-purpose retrieval-optimized encoders such as **Nomic Text v1.5**, **BGE v1.5**, and **E5 Large v2** outperform purely domain-specific chemical models.
* **Clustering analysis** suggests chunking strategies have larger effects on retrieval performance than switching among top-tier encoders.

---

## 📂 Results

Final results (tables and figures from the paper) are stored under:

* `results/tables/`
* `results/figures/`

---

## 🔬 Citation

If you use this repository, please cite:

```bibtex
@inproceedings{YourCitationKey,
  title={Chunk Twice, Embed Once: A Study of Chunking and Embedding Strategies for Retrieval-Augmented Generation in Chemistry},
  author={...},
  booktitle={...},
  year={2025}
}
```

---

## 🤝 Contributing

Pull requests and issues are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📜 License

[MIT License](LICENSE)

```

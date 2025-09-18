---
dataset_info:
- config_name: corpus
  features:
  - name: _id
    dtype: string
  - name: title
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: corpus
    num_bytes: 11814444
    num_examples: 33210
  download_size: 10300000
  dataset_size: 11814444

- config_name: queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: queries
    num_bytes: 132000
    num_examples: 971
  download_size: 132000
  dataset_size: 132000

- config_name: default
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: float64
  splits:
  - name: test
    num_bytes: 86400
    num_examples: 1572  
  download_size: 86400
  dataset_size: 86400

configs:
- config_name: corpus
  data_files:
  - split: corpus
    path: "corpus.jsonl"

- config_name: queries
  data_files:
  - split: queries
    path: "queries.jsonl"

- config_name: default
  data_files:
    - split: test
      path: qrels/test.jsonl

task_categories:
- text-retrieval
language:
- en
tags:
- chemistry
- retrieval
- scientific
- mteb
pretty_name: ChemQest
size_categories:
- 10K<n<100K
license: cc-by-nc-sa-4.0
---


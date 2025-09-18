# Overview of chunking strategies evaluated in this study

_Each method is categorized by type, associated chunk size, and overlap size (if applicable). Short names are used throughout the paper for clarity and visual compactness in figures and tables._

| Method                   | Chunk Size | Overlap Size | Short Name |
|---------------------------|------------|--------------|------------|
| Fixed-Token-Chunker       | 64         | 12           | FX64-12    |
| Fixed-Token-Chunker       | 128        | 25           | FX128-25   |
| Fixed-Token-Chunker       | 256        | 50           | FX256-50   |
| Fixed-Token-Chunker       | 512        | 100          | FX512-100  |
| Recursive-Token-Chunker   | 64         | 16           | RT64-16    |
| Recursive-Token-Chunker   | 100        | 20           | RT100-20   |
| Recursive-Token-Chunker   | 128        | 32           | RT128-32   |
| Recursive-Token-Chunker   | 256        | 64           | RT256-64   |
| Recursive-Token-Chunker   | 512        | 128          | RT512-128  |
| Kamradt-Modified-Chunker  | 50         | –            | KM50       |
| Kamradt-Modified-Chunker  | 100        | –            | KM100      |
| Kamradt-Modified-Chunker  | 200        | –            | KM200      |
| Kamradt-Modified-Chunker  | 400        | –            | KM400      |
| Cluster-Semantic-Chunker  | –          | –            | CL         |
| LLM-Semantic-Chunker      | –          | –            | LLM        |

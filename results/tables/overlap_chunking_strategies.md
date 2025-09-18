# Overview of chunking strategies evaluated in the overlap analysis

_All methods share a consistent chunk size of 100 tokens, with varying overlap values ranging from 0 to 80. Fixed and recursive token chunkers are listed with their respective short identifiers used throughout the study._

| Method                 | Chunk Size | Overlap Size | Short Name |
|-------------------------|------------|--------------|------------|
| Fixed-Token-Chunker     | 100        | 0            | FX100-0    |
| Fixed-Token-Chunker     | 100        | 20           | FX100-20   |
| Fixed-Token-Chunker     | 100        | 40           | FX100-40   |
| Fixed-Token-Chunker     | 100        | 60           | FX100-60   |
| Fixed-Token-Chunker     | 100        | 80           | FX100-80   |
| Recursive-Token-Chunker | 100        | 0            | RT100-0    |
| Recursive-Token-Chunker | 100        | 20           | RT100-20   |
| Recursive-Token-Chunker | 100        | 40           | RT100-40   |
| Recursive-Token-Chunker | 100        | 60           | RT100-60   |
| Recursive-Token-Chunker | 100        | 80           | RT100-80   |

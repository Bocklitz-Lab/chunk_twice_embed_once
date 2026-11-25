| stage | item | hours | basis | note |
|---|---|---|---|---|
| parameter (a ratio, not seconds) | slope: grid seconds per run / Phase 1 evaluation_time (least squares through the origin on 4 measured models) | 0.00 | M |  |
| full rerun: dense | BAAI/bge-base-en x 140 + 3 control configurations | 1.89 | E | 44.7 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | BAAI/bge-large-en x 140 + 3 control configurations | 5.36 | E | 126.8 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | BAAI/bge-large-en-v1.5 x 140 + 3 control configurations | 5.36 | E | 126.8 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | BAAI/bge-small-en-v1.5 x 140 + 3 control configurations | 0.85 | E | 20.1 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | intfloat/e5-base x 140 + 3 control configurations | 1.88 | E | 44.5 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | intfloat/e5-large-v2 x 140 + 3 control configurations | 5.08 | M | 120.2 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | intfloat/e5-small x 140 + 3 control configurations | 0.65 | M | 15.4 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | intfloat/multilingual-e5-base x 140 + 3 control configurations | 2.01 | E | 47.6 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | intfloat/multilingual-e5-large x 140 + 3 control configurations | 5.72 | E | 135.5 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | intfloat/multilingual-e5-small x 140 + 3 control configurations | 0.93 | E | 22.0 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | nomic-ai/nomic-embed-text-v1 x 140 + 3 control configurations | 2.67 | E | 63.2 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | nomic-ai/nomic-embed-text-v1.5 x 140 + 3 control configurations | 2.28 | M | 54.0 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: dense | nomic-ai/nomic-embed-text-v2-moe x 140 + 3 control configurations | 3.86 | E | 91.4 s per run x 143 (control: x0.832) x wall ratio 1.07 |
| full rerun: bm25 (CPU) | BM25 x 143 | 1.42 | M | 35.8 s per run |
| full rerun: hybrid (CPU) | hybrid x 1859 | 0.23 | M | 0.44 s per run (fusion only; inputs counted above) |
| full rerun: rerank (GPU) | rerank of the dense top-50, e5-large-v2 x 143 | 9.18 | M | 231 s per run |
| full rerun: rerank (GPU) | depth ablation (20 and 100) at fixed-token 256/0 | 0.22 | E | depth 20 ~0.5 x, depth 100 ~2 x the depth-50 run |
| full rerun: rerank (GPU) | rerank of the BM25 top-50 x 143 | 9.18 | M | 231 s per run |
| full rerun: transfer (GPU) | 23 models x 6 configurations | 2.67 | E | per-model grid mean x cell factor of the configuration x wall ratio |
| full rerun: prepare (CPU) | chunk + label 140 + 3 configurations | 1.39 | M |  |
| full rerun: report + analysis + figures (CPU) | scaled by the number of runs (4576) | 1.30 | E |  |
| already done | Phase 1, 41 models (prompted run; not part of the rerun) | 2.36 | M | only the pooling-override models are re-run (B9) |
| TOTAL | one GPU: all GPU jobs + CPU work (nothing overlapped) | 64.12 | M+E | GPU 59.8 h + CPU 4.3 h |
| TOTAL | 2 GPUs: prepare, then the longest GPU load, then hybrid and report (BM25 overlaps the GPU jobs) | 32.81 | M+E | GPU loads 29.9 h, 29.9 h; wall ratio 1.07 |

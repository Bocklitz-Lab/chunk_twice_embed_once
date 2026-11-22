"""Check our instruction masking (pipeline_lib/instruction_masking.py) against the official InstructorEmbedding package.

The package lives in a SEPARATE virtual environment (it needs sentence-transformers 2.2.x; the pipeline uses 3.x):

  python3.10 -m venv /path/venv_instructor
  /path/venv_instructor/bin/pip install torch==2.8.0 numpy==1.26.4 transformers==4.44.2 huggingface-hub==0.25.2 \\
        sentence-transformers==2.2.2 InstructorEmbedding==1.0.1
  python -m tools.verify_instructor --official-python /path/venv_instructor/bin/python --prepared artifacts/pilot_v2/prepared/fixed_token_c512_o0__<key>

Encodes 10 queries and 10 chunks (fixed indices of the prepared folder, with the instructions of params.model_prompts) with
(1) the official ``INSTRUCTOR`` class (worker process in the other environment), (2) our masked encoder, (3) the plain
prefix (what a SentenceTransformer does without masking), and prints/saves max |difference| and cosine similarity.
Exit code 1 if ours differs from the official embeddings by more than ``--tol`` (max abs) .
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = "ce48b213095e647a6c3536364b9fa00daf57f436"
MODEL = "hkunlp/instructor-xl"
QUERY_IDX = (0, 1, 2, 50, 100, 200, 300, 400, 500, 900)
DOC_IDX = (0, 1, 500, 2000, 4000, 4100, 4200, 4300, 4400, 4470)      # modulo the corpus size


def worker(texts_json: str, out_prefix: str, model_dir: str) -> int:
    """Runs in the official environment: no repository imports."""
    import importlib.metadata as md
    import numpy as np
    from InstructorEmbedding import INSTRUCTOR
    t = json.load(open(texts_json, encoding="utf-8"))
    m = INSTRUCTOR(model_dir, device="cuda:0")
    q = m.encode([[t["query_instruction"], x] for x in t["queries"]], batch_size=8, normalize_embeddings=True)
    d = m.encode([[t["document_instruction"], x] for x in t["documents"]], batch_size=8, normalize_embeddings=True)
    np.save(out_prefix + "_q.npy", q)
    np.save(out_prefix + "_d.npy", d)
    json.dump({p: md.version(p) for p in ["InstructorEmbedding", "sentence-transformers", "transformers", "torch", "numpy",
                                          "huggingface-hub", "tokenizers"]}, open(out_prefix + "_versions.json", "w"))
    return 0


def snapshot_dir() -> str:
    return os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/{REV}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-python", help="python of the environment with InstructorEmbedding")
    ap.add_argument("--prepared", help="prepared folder (corpus.jsonl / queries.jsonl are sampled at fixed indices)")
    ap.add_argument("--out", help="write the verification report (json)")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--worker", nargs=2, metavar=("TEXTS", "OUT_PREFIX"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args.worker[0], args.worker[1], snapshot_dir())
    if not (args.official_python and args.prepared):
        ap.error("--official-python and --prepared are required")
    import importlib.metadata as md
    import numpy as np
    sys.path.insert(0, _REPO_ROOT)
    from sentence_transformers import SentenceTransformer
    import yaml
    from pipeline_lib import instruction_masking as IM

    task = Path(glob.glob(str(Path(args.prepared)))[0]) / "mteb_task"
    qs = [json.loads(l)["text"] for l in open(task / "queries.jsonl", encoding="utf-8")]
    ds = [json.loads(l)["text"] for l in open(task / "corpus.jsonl", encoding="utf-8")]
    prompts = yaml.safe_load(open(Path(_REPO_ROOT) / "configs/main_config.yaml", encoding="utf-8"))["params"]["model_prompts"][MODEL]
    texts = {"queries": [qs[i % len(qs)] for i in QUERY_IDX], "documents": [ds[i % len(ds)] for i in DOC_IDX],
             "query_instruction": prompts["query"], "document_instruction": prompts["passage"]}
    with tempfile.TemporaryDirectory() as tmp:
        tj = Path(tmp) / "texts.json"
        tj.write_text(json.dumps(texts), encoding="utf-8")
        subprocess.run([args.official_python, "-m", "tools.verify_instructor", "--worker", str(tj), str(Path(tmp) / "official")],
                       cwd=_REPO_ROOT, check=True, env=dict(os.environ, PYTHONPATH=_REPO_ROOT))
        oq, od = np.load(Path(tmp) / "official_q.npy"), np.load(Path(tmp) / "official_d.npy")
        official_versions = json.loads((Path(tmp) / "official_versions.json").read_text())
    st = SentenceTransformer(MODEL, revision=REV, device="cuda:0")
    variants = {
        "ours_masked": (IM.encode_masked(st, texts["query_instruction"], texts["queries"], batch_size=8),
                        IM.encode_masked(st, texts["document_instruction"], texts["documents"], batch_size=8)),
        "plain_prefix": (st.encode([texts["query_instruction"] + x for x in texts["queries"]], batch_size=8, normalize_embeddings=True),
                         st.encode([texts["document_instruction"] + x for x in texts["documents"]], batch_size=8, normalize_embeddings=True)),
    }
    tok = st[0].tokenizer
    report = {"n_queries": len(oq), "n_documents": len(od), "documents_over_max_seq_length": sum(
        len(tok(IM.join_instruction(texts["document_instruction"], x))["input_ids"]) > st.max_seq_length for x in texts["documents"]),
        "n_instruction_tokens": {"query": IM.instruction_length(tok, texts["query_instruction"]),
                                 "document": IM.instruction_length(tok, texts["document_instruction"])},
        "versions_official_env": official_versions,
        "versions_pipeline_env": {p: md.version(p) for p in ["sentence-transformers", "transformers", "torch", "numpy", "huggingface-hub", "tokenizers", "mteb"]}}
    for name, (q, d) in variants.items():
        a, b = np.concatenate([q, d]), np.concatenate([oq, od])
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        report[name] = {"max_abs_diff": float(np.abs(a - b).max()), "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
                        "max_abs_diff_query_doc_scores": float(np.abs(q @ d.T - oq @ od.T).max())}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if report["ours_masked"]["max_abs_diff"] <= args.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())

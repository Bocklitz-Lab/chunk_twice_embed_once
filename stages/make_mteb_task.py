import os, json, re, argparse, unicodedata, math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# YAML is optional but supported if installed
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# --- Optional deps with graceful fallbacks ---
try:
    from rapidfuzz import fuzz  # type: ignore
    def fuzzy_token_set(a, b): return fuzz.token_set_ratio(a, b) / 100.0
except Exception:
    import difflib
    def _ratio(a, b): return difflib.SequenceMatcher(None, a, b).ratio()
    def fuzzy_token_set(a, b):
        a_t = " ".join(sorted(a.lower().split()))
        b_t = " ".join(sorted(b.lower().split()))
        return _ratio(a_t, b_t)

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAVE_RBM25 = True
except Exception:
    HAVE_RBM25 = False

# ----------------- Config & CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config JSON/YAML file")
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    if args.config.lower().endswith((".yaml", ".yml")):
        if yaml is None:
            raise RuntimeError("YAML config provided but PyYAML is not installed. Please `pip install pyyaml` or use JSON.")
        cfg = yaml.safe_load(f)
    else:
        cfg = json.load(f)

# Required
qa_jsonl_file     = cfg["qa_jsonl_file"]
chunks_jsonl_file = cfg["chunks_jsonl_file"]
output_folder     = cfg["output_folder"]

# Lean knobs (sensible defaults)
SEARCH_SCOPE = cfg.get("search_scope", "single")     # "single" or "all"
INCLUDE_SOURCE = cfg.get("include_source_flag", True)
NGRAM_N = int(cfg.get("ngram_n", 1))
TOP_K = int(cfg.get("top_k", 5))
ADJ_WINDOW = int(cfg.get("adjacency_window", 1))     # neighbor boost ±1
NEIGHBOR_PENALTY = float(cfg.get("neighbor_penalty", 0.9))
MERGE_WINDOW = bool(cfg.get("merge_window", True))   # merged neighbor window scoring
WEIGHTS = cfg.get("weights", {
    "bm25": 1.0,
    "jaccard": 1.0,
    "fuzzy_token_set": 1.0
})

os.makedirs(output_folder, exist_ok=True)

# ----------------- Helpers -----------------
_word_re = re.compile(r"[A-Za-z0-9]+")

def norm_doc_id(s: str) -> str:
    """
    Normalize document IDs so that '098' matches '98'.
    Keeps '0' as '0'. Trims whitespace and casts to str.
    """
    s = str(s).strip()
    s = s.lstrip("0")
    return s if s != "" else "0"

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_ngrams(s: str, n: int = 1) -> List[str]:
    toks = _word_re.findall(s)
    if n <= 1:
        return toks
    return ["_".join(toks[i:i+n]) for i in range(len(toks)-n+1)]

def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set, b_set = set(a_tokens), set(b_tokens)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0

# ----------------- Mini BM25 fallback -----------------
def build_bm25_index(docs_tok_lists: List[List[str]]):
    if HAVE_RBM25:
        return BM25Okapi(docs_tok_lists)

    class MiniBM25:
        def __init__(self, corpus_tokens):
            self.corpus = corpus_tokens
            self.N = len(corpus_tokens)
            self.df = Counter()
            for toks in corpus_tokens:
                for t in set(toks):
                    self.df[t] += 1
            self.avgdl = sum(len(t) for t in corpus_tokens) / max(1, self.N)
            self.k1 = 1.5
            self.b = 0.75
            self.idf = {
                t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                for t, df in self.df.items()
            }

        def get_scores(self, query_tokens):
            scores = [0.0] * self.N
            for i, doc in enumerate(self.corpus):
                dl = len(doc) or 1
                f = Counter(doc)
                s = 0.0
                for q in query_tokens:
                    idf = self.idf.get(q)
                    if idf is None:
                        continue
                    fq = f.get(q, 0)
                    if fq == 0:
                        continue
                    denom = fq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    s += idf * (fq * (self.k1 + 1)) / denom
                scores[i] = s
            return scores
    return MiniBM25(docs_tok_lists)

# ----------------- Load chunks -----------------
doc_chunks: Dict[str, List[Dict]] = {}
chunk_store: Dict[str, Dict] = {}
all_chunks_flat = []
corpus_out = []

with open(chunks_jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        dname = norm_doc_id(row["source_doc_id"])
        cidx = int(row["chunk_index"])
        cid = str(row["chunk_id"])
        raw = row.get("text", "")
        norm = normalize_text(raw)

        entry = {
            "doc_name": dname,
            "chunk_index": cidx,
            "chunk_id": cid,
            "raw": raw,
            "norm": norm,
            "ngram_toks": tokens_ngrams(norm, NGRAM_N),
        }
        doc_chunks.setdefault(dname, []).append(entry)
        chunk_store[cid] = entry
        all_chunks_flat.append((dname, cidx, cid))
        corpus_out.append({"id": cid, "title": f"{dname} - Chunk {cidx}", "text": raw})

# Ensure chunk order
for dname in doc_chunks:
    doc_chunks[dname].sort(key=lambda x: x["chunk_index"])

print(f"🔄 Loaded documents: {len(doc_chunks)}")
print(f"   Total chunks: {sum(len(v) for v in doc_chunks.values())}")

# ----------------- Build per-doc BM25 -----------------
bm25_per_doc = {
    dname: build_bm25_index([c["ngram_toks"] for c in chunks])
    for dname, chunks in doc_chunks.items()
}

# ----------------- Load QA -----------------
qa_pairs = []
with open(qa_jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        question = row["question"].strip()
        corpus_id = norm_doc_id(row["corpus_id"])
        if SEARCH_SCOPE == "single" and corpus_id not in doc_chunks:
            print(f"⚠️ Skipping QA for missing corpus_id {row['corpus_id']}")
            continue

        ref = {
            "content": row.get("content", ""),
            "start_index": int(row.get("start_index", -1)),  # kept for schema, not used
            "end_index": int(row.get("end_index", -1)),
        }
        qa_pairs.append((question, [ref], corpus_id))

# ----------------- Scoring -----------------
def score_against_doc(ref_norm: str, ref_tokens: List[str], dname: str) -> List[Tuple[str, float, int]]:
    chunks = doc_chunks[dname]
    bm25 = bm25_per_doc[dname]
    bm25_scores = bm25.get_scores(ref_tokens)

    # convert numpy arrays to lists for safe ops
    if hasattr(bm25_scores, "tolist"):
        bm25_scores = bm25_scores.tolist()

    # min-max normalize BM25 per doc
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
    min_bm25 = min(bm25_scores) if len(bm25_scores) > 0 else 0.0
    rng = max(1e-9, max_bm25 - min_bm25)

    results: List[Tuple[str, float, int]] = []
    for i, c in enumerate(chunks):
        s_bm25 = (bm25_scores[i] - min_bm25) / rng if rng > 0 else 0.0
        s_jaccard = jaccard(ref_tokens, c["ngram_toks"])
        s_fuzzy = fuzzy_token_set(ref_norm, c["norm"])

        score = (
            WEIGHTS.get("bm25", 1.0) * s_bm25 +
            WEIGHTS.get("jaccard", 1.0) * s_jaccard +
            WEIGHTS.get("fuzzy_token_set", 1.0) * s_fuzzy
        )
        results.append((c["chunk_id"], float(score), c["chunk_index"]))

    # simple neighbor/merged boost (±ADJ_WINDOW)
    if ADJ_WINDOW > 0 and len(chunks) > 1:
        for i, c in enumerate(chunks):
            left = max(0, i - ADJ_WINDOW)
            right = min(len(chunks) - 1, i + ADJ_WINDOW)

            if MERGE_WINDOW and right > left:
                merged = " ".join(chunks[k]["norm"] for k in range(left, right + 1))
                merged_tokens = tokens_ngrams(merged, NGRAM_N)
                s_j = jaccard(ref_tokens, merged_tokens)
                s_f = fuzzy_token_set(ref_norm, merged)
                merged_score = (
                    WEIGHTS.get("jaccard", 1.0) * s_j +
                    WEIGHTS.get("fuzzy_token_set", 1.0) * s_f
                ) * NEIGHBOR_PENALTY
                results.append((c["chunk_id"], float(merged_score), c["chunk_index"]))

    return results

# ----------------- Process & Save -----------------
def save_jsonl(data, file_name):
    with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

queries_out, qrels_out = [], []
query_id = 0

for question, references, doc_id in qa_pairs:
    queries_out.append({"id": str(query_id), "title": f"Query {query_id}", "text": question})

    if SEARCH_SCOPE == "all":
        search_docs = list(doc_chunks.keys())
    else:
        if doc_id not in doc_chunks:
            print(f"⚠️ Warning: Document {doc_id} not found in chunks.")
            query_id += 1
            continue
        search_docs = [doc_id]

    per_query_scores: Dict[str, float] = defaultdict(float)

    for ref in references:
        ref_norm = normalize_text(ref.get("content", ""))
        if not ref_norm:
            continue
        ref_tokens = tokens_ngrams(ref_norm, NGRAM_N)

        candidates: List[Tuple[str, float, int]] = []
        for dname in search_docs:
            candidates.extend(score_against_doc(ref_norm, ref_tokens, dname))

        # de-dup by chunk_id with max score
        for cid, sc, _idx in candidates:
            if sc > per_query_scores[cid]:
                per_query_scores[cid] = sc

    selected = [(cid, sc) for cid, sc in per_query_scores.items() if sc > 0.0]
    selected.sort(key=lambda x: x[1], reverse=True)

    for cid, sc in selected[:max(TOP_K, 10)]:
        qrel = {"query_id": str(query_id), "doc_id": cid, "relevance": 1}
        if INCLUDE_SOURCE:
            qrel["score"] = round(sc, 6)
        qrels_out.append(qrel)

    query_id += 1

# outputs
save_jsonl(queries_out, "queries.jsonl")
save_jsonl(corpus_out, "corpus.jsonl")
save_jsonl(qrels_out, "qrels.jsonl")

print("✅ MTEB dataset created successfully (lean)!")
print(f"   Queries: {len(queries_out)} | Corpus: {len(corpus_out)} | Qrels: {len(qrels_out)}")

import os, json, argparse, unicodedata, math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# YAML optional
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Fuzzy matching (stable if RapidFuzz present)
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

# BM25
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

# Knobs
SEARCH_SCOPE = cfg.get("search_scope", "single")         # "single" or "all"
INCLUDE_SOURCE = cfg.get("include_source_flag", True)    # include scores in qrels
NGRAM_N = int(cfg.get("ngram_n", 1))
ADJ_WINDOW = max(0, int(cfg.get("adjacency_window", 1))) # clamp to >=0
MERGE_WINDOW = bool(cfg.get("merge_window", True))
WEIGHTS = cfg.get("weights", {"bm25": 1.0, "jaccard": 1.0, "fuzzy_token_set": 1.0})

# Selection mode: top-1 always if >0, others kept by normalized-to-best final score
MIN_SCORE_THRESHOLD = float(cfg.get("min_score_threshold", 0.30))  # threshold on final norm_score
MAX_RESULTS_PER_QUERY = int(cfg.get("max_results_per_query", 0))   # 0 = no cap

# Neighbor cap relative to J+F base (non-BM25 portion)
ALPHA_NEIGHBOR = float(cfg.get("alpha_neighbor", 0.2))  # convex mix weight
PRINT_LOG = bool(cfg.get("print_log", False))
os.makedirs(output_folder, exist_ok=True)

# ----------------- Helpers -----------------
# Unicode-aware tokenizer if `regex` is available
try:
    import regex as re_u  # type: ignore
    _word_re = re_u.compile(r"\p{L}+\p{M}*|\p{N}+")
    def _find_tokens(s: str) -> List[str]:
        return _word_re.findall(s)
except Exception:
    import re
    _word_re = re.compile(r"[A-Za-z0-9]+")
    def _find_tokens(s: str) -> List[str]:
        return _word_re.findall(s)

def norm_doc_id(s: str) -> str:
    """
    Normalize document IDs so that numeric '098' matches '98'.
    Keeps '0' as '0'. Only strips leading zeros if the ID is all digits.
    """
    s = str(s).strip()
    if s.isdigit():
        s = s.lstrip("0")
        return s if s != "" else "0"
    return s  # non-numeric ids left intact

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    import re
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_ngrams(s: str, n: int = 1) -> List[str]:
    toks = _find_tokens(s)
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
        question = (row.get("question") or "").strip()
        corpus_id = norm_doc_id(row["corpus_id"])

        refs = [{
            "content": row.get("content", ""),
            "start_index": int(row.get("start_index", -1)),
            "end_index": int(row.get("end_index", -1)),
        }]

        qa_pairs.append((question, refs, corpus_id))

# ----------------- Scoring -----------------
def score_components_against_doc(ref_norm: str, ref_tokens: List[str], dname: str) -> Dict[str, Dict[str, float]]:
    """
    Return per-chunk component scores for this document:
      - bm25_raw: raw BM25 score (unnormalized)
      - jf_base: J+F *after* neighbor mixing via alpha_neighbor
      - jaccard: J
      - fuzzy: F
    BM25 normalization to [0,1] is done later across ALL docs per query.
    """
    chunks = doc_chunks[dname]
    bm25 = bm25_per_doc[dname]
    bm25_scores = bm25.get_scores(ref_tokens)

    # base J and F for each chunk
    Js, Fs = [], []
    for c in chunks:
        J = jaccard(ref_tokens, c["ngram_toks"])
        F = fuzzy_token_set(ref_norm, c["norm"])
        Js.append(J); Fs.append(F)

    wJ = WEIGHTS.get("jaccard", 1.0)
    wF = WEIGHTS.get("fuzzy_token_set", 1.0)

    # base J+F (per chunk)
    jf_base = [wJ * Js[i] + wF * Fs[i] for i in range(len(chunks))]

    # --- NEW: convex mixing with neighbors (Option A) ---
    if MERGE_WINDOW and ADJ_WINDOW > 0 and len(chunks) > 1 and ALPHA_NEIGHBOR > 0:
        for i in range(len(chunks)):
            left  = max(0, i - ADJ_WINDOW)
            right = min(len(chunks) - 1, i + ADJ_WINDOW)
            if right <= left:
                continue
            merged = " ".join(chunks[k]["norm"] for k in range(left, right + 1))
            merged_tokens = tokens_ngrams(merged, NGRAM_N)

            Jm = jaccard(ref_tokens, merged_tokens)
            Fm = fuzzy_token_set(ref_norm, merged)
            merged_jf = wJ * Jm + wF * Fm

            # convex combination: jf_final = (1-α)*base + α*merged
            jf_base[i] = (1.0 - ALPHA_NEIGHBOR) * jf_base[i] + ALPHA_NEIGHBOR * merged_jf
    # --- end NEW ---

    out: Dict[str, Dict[str, float]] = {}
    for i, c in enumerate(chunks):
        out[c["chunk_id"]] = {
            "bm25_raw": float(bm25_scores[i]),
            "jf_base": float(jf_base[i]),
            "jaccard": float(Js[i]),
            "fuzzy": float(Fs[i]),
        }
    return out


# ----------------- Process & Save -----------------
def save_jsonl(data, file_name):
    with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

queries_out, qrels_out = [], []
query_id = 0

for question, references, doc_id in qa_pairs:
    # Build the reference text: content + question (simple concat)
    ref_content = (references[0].get("content") or "").strip()
    combined_ref = (ref_content + " " + (question or "")).strip()
    ref_norm = normalize_text(combined_ref)

    if not ref_norm:
        print(f"⚠️ Skipping query {query_id}: empty content+question.")
        continue

    queries_out.append({"id": str(query_id), "title": f"Query {query_id}", "text": question or ref_content})

    # Decide search docs
    if SEARCH_SCOPE == "all":
        search_docs = list(doc_chunks.keys())
    else:
        if doc_id in doc_chunks:
            search_docs = [doc_id]
        else:
            print(f"⚠️ corpus_id '{doc_id}' not found; falling back to search ALL docs for query {query_id}.")
            search_docs = list(doc_chunks.keys())

    ref_tokens = tokens_ngrams(ref_norm, NGRAM_N)

    # 1) Collect component scores for all candidate chunks across the search space
    cand: Dict[str, Dict[str, float]] = {}  # cid -> components
    for dname in search_docs:
        comps = score_components_against_doc(ref_norm, ref_tokens, dname)
        for cid, vals in comps.items():
            # chunk_id is unique globally, but in case of duplicates take the best per component
            if cid not in cand:
                cand[cid] = vals
            else:
                # keep max per component
                cand[cid]["bm25_raw"] = max(cand[cid]["bm25_raw"], vals["bm25_raw"])
                cand[cid]["jf_base"] = max(cand[cid]["jf_base"], vals["jf_base"])
                cand[cid]["jaccard"] = max(cand[cid]["jaccard"], vals["jaccard"])
                cand[cid]["fuzzy"] = max(cand[cid]["fuzzy"], vals["fuzzy"])

    if not cand:
        print(f"⚠️ No candidates scored for query {query_id}.")
        query_id += 1
        continue

    # 2) Normalize BM25 by max BM25 among ALL candidates (per query)
    eps = 1e-9
    max_bm25 = max(v["bm25_raw"] for v in cand.values()) if cand else 0.0
    wB = WEIGHTS.get("bm25", 1.0)

    # 3) Build final scores: w_bm25 * bm25_norm + jf_base
    final_scores: Dict[str, float] = {}
    bm25_norm_map: Dict[str, float] = {}
    for cid, vals in cand.items():
        bm25_norm = (vals["bm25_raw"] / (max_bm25 + eps)) if max_bm25 > 0 else 0.0
        bm25_norm_map[cid] = bm25_norm
        final_scores[cid] = wB * bm25_norm + vals["jf_base"]

    # 4) Rank and apply top-1 + normalized-threshold selection
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    best_cid, best_score = ranked[0]

    if best_score <= 0.0:
        print(f"⚠️ Best final score <= 0 for query {query_id}; no results.")
        query_id += 1
        continue

    kept = [(best_cid, best_score, 1.0)]  # (cid, score, norm_score)

    for cid, sc in ranked[1:]:
        norm_sc = sc / best_score if best_score != 0 else 0.0
        if norm_sc >= MIN_SCORE_THRESHOLD:
            kept.append((cid, sc, norm_sc))

    # Optionally cap results (always keep top-1)
    if MAX_RESULTS_PER_QUERY > 0 and len(kept) > MAX_RESULTS_PER_QUERY:
        kept = kept[:MAX_RESULTS_PER_QUERY]

    # Optional: prints
    if PRINT_LOG:
        print(f"[query {query_id}] kept {len(kept)} docs at norm-threshold {MIN_SCORE_THRESHOLD} (max={MAX_RESULTS_PER_QUERY or '∞'})")

    # 5) Write qrels
    for cid, sc, nsc in kept:
        qrel = {"query_id": str(query_id), "doc_id": cid, "relevance": 1}
        if INCLUDE_SOURCE:
            qrel["score"] = float(f"{sc:.6f}")            # final hybrid score
            qrel["norm_score"] = float(f"{nsc:.6f}")      # normalized to best final score
            qrel["bm25_norm"] = float(f"{bm25_norm_map[cid]:.6f}")  # BM25 normalized to query max
            qrel["bm25_raw"] = float(f"{cand[cid]['bm25_raw']:.6f}")
            qrel["jaccard"] = float(f"{cand[cid]['jaccard']:.6f}")
            qrel["fuzzy"] = float(f"{cand[cid]['fuzzy']:.6f}")
        qrels_out.append(qrel)

    query_id += 1

# outputs
def save_jsonl(data, file_name):
    with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

save_jsonl(queries_out, "queries.jsonl")
save_jsonl(corpus_out, "corpus.jsonl")
save_jsonl(qrels_out, "qrels.jsonl")

print("✅ MTEB dataset created successfully (BM25 normalized per query to max)!")
print(f"   Queries: {len(queries_out)} | Corpus: {len(corpus_out)} | Qrels: {len(qrels_out)}")

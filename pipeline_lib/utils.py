import os, json, re, unicodedata, math
from typing import List, Dict, Any
from enum import Enum
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
# from utils import embedding_functions
import tiktoken
import unicodedata
import re
from fuzzywuzzy import process, fuzz

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p: str):
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_jsonl(rows: List[Dict[str, Any]], p: str):
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(p: str) -> List[Dict[str, Any]]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_config(path: str) -> Dict[str, Any]:
    if path.lower().endswith((".yml", ".yaml")):
        import yaml  # pip install pyyaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cfg_get(cfg: Dict[str, Any], path: str, default=None, required=False):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            if required:
                raise KeyError(f"Missing config key: {path}")
            return default
        cur = cur[k]
    return cur

# --- Text math ---
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens: return 0.0
    A, B = set(a_tokens), set(b_tokens)
    inter, union = len(A & B), len(A | B)
    return inter / union if union else 0.0

def lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

def rougeL_like(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens: return 0.0
    L = lcs_len(a_tokens, b_tokens)
    prec, rec = L/len(b_tokens), L/len(a_tokens)
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)



def cosine(u: List[float], v: List[float]) -> float:
    if not u or not v:
        return 0.0
    assert len(u) == len(v), "Vectors must be the same length"

    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return 0.0 if nu == 0 or nv == 0 else dot / (nu * nv)


def overlap_ratio(rs:int,re:int, cs:int,ce:int)->float:
    left, right = max(rs,cs), min(re,ce)
    inter = max(0, right-left)
    ref_len = max(1, re-rs)
    return inter/ref_len

def find_query_despite_whitespace(document, query):

    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()
    
    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())
    
    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)
    
    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None
    



def normalize_text(text):
    """Normalize Unicode text to NFKC form for consistent comparison."""
    return unicodedata.normalize("NFKC", text)

def rigorous_document_search(document: str, target: str):
    """
    Searches for a target string within a document, handling Unicode normalization, whitespace variations, 
    and fuzzy matching for approximate matches.
    
    Args:
        document (str): The document to search within.
        target (str): The text string to find.

    Returns:
        tuple: (best_match, start_index, end_index) if found, otherwise None.
    """
    if not document or not target:
        return None  # Ensure inputs are valid
    
    # Normalize both document and target
    document = normalize_text(document)
    target = normalize_text(target)
# In your evaluation or data prep code:
    document = document.replace("�", "")
    target = target.replace("�", "")

    # Remove trailing period from target (common in chunk searches)
    target = target.rstrip('.')

    # 1️⃣ Exact Match Search
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index

    # 2️⃣ Whitespace-Insensitive Search
    raw_search = find_query_despite_whitespace(document, target)
    if raw_search is not None:
        return raw_search

    # 3️⃣ Fuzzy Matching for Approximate Searches
    sentences = re.split(r'[.!?]\s*|\n', document)  # Split into sentences
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match and best_match[1] >= 95:  # Adjusted threshold for flexibility
        reference = best_match[0]
        start_index = document.find(reference)
        end_index = start_index + len(reference)
        return reference, start_index, end_index

    # 4️⃣ No match found
    return None


# def get_openai_embedding_function():
#     openai_api_key = os.getenv('OPENAI_API_KEY')
#     if openai_api_key is None:
#         raise ValueError("You need to set an embedding function or set an OPENAI_API_KEY environment variable.")
#     embedding_function = embedding_functions.OpenAIEmbeddingFunction(
#         api_key=os.getenv('OPENAI_API_KEY'),
#         model_name="text-embedding-3-large"
#     )
#     return embedding_function

# Count the number of tokens in each page_content
def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"
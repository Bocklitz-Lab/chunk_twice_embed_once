import os, json, re, unicodedata, math
from typing import List, Dict, Any

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
    if not u or not v: return 0.0
    dot = sum(x*y for x,y in zip(u,v))
    nu = math.sqrt(sum(x*x for x in u)); nv = math.sqrt(sum(y*y for y in v))
    return 0.0 if nu==0 or nv==0 else dot/(nu*nv)

def overlap_ratio(rs:int,re:int, cs:int,ce:int)->float:
    left, right = max(rs,cs), min(re,ce)
    inter = max(0, right-left)
    ref_len = max(1, re-rs)
    return inter/ref_len

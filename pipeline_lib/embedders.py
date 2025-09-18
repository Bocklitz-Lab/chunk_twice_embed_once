from typing import List
from .utils import cfg_get

class BaseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str, max_batch: int):
        try:
            from openai import OpenAI  # pip install openai>=1.0
        except Exception as e:
            raise RuntimeError("Install openai and set OPENAI_API_KEY") from e
        self.client = OpenAI()
        self.model = model
        self.max_batch = max_batch

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.max_batch):
            batch = texts[i:i+self.max_batch]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out

class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        except Exception as e:
            raise RuntimeError("Install sentence-transformers for LocalEmbedder") from e
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

def build_embedder(cfg) -> BaseEmbedder:
    provider = cfg_get(cfg, "embedding.provider", required=True)
    if provider == "openai":
        return OpenAIEmbedder(
            model=cfg_get(cfg, "embedding.openai_model", required=True),
            max_batch=int(cfg_get(cfg, "embedding.max_batch", 1024)),
        )
    elif provider == "local":
        return LocalEmbedder(
            model_name=cfg_get(cfg, "embedding.local_model", required=True)
        )
    else:
        raise ValueError(f"Unknown embedding.provider: {provider}")

def make_embedding_function_from_embedder(embedder: BaseEmbedder):
    def fn(batch: List[str]) -> List[List[float]]:
        return embedder.embed(batch)
    return fn

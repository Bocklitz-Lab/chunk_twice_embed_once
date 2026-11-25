"""params.model_pooling: pooling override for models without modules.json (pipeline_lib/pooling.py)."""
import pytest
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling

from pipeline_lib import pooling as P


class TinyEncoder(torch.nn.Module):
    """A word-embedding module with a fixed vocabulary: token ids -> 8-dim vectors (enough to tell CLS from mean pooling)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(50, 8)
        self.max_seq_length = 16

    def forward(self, features):
        features["token_embeddings"] = self.emb(features["input_ids"])
        return features

    def get_word_embedding_dimension(self):
        return 8

    def tokenize(self, texts, padding=True):
        ids = [[1] + [2 + (ord(c) % 40) for c in t[:10]] for t in texts]
        n = max(len(x) for x in ids)
        return {"input_ids": torch.tensor([x + [0] * (n - len(x)) for x in ids]), "attention_mask": torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids])}

    def save(self, path):
        pass


def _model():
    return SentenceTransformer(modules=[TinyEncoder(), Pooling(8, pooling_mode="mean")], device="cpu")


def test_resolve_and_validation():
    assert P.resolve("a/b", None) is None and P.resolve("a/b", {}) is None
    assert P.resolve("a/b", {"a/b": {"mode": "cls", "source": "card sentence"}}) == "cls"
    with pytest.raises(P.PoolingConfigError, match="mode must be"):
        P.resolve("a/b", {"a/b": {"mode": "best", "source": "x"}})
    with pytest.raises(P.PoolingConfigError, match="source"):
        P.resolve("a/b", {"a/b": {"mode": "cls"}})


def test_apply_replaces_mean_by_cls_and_changes_the_embedding():
    m = _model()
    mean = m.encode(["alpha beta", "gamma"], convert_to_numpy=True)
    P.apply(m, "cls", has_modules_json=False, model_name="x")
    cls = m.encode(["alpha beta", "gamma"], convert_to_numpy=True)
    assert cls.shape == mean.shape and abs(cls - mean).max() > 1e-4
    emb = TinyEncoder()(TinyEncoder().tokenize(["alpha beta"]))["token_embeddings"][0, 0].detach().numpy()          # the CLS (first) token vector
    assert abs(cls[0] - emb).max() < 1e-5
    assert m[1].pooling_mode_cls_token and not m[1].pooling_mode_mean_tokens


def test_a_model_with_modules_json_is_refused():
    with pytest.raises(P.PoolingConfigError, match="modules.json"):
        P.apply(_model(), "cls", has_modules_json=True, model_name="x")


def test_unknown_snapshot_counts_as_having_modules_json(monkeypatch):
    from pipeline_lib import model_info
    monkeypatch.setattr(model_info, "has_modules_json", lambda n, r: None)
    assert P.has_modules_json(None, "x/y", "r") is True                      # cannot be sure: refuse the override


def test_config_overrides_only_specter_and_every_entry_has_a_source():
    import yaml
    from pathlib import Path
    tab = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/main_config.yaml").read_text())["params"]["model_pooling"]
    assert set(tab) == {"allenai/specter"}
    for name in tab:
        assert P.resolve(name, tab) == "cls" and "[CLS]" in tab[name]["source"]


def test_model_properties_marks_the_override(tmp_path):
    from tools import model_properties as MP
    rows, skipped = MP.build_rows([{"name": "allenai/specter", "revision": "81cbfb43d4fc2e728d5b4201ce14987db8d0854c"}], {}, load_tokenizers=False,
                                  pooling_overrides={"allenai/specter": {"mode": "cls", "source": "s"}})
    if skipped:
        pytest.skip("SPECTER snapshot not in the local cache")
    assert rows[0]["pooling_mode"] == "cls" and rows[0]["pooling_override"] is True and rows[0]["pooling_is_library_default"] is False

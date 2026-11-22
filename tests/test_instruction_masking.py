"""INSTRUCTOR instruction masking (pipeline_lib/instruction_masking.py) on a tiny random BERT built from the cached tokenizer.

The official class is checked in tests/test_instruction_masking_official.py (skipped unless InstructorEmbedding is installed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
transformers = pytest.importorskip("transformers")
from sentence_transformers import SentenceTransformer, models

from pipeline_lib import instruction_masking as IM
from pipeline_lib.model_wrapper import STNoPrompt
from pipeline_lib.prompts import resolve_prompts

REV = "86b5e0934494bd15c9632b12f734a8a67f723594"
INS_D = "Represent the Wikipedia document for retrieval: "
INS_Q = "Represent the Wikipedia question for retrieving supporting documents: "
TEXTS = ["furfural yield from bagasse", "a much longer text about the extraction of furfural from sugarcane bagasse in Pakistan", "x"]


@pytest.fixture(scope="module")
def st(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_bert")
    tok = transformers.AutoTokenizer.from_pretrained("bert-base-uncased", revision=REV)
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = transformers.BertConfig(vocab_size=tok.vocab_size, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=128)
    transformers.BertModel(cfg).save_pretrained(d)
    tr = models.Transformer(str(d), max_seq_length=64)
    pool = models.Pooling(16, pooling_mode="mean")
    return SentenceTransformer(modules=[tr, pool, models.Dense(16, 8, bias=False)], device="cpu")


def test_instruction_length_is_the_token_count_minus_the_end_token(st):
    tok = st[0].tokenizer
    n = IM.instruction_length(tok, INS_D)
    assert n == len(tok(INS_D.strip())["input_ids"]) - 1 and n > 5
    assert IM.instruction_length(tok, "  " + INS_D + "  ") == n                     # the official class strips


def test_join_has_no_separator_and_strips():
    assert IM.join_instruction("Represent: ", " text ") == "Represent:text"


def test_pooling_mask_zeroes_the_prefix_only():
    m = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    out = IM.pooling_mask(m, 2)
    assert out.tolist() == [[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]] and m[0, 0] == 1          # a copy


def _manual(st, instruction, texts, normalize=True):
    """Reference written straight from the algorithm: full-attention encoder pass, masked mean, Dense."""
    tok, bert = st[0].tokenizer, st[0].auto_model
    n = len(tok(instruction.strip())["input_ids"]) - 1
    rows = []
    for t in texts:
        enc = tok(instruction.strip() + t.strip(), return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            h = bert(**enc)[0][0]                       # (seq, 16): the text attended to the instruction
            pooled = h[n:].mean(0, keepdim=True)        # the instruction tokens are not averaged
            v = st[2]({"sentence_embedding": pooled})["sentence_embedding"][0]
        rows.append(torch.nn.functional.normalize(v, dim=0) if normalize else v)
    return torch.stack(rows).numpy()


def test_masked_embedding_equals_the_manual_reference_and_keeps_the_input_order(st):
    got = IM.encode_masked(st, INS_D, TEXTS, batch_size=2)             # batches of different lengths, padding inside a batch
    assert got.shape == (3, 8)
    np.testing.assert_allclose(got, _manual(st, INS_D, TEXTS), atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
    single = IM.encode_masked(st, INS_D, TEXTS[1], batch_size=1)
    np.testing.assert_allclose(single[0], got[1], atol=1e-5)           # batch composition does not matter


def test_masked_differs_from_the_plain_prefix_but_the_instruction_still_conditions_the_text(st):
    masked = IM.encode_masked(st, INS_D, TEXTS)
    plain = st.encode([INS_D + t for t in TEXTS], normalize_embeddings=True)
    assert np.abs(masked - plain).max() > 1e-3                         # averaging the instruction tokens changes the vector
    other = IM.encode_masked(st, INS_Q, TEXTS)
    assert np.abs(masked - other).max() > 1e-3                         # a different instruction still moves the text vectors


def test_only_mean_pooling_is_supported(st, tmp_path):
    cls = SentenceTransformer(modules=[st[0], models.Pooling(16, pooling_mode="cls")], device="cpu")
    with pytest.raises(ValueError, match="mean pooling"):
        IM.encode_masked(cls, INS_D, TEXTS)


def test_wrapper_uses_masking_only_when_asked_and_only_for_the_prompted_side(st):
    table = {"m": {"query": INS_Q, "passage": INS_D, "card": "", "mask_instruction": True}}
    for flag, expect_mask in [(True, True), (False, False)]:
        prompts = resolve_prompts("m", table, True, mask_instruction=flag)
        assert prompts["mask_instruction"] is expect_mask
        w = STNoPrompt(st, normalize=True, prompts=prompts)
        got = w.encode(TEXTS, prompt_type="passage")
        ref = IM.encode_masked(st, INS_D, TEXTS) if expect_mask else st.encode([INS_D + t for t in TEXTS], normalize_embeddings=True)
        np.testing.assert_allclose(got, ref, atol=1e-5)
    # a side without instruction is a plain encode even with masking on
    prompts = resolve_prompts("m", {"m": {"query": INS_Q, "passage": "", "mask_instruction": True}}, True, mask_instruction=True)
    w = STNoPrompt(st, normalize=True, prompts=prompts)
    np.testing.assert_allclose(w.encode(TEXTS, prompt_type="passage"), st.encode(TEXTS, normalize_embeddings=True), atol=1e-5)


def test_flag_off_is_the_old_behaviour_and_other_models_never_mask():
    p = resolve_prompts("intfloat/e5-large-v2", {"intfloat/e5-large-v2": {"query": "query: ", "passage": "passage: "}}, True, mask_instruction=True)
    assert p["mask_instruction"] is False                              # the entry does not declare it: e5 keeps a plain prefix
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/main_config.yaml").read_text())["params"]
    assert cfg["mask_instruction_tokens"] is False
    assert [n for n, e in cfg["model_prompts"].items() if e.get("mask_instruction")] == ["hkunlp/instructor-xl"]
    assert resolve_prompts("hkunlp/instructor-xl", cfg["model_prompts"], True, mask_instruction=False)["mask_instruction"] is False
    assert resolve_prompts("hkunlp/instructor-xl", cfg["model_prompts"], True, mask_instruction=True)["mask_instruction"] is True

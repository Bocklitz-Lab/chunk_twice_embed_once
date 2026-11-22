"""Instruction masking of INSTRUCTOR models (hkunlp/instructor-xl), without the ``InstructorEmbedding`` package.

Follows the released package (InstructorEmbedding 1.0.1, ``INSTRUCTOR_Transformer.tokenize`` / ``.forward``) for a pair
[instruction, text]:

  1. tokenise  instruction + text.strip()   (the instruction is NOT stripped: its trailing space stays) and truncate to max_seq_length;
  2. tokenise instruction.strip() alone: n_instruction = number of tokens minus one (the end-of-sequence token);
     n_instruction <= 1 is treated as 0;
  3. run the encoder over the whole sequence: the text attends to the instruction (the instruction conditions the text);
  4. set the pooling attention mask to 0 for the first n_instruction positions, so the instruction tokens do not
     enter the mean pooling (nor its divisor); then Pooling (mean) -> Dense -> normalise.

A plain SentenceTransformer, given the instruction as a text prefix, does steps 1 and 3 but averages over the
instruction tokens too (the pooling mask is the attention mask). This module adds step 2 and 4 on top of the same
SentenceTransformer modules (Transformer, Pooling, Dense), so weights and tokenizer are the same ones.
(The GitHub ``main`` branch of InstructorEmbedding differs from the release in small ways, e.g. it strips the instruction;
the release is what ``pip install InstructorEmbedding`` gives and is the reference here: tools/verify_instructor.py.)

Used by ``STNoPrompt`` only when ``params.instructor_mask_instruction`` is true and the model's prompt entry declares
``mask_instruction: true``. The published runs (and the plain-prefix runs) do not mask.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np


def instruction_length(tokenizer, instruction: str, do_lower_case: bool = False) -> int:
    """Number of instruction tokens that are masked: token count of the instruction alone minus the end token."""
    instruction = instruction.strip()
    if do_lower_case:
        instruction = instruction.lower()
    n = len(tokenizer(instruction)["input_ids"]) - 1
    return 0 if n <= 1 else n


def join_instruction(instruction: str, text: str, do_lower_case: bool = False) -> str:
    """The text of the released package: the instruction as it is (trailing space included) + the stripped text."""
    text = text.strip()
    if do_lower_case:
        text = text.lower()
    return instruction + text


def pooling_mask(attention_mask, n_instruction: int):
    """attention_mask with the first ``n_instruction`` positions of every row set to 0 (a copy)."""
    mask = attention_mask.clone()
    mask[:, :n_instruction] = 0
    return mask


def check_supported(st_model) -> None:
    pool = st_model[1]
    if not getattr(pool, "pooling_mode_mean_tokens", False):
        raise ValueError("instruction masking is defined for mean pooling (INSTRUCTOR); this model pools differently")


def encode_masked(st_model, instruction: str, texts: Sequence[str] | str, batch_size: int = 32, normalize: bool = True,
                  device: Optional[str] = None) -> np.ndarray:
    """Embeddings of ``texts`` with ``instruction`` as the INSTRUCTOR instruction (see the module docstring)."""
    import torch

    check_supported(st_model)
    if isinstance(texts, str):
        texts = [texts]
    transformer = st_model[0]
    tok = transformer.tokenizer
    lower = bool(getattr(transformer, "do_lower_case", False))
    max_len = transformer.max_seq_length
    n_instr = instruction_length(tok, instruction, lower)
    if n_instr >= max_len:
        raise ValueError(f"the instruction ({n_instr} tokens) does not fit into max_seq_length={max_len}")
    device = device or str(st_model.device)
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out: List[Optional[np.ndarray]] = [None] * len(texts)
    st_model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            enc = tok([join_instruction(instruction, texts[i], lower) for i in idx], padding=True, truncation="longest_first",
                      max_length=max_len, return_tensors="pt")
            features = {k: v.to(device) for k, v in enc.items()}
            features = transformer(features)                              # full attention: the text sees the instruction
            features["attention_mask"] = pooling_mask(features["attention_mask"], n_instr)
            for module in list(st_model._modules.values())[1:]:           # Pooling, Dense, (Normalize)
                features = module(features)
            emb = features["sentence_embedding"].detach()
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = emb[j]
    return np.stack(out) if out else np.zeros((0, 0), dtype=np.float32)

import time

from pipeline_lib.prompts import apply_prompt, has_any_prompt, side_of


class STNoPrompt:
    """SentenceTransformer adapter for MTEB.

    ``prompts=None`` (the default, and ``use_prompts: false`` in the config) is the original behaviour: MTEB's
    ``prompt_name`` / ``prompt_type`` are dropped and the text is embedded as it is.
    With ``prompts`` ({"query": ..., "passage": ...}, see ``pipeline_lib.prompts``) the prompt of the side MTEB asks
    for (``prompt_type`` query / passage) is put in front of every text. This happens here, at encoding time, and
    nowhere else: chunks, token counts and qrels are not touched.
    With ``prompts["mask_instruction"]`` (INSTRUCTOR models, ``params.instructor_mask_instruction``) the prompt is an INSTRUCTOR
    instruction: it goes through the encoder with the text but is masked out of the mean pooling (instruction_masking.py).
    """

    def __init__(self, st_model, normalize=True, prompts=None):
        self.model = st_model
        self.normalize = normalize
        self.prompts = prompts if has_any_prompt(prompts) else None
        # one record per encode call ({side, n_texts, seconds}); see pipeline_lib.run_manifest.summarize_encode_timings
        self.timings = []

        # --- expose what MTEB looks for -----------------
        self.name = getattr(st_model, "name", None) \
                    or getattr(st_model, "model_name_or_path", None) \
                    or "all-MiniLM-L6-v2"
        self.revision = "main"       # or a commit hash
        # -------------------------------------------------

    def encode(self, sentences, batch_size=64, **kwargs):
        prompt_type = kwargs.get("prompt_type")
        kwargs.pop("task_name", None)
        kwargs.pop("content_type", None)
        kwargs.pop("prompt_name", None)
        side = side_of(prompt_type)
        instruction = (self.prompts or {}).get(side) if side else ""
        t0 = time.perf_counter()
        if instruction and (self.prompts or {}).get("mask_instruction"):
            # INSTRUCTOR: the instruction conditions the text but is left out of the mean pooling (not a plain prefix)
            from pipeline_lib.instruction_masking import encode_masked
            emb = encode_masked(self.model, instruction, sentences, batch_size=batch_size, normalize=self.normalize)
        else:
            sentences = apply_prompt(sentences, self.prompts, prompt_type)
            emb = self.model.encode(
                sentences,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        # convert_to_numpy copies the result to the host, so the GPU work is finished here
        self.timings.append({"side": side_of(prompt_type), "n_texts": len(sentences) if not isinstance(sentences, str) else 1,
                             "seconds": time.perf_counter() - t0})
        return emb

    # make every other attribute fall through
    def __getattr__(self, attr):
        return getattr(self.model, attr)

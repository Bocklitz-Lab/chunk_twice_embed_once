from pipeline_lib.prompts import apply_prompt, has_any_prompt


class STNoPrompt:
    """SentenceTransformer adapter for MTEB.

    ``prompts=None`` (the default, and ``use_prompts: false`` in the config) is the original behaviour: MTEB's
    ``prompt_name`` / ``prompt_type`` are dropped and the text is embedded as it is.
    With ``prompts`` ({"query": ..., "passage": ...}, see ``pipeline_lib.prompts``) the prompt of the side MTEB asks
    for (``prompt_type`` query / passage) is put in front of every text. This happens here, at encoding time, and
    nowhere else: chunks, token counts and qrels are not touched.
    """

    def __init__(self, st_model, normalize=True, prompts=None):
        self.model = st_model
        self.normalize = normalize
        self.prompts = prompts if has_any_prompt(prompts) else None

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
        sentences = apply_prompt(sentences, self.prompts, prompt_type)
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    # make every other attribute fall through
    def __getattr__(self, attr):
        return getattr(self.model, attr)

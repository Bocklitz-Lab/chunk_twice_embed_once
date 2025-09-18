
class STNoPrompt:
    def __init__(self, st_model, normalize=True):
        self.model = st_model
        self.normalize = normalize

        # --- expose what MTEB looks for -----------------
        self.name = getattr(st_model, "name", None) \
                    or getattr(st_model, "model_name_or_path", None) \
                    or "all-MiniLM-L6-v2"
        self.revision = "main"       # or a commit hash
        # -------------------------------------------------

    def encode(self, sentences, batch_size=64, **kwargs):
        kwargs.pop("task_name", None)
        kwargs.pop("content_type", None)
        kwargs.pop("prompt_name", None)
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


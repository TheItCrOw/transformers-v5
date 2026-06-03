"""Embedding model wrapper with transformers 5.x compatibility fixes.

Fixes two regressions in transformers 5.x that silently produce garbage
embeddings for custom-code models (trust_remote_code=True) based on the
GTE architecture (e.g. Snowflake Arctic Embed).

Affected versions: transformers >= 5.0.0
Upstream issues:
  - https://github.com/huggingface/transformers/issues/43950  (buffer corruption)
  - https://github.com/huggingface/transformers/issues/43646  (_init_weights re-run)
"""

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer


def _fix_model_for_v5(model, model_name):
    """Fix transformers 5.x regressions for custom-code models.

    Problem 1 -- Non-persistent buffer corruption (#43950):
      from_pretrained() initializes on meta tensors, then materializes from
      state_dict. Non-persistent buffers (position_ids, inv_freq, cos_cached,
      sin_cached) are excluded from the state_dict by PyTorch design.
      After materialization the underlying memory is freed; the buffer tensors
      still exist but now point at garbage (use-after-free).

    Problem 2 -- _init_weights re-initialization (#43646):
      The GTE model's _init_weights() uses .data.normal_() / .data.fill_()
      which don't set the _is_hf_initialized flag. Transformers 5.x therefore
      considers every module uninitialized and re-runs _init_weights() AFTER
      loading, overwriting all learned parameters with random values.

    Fix:
      1. Re-load the original weights from the cached safetensors file.
      2. Rebuild all corrupted non-persistent buffers from their definitions.
    """
    # --- Fix 1: Reload learned weights from the original safetensors file ---
    weights_path = hf_hub_download(model_name, filename="model.safetensors")
    original_weights = load_file(weights_path)
    model.load_state_dict(original_weights, strict=False)

    # --- Fix 2: Rebuild corrupted non-persistent buffers ---

    # position_ids: torch.arange(max_position_embeddings)
    emb = model.embeddings
    max_pos = emb.position_ids.shape[0]
    emb.register_buffer(
        "position_ids",
        torch.arange(max_pos, device=emb.position_ids.device),
        persistent=False,
    )

    # RotaryEmbedding: inv_freq, cos_cached, sin_cached
    for _, mod in model.named_modules():
        if hasattr(mod, "inv_freq") and mod.inv_freq is not None:
            dim = mod.dim
            base = mod.base
            device = mod.inv_freq.device

            # inv_freq = 1 / (base ^ (2i / dim))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            mod.register_buffer(
                "inv_freq", inv_freq.to(device), persistent=False
            )

            # cos/sin cache: mirrors RotaryEmbedding._set_cos_sin_cache
            # freqs are doubled via cat((freqs, freqs), dim=-1) before cos/sin
            max_seq = mod.max_seq_len_cached
            t = torch.arange(max_seq, dtype=torch.float32, device=device)
            freqs = torch.outer(t, inv_freq.to(device))
            doubled = torch.cat((freqs, freqs), dim=-1)
            mod.register_buffer(
                "cos_cached",
                doubled.cos().to(torch.get_default_dtype()),
                persistent=False,
            )
            mod.register_buffer(
                "sin_cached",
                doubled.sin().to(torch.get_default_dtype()),
                persistent=False,
            )


class EmbeddingModel:
    """Wrapper around Snowflake Arctic Embed that supports both
    sentence-transformers and raw transformers backends.

    Usage:
        model = EmbeddingModel(backend="transformers")
        embeddings = model.embed(["Hello world", "Another sentence"])
        # embeddings.shape == (2, 768)
    """

    MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"

    def __init__(self, backend: str = "sentence_transformers"):
        if backend not in ("sentence_transformers", "transformers"):
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self):
        if self.backend == "sentence_transformers":
            st_model = SentenceTransformer(
                self.MODEL_NAME,
                trust_remote_code=True,
                config_kwargs={"use_memory_efficient_attention": False},
            )
            _fix_model_for_v5(st_model[0].auto_model, self.MODEL_NAME)
            self._model = st_model
        else:
            config = AutoConfig.from_pretrained(
                self.MODEL_NAME, trust_remote_code=True
            )
            config.use_memory_efficient_attention = False
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.MODEL_NAME,
                config=config,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            _fix_model_for_v5(self._model, self.MODEL_NAME)
            self._model.eval()

    def embed(self, sentences: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings, shape (n, 768)."""
        if self.backend == "sentence_transformers":
            return self._model.encode(sentences, normalize_embeddings=True)

        tokens = self._tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            output = self._model(**tokens)
        # CLS pooling + L2 normalize
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()

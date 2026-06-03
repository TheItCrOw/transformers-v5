# Transformers 5.x Embedding Regression: Root Cause Analysis & Fix

## Problem

Upgrading from `transformers==4.x` to `transformers==5.0.0` causes **Snowflake/snowflake-arctic-embed-m-v2.0** (GTE-based, `trust_remote_code=True`) to produce completely wrong embeddings. Cosine similarity between v4 and v5 outputs was ~0.0 (uncorrelated random noise) instead of ~1.0.

Both backends affected: raw `transformers` and `sentence-transformers` (which wraps transformers internally).

## Root Cause

Two independent bugs in transformers 5.x compound to corrupt the model entirely:

### Bug 1: Non-persistent buffer corruption ([#43950](https://github.com/huggingface/transformers/issues/43950))

Transformers 5.x loads models via meta-tensor initialization + weight materialization from `state_dict`. Non-persistent buffers (`register_buffer(..., persistent=False)`) are **not** stored in the `state_dict`. After materialization, the underlying tensor storage is freed, but the buffer objects still reference the freed memory (use-after-free). Result: garbage values.

**Affected buffers in GTE/Arctic Embed:**

| Buffer | Expected | Actual (v5) |
|--------|----------|-------------|
| `embeddings.position_ids` | `[0, 1, 2, ...]` | `[4.88e+12, 0, 0, ...]` |
| `embeddings.rotary_emb.inv_freq` | `[1.0, 0.688, 0.473, ...]` | `[-4.17e+25, 1.59e-42, 0, ...]` |
| `embeddings.rotary_emb.cos_cached` | Derived from `inv_freq` | Zeros / garbage |
| `embeddings.rotary_emb.sin_cached` | Derived from `inv_freq` | Zeros / garbage |

Since `inv_freq` drives all RoPE positional encodings across every attention layer, corrupted `inv_freq` alone is enough to produce completely wrong outputs.

### Bug 2: `_init_weights()` re-initialization ([#43646](https://github.com/huggingface/transformers/issues/43646))

The GTE model's `_init_weights()` uses direct `.data` mutations:

```python
module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
module.bias.data.zero_()
module.weight.data.fill_(1.0)
```

These mutations **don't set** the `_is_hf_initialized` flag. Transformers 5.x checks this flag after loading and re-runs `_init_weights()` on every module it considers "uninitialized" -- overwriting all 305M learned parameters with random values.

**Verified:** direct comparison of loaded weights vs the safetensors file showed `max_diff=2.07` on LayerNorm alone.

## Fix

Implemented in `embedding_model_v5_fixed.py` / `_fix_model_for_v5()`:

1. **Re-load weights** from the cached `model.safetensors` via `hf_hub_download()` + `load_state_dict()` -- undoes the `_init_weights` damage.
2. **Rebuild `position_ids`** as `torch.arange(max_position_embeddings)`.
3. **Recompute `inv_freq`** as `1 / (base ^ (2i/dim))` with `base=160000`, `dim=64`.
4. **Recompute `cos_cached` / `sin_cached`** from corrected `inv_freq`, mirroring the model's `_set_cos_sin_cache()` logic (freqs doubled via `cat((freqs, freqs), dim=-1)` before cos/sin).

## Verification

Cosine similarity across all v4/v5 and backend combinations after fix:

```
v4x_transformers        vs v5x_transformers:          mean=1.00000000
v4x_sentence_transformers vs v5x_sentence_transformers: mean=1.00000000
v5x_sentence_transformers vs v5x_transformers:          mean=1.00000000
```

Zero drift detected on all 10 test sentences.

## Affected Stack

| Package | Broken | Known-good |
|---------|--------|------------|
| `transformers` | `5.0.0` | `4.57.x` |
| `sentence-transformers` | `5.2.0` (uses transformers 5.x internally) | `4.0.2` |

Applies to any model using `trust_remote_code=True` with GTE-style custom code that registers non-persistent buffers and uses `.data` mutations in `_init_weights()`.

## Upstream Tracking

- [huggingface/transformers#43950](https://github.com/huggingface/transformers/issues/43950) -- `from_pretrained()` silently corrupts non-persistent buffers
- [huggingface/transformers#43646](https://github.com/huggingface/transformers/issues/43646) -- `_init_weights` re-runs after loading, overwrites saved weights
- [huggingface/transformers#44534](https://github.com/huggingface/transformers/issues/44534) -- duplicate report on buffer corruption

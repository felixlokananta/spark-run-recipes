# Qwen3.8-27B on DGX Spark

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — dense 27B hybrid-attention
vision-language model, Apache 2.0, 262,144-token native context (1M with YaRN).

Not a MoE, unlike everything else in this registry: all 27B parameters are active on every
token. What keeps it tractable on a Spark is the attention layout — `full_attention_interval: 4`
over `num_hidden_layers: 64`, so 16 layers run full attention and the other 48 run Gated
DeltaNet linear attention. It also ships a vision tower and a Multi-Token Prediction draft head
in the same checkpoint.

## Which recipe

| Recipe | Weights on disk | `gpu_memory_utilization` | Notes |
|---|---|---|---|
| `qwen3.8-27b-nvfp4-vllm-felixlokananta` | 23.4 GB | 0.70 | **Start here.** NVFP4 is native to GB10 Blackwell, and MTP is on. |
| `qwen3.8-27b-fp8-vllm-felixlokananta` | 30.9 GB | 0.75 | Official Qwen quant. Quality-safe middle tier. |
| `qwen3.8-27b-bf16-vllm-felixlokananta` | 55.6 GB | 0.85 | Reference quality. Fits, but it is the tight tier. |

All three serve the model as `qwen3.8-27b` on port 8000, single node, `-tp 1`, 262,144-token
context, multimodal.

```bash
sparkrun recipe validate recipes/qwen3.8-27b/felixlokananta/qwen3.8-27b-nvfp4-vllm-felixlokananta.yaml
sparkrun run recipes/qwen3.8-27b/felixlokananta/qwen3.8-27b-nvfp4-vllm-felixlokananta.yaml --solo --dry-run
sparkrun run recipes/qwen3.8-27b/felixlokananta/qwen3.8-27b-nvfp4-vllm-felixlokananta.yaml --solo
```

The NVFP4 recipe points at `unsloth/Qwen3.8-27B-NVFP4` — Qwen publishes BF16 and FP8 only.

## Why the flags look the way they do

**`--tool-call-parser qwen3_coder` + `--reasoning-parser qwen3`.** `chat_template.jinja` emits
the nested `<tool_call><function=name><parameter=x>…` XML form that the `qwen3_coder` parser
expects — the template even inlines the format spec into the system prompt — and wraps
reasoning in `<think>…</think>`. Same pairing as the KAT-Coder recipes.

**No `--language-model-only`.** Unlike KAT-Coder, the vision weights are actually in the
release. `config.json` sets `language_model_only: false`, carries a real `vision_config`
(27-layer encoder, patch 16, merge 2), and defines `image_token_id` / `video_token_id`. All
three checkpoints — including the unsloth NVFP4 one — ship the tower. Serve it multimodal or
don't serve it.

**`--limit-mm-per-prompt '{"image":4,"video":1}'`.** Multimodal preprocessing allocates outside
the KV budget, and video especially can balloon. On a 128 GB unified pool where host and device
compete, bounding it is cheaper than debugging it. Raise it if you're doing document batches.

**`--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`.** The template's default is
`xhigh`, which the model card backs with up to 262,144 thinking tokens before it starts on the
answer. That's a fine default on a rented H200 and a bad one at ~25 tok/s. `medium` is the
usable middle; the accepted values are `xhigh`, `medium`, `low`, and anything else makes the
template raise. Add `"enable_thinking":false` to turn reasoning off entirely.

Note there's no `"preserve_thinking":true` here, unlike the KAT-Coder recipes. Qwen's template
already preserves prior-turn thinking unless you explicitly pass `false`.

**`--kv-cache-dtype fp8` makes 262K context nearly free.** Only 16 layers feed the paged cache.
With `num_key_value_heads: 4` and `head_dim: 256` that's 16 × 2 × 4 × 256 = 32 KB/token at fp8,
so the full 262,144-token window costs 8.0 GB. `sparkrun recipe vram` agrees exactly. This is
also why `metadata.num_layers` is 16 rather than 64 — the estimator would otherwise overshoot
the KV figure by 4x.

**`--max-num-seqs 8` is the flag to watch, not `max_model_len`.** The 48 linear-attention layers
don't use the KV cache; each holds a fixed recurrent state per *sequence*:
48 v-heads × 128 v-dim × 128 k-dim × 4 bytes (`mamba_ssm_dtype: float32`) ≈ 3.1 MB per layer,
≈ 151 MB per sequence across all 48. At `max_num_seqs: 8` that's ~1.2 GB; at 32 it's ~4.8 GB
that no VRAM estimate here will show you. Concurrency costs more than context on this
architecture.

**`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (NVFP4 only).** The draft
head is trained in (`mtp_num_hidden_layers: 1`) and unsloth ships it as `model_mtp.safetensors`
in the same repo, so there's no second model to fetch. 3 is the vLLM recipe default; a Spark
measurement at `num_speculative_tokens: 5` reported roughly doubled decode versus no MTP, so
try 5 if your workload is decode-heavy and low-concurrency. It's left off the FP8 and BF16
recipes only because I haven't confirmed those repos expose the head the same way — add the
same line if it loads.

**`--attention-backend flashinfer`, `--load-format fastsafetensors`,
`VLLM_MARLIN_USE_ATOMIC_ADD=1`** — standard Spark tuning, same as the rest of the registry.

**`--load-format fastsafetensors` is kept on all three here.** The Spark has no GPUDirect
Storage, so fastsafetensors falls back to a host bounce buffer: it reads a whole file into host
memory before copying it to device, and host and device are the same pool. That's what killed
the KAT-Coder FP8 recipe. The shard shapes here are all safe:

| Recipe | `.safetensors` files | Largest file | Bounce-buffer peak |
|---|---|---|---|
| NVFP4 | 2 | 22.6 GB | ~46 GB |
| FP8 | 66 | 6.0 GB | ~37 GB |
| BF16 | 18 | 4.0 GB | ~60 GB |

The NVFP4 case is the only close one, and it's the same shape as the KAT NVFP4 checkpoint
(21.9 GB single file) that loads fine. If a load dies silently mid-`Loading … checkpoint shards`
with no traceback, that's the loader, not the memory budget — drop the flag or switch to
`--load-format runai_streamer`.

## Sampling

vLLM picks these up from `generation_config.json`; set them client-side if you override.

| Mode | Settings |
|---|---|
| Thinking (default) | `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0` |
| Non-thinking | `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5` |

## Tuning notes

- Requires `transformers >= 5.8.0` (the `config.json` is stamped `5.8.0.dev0`) and
  vLLM 0.17.0+. The `dgx-vllm-eugr-nightly` container is well past both.
- `sparkrun recipe vram` under-reports weights on every tier — 12.6 GB vs 23.4 GB actual for
  NVFP4, 25.2 vs 30.9 for FP8, 50.3 vs 55.6 for BF16. It applies nominal bits-per-param to a
  flat 27B and doesn't know about the vision tower, the MTP head, or the modules each
  checkpoint leaves at higher precision (Qwen's FP8 skips the first visual block; unsloth's
  NVFP4 keeps FP8 activations on the attention projections). Trust the on-disk column above.
- If startup OOMs, check *where*. After memory profiling it's a KV budget problem — lower
  `max_model_len` first, since KV scales linearly with it. Before that, it's the loader.
- BF16 at 0.85 leaves ~47 GB above the weights, which is comfortable in isolation but not if
  your Spark is also running a desktop session. Drop to 0.80 and `max_model_len` 131072 there.
- For the 1M-token window, add YaRN rope scaling per the model card and expect to cut
  `max_num_seqs` to 2–4. A Spark measurement reached ~778K KV tokens at
  `gpu_memory_utilization: 0.45` with `max_num_seqs: 4`.
- MXFP4 quants of this model don't work on NVIDIA — no linear-attention support. NVFP4 is the
  4-bit path.

## Sources

- Model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- vLLM recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-27B>
- NVFP4 on a single Spark, with measurements:
  <https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-a-single-dgx-spark-up-to-1m-context-a-tokenizer-bug-worth-knowing-about-and-measurements/380244>
- Recipe format: <https://sparkrun.dev/recipes/format/>
- vLLM runtime: <https://sparkrun.dev/runtimes/vllm/>

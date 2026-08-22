# Qwen3.8-27B-Uncensored on DGX Spark

[orcarouter/Qwen3.8-27B-Uncensored-FP8](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8)
— [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the refusal direction
abliterated, then quantized to block-FP8. Apache 2.0, 262,144-token context.

Architecturally identical to the stock [`qwen3.8-27b`](../../qwen3.8-27b/felixlokananta/)
recipes — dense 27B hybrid-attention VLM, `full_attention_interval: 4` over
`num_hidden_layers: 64`, vision tower and MTP draft head in the same checkpoint. Read that
README first; this one only covers what differs. What differs is two things: the checkpoint
is gated and abliterated, and MTP is turned up.

| Recipe | Weights on disk | `gpu_memory_utilization` | Context | MTP |
|---|---|---|---|---|
| `qwen3.8-27b-uncensored-fp8-vllm-felixlokananta` | 30.9 GB | 0.75 | 262,144 | aggressive ladder, 6 → 2 |

```bash
sparkrun recipe validate recipes/qwen3.8-27b-uncensored/felixlokananta/qwen3.8-27b-uncensored-fp8-vllm-felixlokananta.yaml
sparkrun run recipes/qwen3.8-27b-uncensored/felixlokananta/qwen3.8-27b-uncensored-fp8-vllm-felixlokananta.yaml --solo --dry-run
sparkrun run recipes/qwen3.8-27b-uncensored/felixlokananta/qwen3.8-27b-uncensored-fp8-vllm-felixlokananta.yaml --solo
```

Serves as `qwen3.8-27b-uncensored`, not `qwen3.8-27b`, so it can sit alongside the stock
recipes without a client silently landing on the abliterated weights.

## Before the first run: the repo is gated

`gated: auto` on the Hub. Two steps, both outside the recipe:

1. Accept the conditions on the model page while signed in. Auto-approve, so it is instant.
2. Have `HF_TOKEN` in the environment sparkrun itself runs in — it does the
   `snapshot_download` and propagates the token to nodes via `mpirun -x`.

**Do not put the token in the recipe's `env:` block.** Sparkrun stopped expanding recipe `env`
values against the control machine's environment specifically to keep this from working: a
`${HF_TOKEN}` in a recipe is delivered to the container as the literal seven-character string,
and the failure mode is a 401 that looks like the gate was never accepted.

## Aggressive MTP

The stock NVFP4 recipe runs `{"method":"mtp","num_speculative_tokens":3}` — the vLLM recipe
default. This one runs a batch-size-dependent ladder instead:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":6,
  "num_speculative_tokens_per_batch_size":[[1,2,6],[3,4,4],[5,6,3],[7,8,2]]}'
```

which expands to:

| Running batch size | Draft tokens per step |
|---|---|
| 1–2 | 6 |
| 3–4 | 4 |
| 5–6 | 3 |
| 7–8 | 2 |

**Why deeper than 3 pays on a Spark.** The draft head is a single decoder layer
(`text_config.mtp_num_hidden_layers: 1`), which vLLM reuses autoregressively to reach any
depth — `n_predict` is 1, and the only constraint is that `num_speculative_tokens` be
divisible by it, so every value is legal. At batch 1 the Spark is bandwidth-bound: a verify
pass reads ~31 GB of weights, one draft step reads ~0.4 GB. Each extra draft token costs on the
order of 1.3% of a step and can save a whole one. The usual argument against deep speculation —
that drafting competes with the target model for compute — is an argument about GPUs that are
compute-bound at batch 1. This one is not.

**Why it decays with batch size.** That argument inverts as sequences accumulate. Verification
work scales with `batch × K` while the win per sequence stays bounded by acceptance, so the
ladder walks K down rather than letting a busy server pay for drafts it will throw away. The
ranges are inclusive, must start at 1, and must not overlap; vLLM sorts them, carries the last
value forward past the end, and clamps every rung to `num_speculative_tokens` — which is why
that field is set to 6, the ladder's own maximum, and functions as a ceiling rather than a
setting. Set it lower than a rung and the rung is silently capped.

**Where the ceiling actually is.** 6 is an extrapolation, not a measurement. The measured point
on this architecture is `num_speculative_tokens: 5`, which roughly doubled decode against no MTP
on a Spark — see the sources in the stock recipe's README. Acceptance on a one-layer head falls
off with depth, and there is a point past which the sixth draft token is nearly never accepted
and you are paying for it every step. If you benchmark this and 6 is not beating 5, change the
first rung to `[1,2,5]` and the ceiling to 5; nothing else needs to move.

**Don't add `--async-scheduling`.** It is incompatible with speculative decoding. Prefix caching
and chunked prefill are both fine and both stay on.

**Memory.** MTP's cost here is the draft head's own KV, already priced into the measured
37,169 B/token that the stock README uses in place of the 32 KB/token the layer math gives.
262,144 tokens at that rate is ~9.1 GiB of cache on top of ~28.8 GiB of weights, so 0.75 of the
~120 GiB vLLM sees is not tight. The ladder itself adds `K` lookahead token slots per running
sequence — 8 × 6 × 37 KB, under 2 MB, not a number to plan around. If startup OOMs it is not
the speculative config; check the loader first, then `max_model_len`.

## What abliteration changed

A single refusal direction estimated at layer 38 (Arditi et al. 2024), orthogonalized out of
131 residual-writing matrices across attention, MLP, and the embeddings. Reported effect:

| | Base | This model |
|---|---|---|
| Refusal, thinking off | 64–99% | 0–6% |
| Refusal, thinking on | — | 0–1.7% |
| MMLU (300) | 84.3% | 84.7% |
| MMLU-Pro (250) | 77.6% | 76.8% |
| GSM8K (150) | 90.0% | 88.7% |
| CMMLU (500) | 81.4% | 80.8% |

Capability is within ±1.3 points across the board, which is the useful part: the recipe does not
need different sampling or a different reasoning budget to compensate. **The safety behavior is
genuinely gone.** This is a red-teaming and eval checkpoint. Don't put it behind a public
endpoint, and don't make it the default a coding client picks up by accident — hence the
distinct `--served-model-name`.

## Quantization differences from Qwen's own FP8

Same nominal scheme, differently drawn line:

| | `Qwen/Qwen3.8-27B-FP8` | `orcarouter/…-Uncensored-FP8` |
|---|---|---|
| Scheme | fine-grained FP8 E4M3 | block FP8 E4M3, `weight_block_size [128,128]` |
| Activations | dynamic per-token | dynamic per-token, no calibration set |
| Vision tower | block 0 left BF16 | **all 27 blocks left BF16** (333 tensors copied) |
| Layout | 66 shards, largest 6.01 GB | 7 shards, largest 4.97 GB |
| Total | 30.87 GB | 30.89 GB |
| Tensors | 1606 | 1606 |

The tensor counts match, which is the check that matters here — the MTP head survives
quantization as the same 22 `mtp.*` tensors the official FP8 checkpoint carries, so
`--speculative-config` has something to load. Keeping the whole vision tower in BF16 instead of
one block costs nothing measurable on disk and is why this serves multimodal as happily as the
stock recipes.

`--load-format fastsafetensors` is safe here for the reason it is safe on the stock FP8 recipe:
the Spark has no GPUDirect Storage, so the loader bounces through host memory at roughly
total + largest shard = ~36 GB peak. The 7-shard reshard is actually friendlier than Qwen's
66-shard layout. If a load dies silently mid-`Loading … checkpoint shards` with no traceback,
that's the loader — drop the flag or switch to `--load-format runai_streamer`.

## Sampling and template

`chat_template.jinja` is byte-identical to the base model's, so everything in the stock README
carries over unchanged: `qwen3_coder` tool parser, `qwen3` reasoning parser, and
`reasoning_effort` accepting `xhigh` / `medium` / `low` with `xhigh` as the template default.
The recipe pins `medium` for the same reason as the stock recipes — `xhigh` will happily spend
thousands of thinking tokens at Spark decode speeds. `{"enable_thinking":false}` turns reasoning
off entirely.

The model card gives no sampling guidance of its own. Use the base model's, which
`generation_config.json` should carry:

| Mode | Settings |
|---|---|
| Thinking (default) | `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0` |
| Non-thinking | `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5` |

## Sources

- Model card: <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8>
- Base model: <https://huggingface.co/Qwen/Qwen3.8-27B>
- Stock recipes and the Spark measurements: [`qwen3.8-27b`](../../qwen3.8-27b/felixlokananta/README.md)
- vLLM dynamic speculative decoding: `vllm/v1/spec_decode/dynamic/utils.py`
- Recipe format: <https://sparkrun.dev/recipes/format/>

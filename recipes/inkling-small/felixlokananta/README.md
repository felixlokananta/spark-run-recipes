# Inkling-Small on DGX Spark

[thinkingmachines/Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4) —
Thinking Machines Lab's open-weight multimodal MoE, Apache 2.0, 1,048,576-token native context.
276B total parameters, 12B active. Text, image and audio in; text out.

**This is the first cluster-only recipe in this registry.** The NVFP4 checkpoint is ~171 GB on
disk. One Spark has 128 GB of unified memory, so there is no `--solo` path — `min_nodes: 2`.
vLLM's own recipe puts the floor at 180 GB aggregated VRAM, which two Sparks (256 GB) clear and
one does not.

> **Untested.** Every number below is computed from `config.json`, `hf_quant_config.json` and the
> actual shard sizes in the repo, or taken from the upstream vLLM recipe. Nothing here has been
> measured on hardware yet. Treat the `gpu_memory_utilization` values as starting points.

## Which recipe

| Recipe | Nodes | `gpu_memory_utilization` | Context | Serves as |
|---|---|---|---|---|
| `inkling-small-nvfp4-vllm-felixlokananta` | 2 | 0.80 | 262,144 | `inkling-small` |
| `inkling-small-nvfp4-1m-vllm-felixlokananta` | 2 | 0.86 | 1,000,000 | `inkling-small-1m` |

Both are `-tp 2`, multimodal, on port 8000, off the same checkpoint. Start with the 262K one —
it comes up with ~20 GiB of slack per node, and the 1M one has about 6.

```bash
sparkrun recipe validate recipes/inkling-small/felixlokananta/inkling-small-nvfp4-vllm-felixlokananta.yaml
sparkrun run recipes/inkling-small/felixlokananta/inkling-small-nvfp4-vllm-felixlokananta.yaml --tp 2 --dry-run
sparkrun run recipes/inkling-small/felixlokananta/inkling-small-nvfp4-vllm-felixlokananta.yaml --tp 2
```

`--tp 2` has to be passed to `stop` and `logs` as well — it feeds the cluster ID sparkrun uses to
find the containers.

There is no FP8 or BF16 tier here. Thinking Machines publish BF16 (`thinkingmachines/Inkling-Small`)
and NVFP4 only, and BF16 at 276B is ~550 GB — five Sparks before you have anywhere to put the KV
cache. NVFP4 is the path. `hf_quant_config.json` shows it is a real W4A4 ModelOpt quant, and a
conservative one: the only tensors actually in NVFP4 are the routed experts of layers 3–41.
`exclude_modules` leaves everything else at higher precision — every attention block and its
short convolution, every norm, the embeddings and unembed, the router gates, the shared experts,
the vision and audio encoders, and all of layers 0–2 (layer 2's experts are excluded outright,
and layers 0–1 are dense MLP per `dense_mlp_idx: 2`). That is what the ~29 GB of BF16 and ~16 GB
of FP8 tensors inside the checkpoint are — and it is a good part of why the quality story on this
checkpoint is less fraught than a blanket 4-bit quant would be.

## Before you launch: the download

~171 GB has to land on **both** nodes — sparkrun syncs the model to every placed host. Budget the
disk on each Spark, and have `sparkrun setup cx7` done first; the tutorial's advice about
ConnectX-7 mattering for model sync is not decoration at this size.

| | |
|---|---|
| `model-0000{1..9}-of-00009.safetensors` | 166.3 GB |
| `mtp.safetensors` | 4.5 GB |
| **Total** | **170.8 GB** |

## Why the flags look the way they do

**`--tokenizer-mode inkling`, `--reasoning-parser inkling`, `--tool-call-parser inkling`.** All
three are required and all three are Inkling-specific; there is no generic parser that works here.
The model does not use `<think>` tags or any of the XML/JSON tool-call conventions the other
recipes in this registry rely on. It emits its own control tokens:

- reasoning is `<|message_model|><|content_thinking|>…<|end_message|>`
- a tool call is `<|message_model|>{name}<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`
- tool declarations go in as `<|message_system|>tool_declare<|content_xml|>[…]<|end_message|>`
- a turn ends on `<|content_model_end_sampling|>` (id 200006, the `eos_token_id`)

The tokenizer is tiktoken-backed (`tiktoken/tokenizer.model` is in the repo alongside
`tokenizer.json`), which is what `--tokenizer-mode inkling` is for. Substituting `qwen3_coder` or
`deepseek_v3` here doesn't degrade gracefully — it produces tool calls the parser silently drops.

**`--kernel-config.enable_flashinfer_autotune=False`.** Straight from the upstream vLLM recipe.
Keep it. Note that the nested `--kernel-config.*` flag syntax needs vLLM 0.26.0+; if the container
rejects the flag outright, the image is too old for Inkling anyway.

**`VLLM_USE_V2_MODEL_RUNNER=1` and `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`.** Also from the
upstream recipe. Inkling needs the V2 model runner — the relative-attention and short-convolution
serving paths live there.

**No `--attention-backend`, and no `--kv-cache-dtype fp8`.** Both are house style in the rest of
this registry and both are deliberately absent here. The attention stack is unusual enough
(relative attention with `d_rel: 16` / `rel_extent: 1024`, a 512-token sliding window on 35 of 42
layers, a `sconv_kernel_size: 4` short convolution on every attention and MLP block) that pinning
a backend is a good way to hit an unimplemented path. And `hf_quant_config.json` sets
`kv_cache_quant_algo: "none"` — the checkpoint was not calibrated for a quantized cache. There is
enough room without it; see the budget below. If you want the memory back, fp8 KV would halve the
per-token cost and is worth trying, but measure the quality hit rather than assuming there isn't one.

**No `--load-format fastsafetensors`.** The Spark has no GPUDirect Storage, so fastsafetensors
bounces each shard through a host buffer — and host and device are the same pool. With weights
already occupying ~77 GiB of ~120 GiB per node, a ~20 GB bounce buffer on top of that is the exact
failure that killed the KAT-Coder FP8 recipe. The default loader is slower and it finishes.

**`--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`.** `chat_template.jinja` carries
its own effort scale and defaults to `0.9` when nothing is passed:

| key | value |
|---|---|
| `none` | 0.0 |
| `minimal` | 0.1 |
| `low` | 0.2 |
| `medium` | 0.7 |
| `high` | 0.9 |
| `max` | 0.99 |

It emits this as a literal `<|message_system|><|content_text|>Thinking effort level: 0.7` message
ahead of the conversation. Anything outside that table raises, but a bare float in `[0.0, 0.99]`
is accepted too, so `{"reasoning_effort": 0.45}` is a legal way to sit between `low` and `medium`.
`0.9` is a fine default on a rented rack and a bad one at Spark decode speeds. `none` (0.0) turns
thinking off.

**`--limit-mm-per-prompt '{"image":4,"audio":2}'`.** Multimodal preprocessing allocates outside the
KV budget. Audio is 16 kHz WAV and the card asks for clips under ~2 minutes; images want each
dimension between 40px and 4096px, and the encoder is a hierarchical patch encoder at
`patch_size: 40`. Raise the limits if you're doing document or transcript batches.

**`--enable-prefix-caching`.** Cheap insurance on the 262K recipe and close to load-bearing on the
1M one. It is also the first flag to pull if startup asserts — the short-convolution state is the
part of this architecture most likely to disagree with block reuse.

## The memory budget

The KV cache is the surprise here: it is small. `text_config.local_layer_ids` lists 35 of the 42
decoder layers, and those run a 512-token sliding window. Only layers **5, 11, 17, 23, 29, 35 and
41** are global attention, and only those seven feed the paged cache.

```
global KV/token   = 7 layers x 2 (K+V) x 8 kv heads x 128 head_dim x 2 B (bf16) = 28,672 B
sliding-window    = 35 layers x 2 x 8 x 128 x 512 tokens x 2 B                  = 73.4 MB per sequence
```

The sliding-window cost is per *sequence*, not per token, and at `-tp 2` it is ~37 MB per node per
sequence. That is why `max_num_seqs` is 16 on the 262K recipe rather than the cautious 8 the
Qwen3.8 recipes use — concurrency is genuinely cheap on this architecture, unlike a Gated DeltaNet
one where it isn't.

Per node at `-tp 2`, in GiB of the ~120 GiB vLLM sees:

| | 262K recipe | 1M recipe |
|---|---|---|
| Weights (166.3 GB / 2) | 77.4 | 77.4 |
| Paged KV (14,336 B/token/node) | 3.5 | 13.4 |
| Activations + profiling | ~6 | ~6 |
| **Total** | **~87** | **~97** |
| `gpu_memory_utilization` budget | 95.8 (0.80) | 103.0 (0.86) |
| Slack | ~9 | ~6 |

`metadata.model_vram` is set explicitly because auto-detection has no chance on this
`config.json`: there is no top-level `num_hidden_layers`, the real shape is nested under
`text_config`, and the NVFP4 weights are stored as packed `uint8` — the HF metadata reports 125.6B
`U8` elements for what are actually ~251B four-bit weights. `num_layers` is set to 7 rather than 42
for the same reason as in the Qwen3.8 recipe: it is the number of layers that touch the paged cache,
and reporting 42 overstates KV by 6x.

With those two set, `sparkrun recipe vram` lands on the table above:

```
                        262K recipe          1M recipe
  Model weights         166.30 GB            166.30 GB
  KV cache                7.00 GB (262,144)   26.70 GB (1,000,000)
  Tensor parallel             2                    2
  Per-GPU total          86.65 GB             96.50 GB
  DGX Spark fit          YES                  YES
```

**Do not also set `metadata.kv_vram_per_token`.** The format reference documents it as "KV cache
bytes per token", but the implementation (`sparkrun/models/vram.py`) treats it as *gibibytes* per
token — it multiplies the value straight into the total and separately does
`value * 1024**3` to get bytes for display. Passing the honest byte count blows the estimate up by
a factor of 2^30 and the fit line flips to `EXCEEDS 121 GB` with `Max context tokens: 0`. Leaving it
unset lets `num_layers × num_kv_heads × head_dim × kv_dtype` compute the same 28,672 B/token the
layer math above gives, which is what the numbers in that block are.

If startup OOMs, check *where*. After memory profiling it is a KV problem — drop `max_model_len`,
it scales linearly. Before profiling it is the loader or the weights, and the answer is `--tp 4`.

## Going wider

`max_nodes: 4`. At `-tp 4` the weights fall to ~39 GiB per node and everything above stops being
tight — 1M context lands around 52 GiB per node total. If you have four Sparks, this is the
comfortable way to run the 1M recipe:

```bash
sparkrun run recipes/inkling-small/felixlokananta/inkling-small-nvfp4-1m-vllm-felixlokananta.yaml \
  --tp 4 -o gpu_memory_utilization=0.75
```

The shapes divide cleanly at 4: 8 KV heads, 32 attention heads, 256 routed experts. They do not
divide at 8 in the KV dimension, so `-tp 8` would need `--enable-expert-parallel` and a different
conversation. Whether the extra node buys throughput depends on how much of the all-reduce the
ConnectX-7 link absorbs; at 12B active parameters this model is not obviously bandwidth-bound, but
that is a measurement, not a claim.

## MTP is in the repo and not in the recipes

`mtp.safetensors` (4.5 GB) ships alongside the weights, and `config.json` carries an `mtp_config`
with `num_nextn_predict_layers: 8`. The model card lists MTP speculative decoding as supported.
It is left out of both recipes because the upstream vLLM recipe does not show the invocation and
I have not confirmed vLLM registers Inkling's draft head under the generic `mtp` method name.

If it does, the line is:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Budget for it: +2.2 GiB of weights per node, and the draft head's own two global layers
(`mtp_config.local_layer_ids` covers 6 of its 8, leaving indices 1 and 3 global) add
2 × 2 × 8 × 128 × 2 = 8,192 B/token, so +4,096 B/token/node at `-tp 2`. On the 262K recipe that is
~3.2 GiB of the ~9 GiB slack — fine. On the 1M recipe it is ~6.0 GiB against ~6 GiB of slack, so
raise to `-tp 4` rather than trying to squeeze it in.

## Sampling

The repo ships **no `generation_config.json`**, and neither the model card nor the vLLM recipe
publishes recommended sampling parameters. So there is nothing for vLLM to pick up and nothing for
this README to repeat honestly. Set temperature and top-p client-side and tune them yourself; the
Tinker Playground is the reference for what the model is supposed to sound like.

## Tuning notes

- Needs vLLM 0.26.0+ with Inkling support. `dgx-vllm-eugr-nightly:latest` is a sentinel image —
  sparkrun resolves it to the current published nightly at launch. If Inkling support is newer than
  the nightly, that is the failure you will see, and it is worth checking
  <https://recipes.vllm.ai/thinkingmachines/Inkling-Small> for the minimum before debugging memory.
- Head node selection is up to the scheduler, not your host ordering. `sparkrun status` tells you
  which Spark actually took the head role and is serving the endpoint.
- Prefill, not decode, is what you will feel on the 1M recipe. Keep the stable part of a long
  context at the *front* of the prompt so prefix caching actually hits.
- There is no YaRN and no rope override anywhere in these recipes, and that is not an omission —
  Inkling has no RoPE. Long context is relative attention plus log-scaled attention logits above
  128K tokens, trained in. Unlike the Qwen3.8 1M recipe, running short prompts against
  `inkling-small-1m` costs nothing in quality. The only reason it is a separate recipe is the
  memory posture.
- The MXFP4 variant (`EmbeddedLLM/Inkling-Small-MXFP4`) is the AMD path and needs AITER. Not
  relevant on GB10.

## Sources

- Model card: <https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4>
- BF16 weights: <https://huggingface.co/thinkingmachines/Inkling-Small>
- vLLM recipe: <https://recipes.vllm.ai/thinkingmachines/Inkling-Small>
- SGLang recipe: <https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-Small>
- Multi-node tensor parallelism on Spark: <https://sparkrun.dev/tutorials/multi-node/>
- Recipe format: <https://sparkrun.dev/recipes/format/>
- vLLM runtime: <https://sparkrun.dev/runtimes/vllm/>

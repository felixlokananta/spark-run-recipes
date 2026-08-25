# Ornith-1.5-35B-A3B on DGX Spark

[ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) —
Ornith's mid-size agentic-coding MoE, MIT, 262,144-token native context (1M with YaRN).
35B total parameters, ~3B active. Reasoning model: every assistant turn opens with `<think>`.

The headline against the rest of this registry is that **this one is not memory-constrained**.
23.4 GB of weights and a KV cache that costs 10 KB/token means a Spark has 90+ GiB of slack after
everything is loaded. The tuning question here is throughput and concurrency, not whether it fits.

> **Untested.** The numbers below come from `config.json`, `hf_quant_config.json`,
> `model.safetensors.index.json` and the shard sizes in the repo, or from the model card. None of
> it has been measured on hardware yet.

## Which recipe

| Recipe | Weights on disk | `gpu_memory_utilization` | Context | Serves as |
|---|---|---|---|---|
| `ornith-1.5-35b-a3b-nvfp4-vllm-felixlokananta` | 23.4 GB | 0.70 | 262,144 | `ornith-1.5-35b-a3b` |
| `ornith-1.5-35b-a3b-nvfp4-1m-vllm-felixlokananta` | 23.4 GB | 0.70 | 1,000,000 | `ornith-1.5-35b-a3b-1m` |

Both are single node, `-tp 1`, multimodal, MTP on, on port 8000.

```bash
sparkrun recipe validate recipes/ornith-1.5-35b-a3b/felixlokananta/ornith-1.5-35b-a3b-nvfp4-vllm-felixlokananta.yaml
sparkrun run recipes/ornith-1.5-35b-a3b/felixlokananta/ornith-1.5-35b-a3b-nvfp4-vllm-felixlokananta.yaml --solo --dry-run
sparkrun run recipes/ornith-1.5-35b-a3b/felixlokananta/ornith-1.5-35b-a3b-nvfp4-vllm-felixlokananta.yaml --solo
```

There is no FP8 or BF16 tier here. Ornith publish BF16 (`ornith-ai/Ornith-1.5-35B-A3B`, ~70 GB —
it would fit a Spark at `gpu_memory_utilization: 0.85`, and if you want reference quality that is
the repo to point at) and a GGUF build. This NVFP4 quant is the one that leaves room to work in.

## What the quant actually did

`quant_algo` is `MIXED_PRECISION`, not a blanket NVFP4, and the split is worth knowing:

| Tier | What's in it |
|---|---|
| **NVFP4** (W4A16, group 16) | the 256 routed experts and the shared expert of all 40 layers, plus `lm_head` |
| **FP8** (W8A8, static) | every attention projection — `self_attn.{q,k,v,o}_proj` on the 10 full-attention layers, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` on the 30 linear ones |
| **BF16** (untouched) | the whole 27-layer vision tower, the embeddings, every norm, the router gates, and the entire MTP head |

That last one matters twice. `hf_quant_config.json` lists `mtp*` in `ignore`, and the index
confirms 785 `mtp.*` tensors sitting in shard 3 at full precision — so the draft head is present,
unquantized, and roughly 1.8 GB of the 23.4. And the vision tower being excluded outright
(`.quant_summary.txt` shows every `model.visual.*` quantizer disabled) means multimodal quality is
not what this quant traded away.

## Why the flags look the way they do

**`--kv-cache-dtype fp8` is calibrated here, not a guess.** `config.json` carries
`kv_cache_scheme: {num_bits: 8, type: float}` and `hf_quant_config.json` sets
`kv_cache_quant_algo: FP8`. Most NVFP4 quants leave this at `none` and fp8 KV is then something you
opt into and measure; this checkpoint was calibrated for it. Take the free memory.

**`--tool-call-parser qwen3_xml` + `--reasoning-parser qwen3`.** `chat_template.jinja` inlines the
format spec into the system prompt and emits the nested
`<tool_call><function=name><parameter=x>…` XML form, and wraps reasoning in `<think>…</think>`.
The model card names `qwen3_xml` for vLLM (and `qwen3_coder` for SGLang). Both vLLM parsers read
that same XML; `qwen3_xml` is the newer one. **If the container's vLLM predates it, swap in
`qwen3_coder`** — that is the parser the other recipes in this registry use for the identical
format, and it is a one-word override:

```bash
sparkrun run … -o tool_call_parser=qwen3_coder
```

**No `--default-chat-template-kwargs`.** Unlike the Qwen3.8 recipes there is no `reasoning_effort`
knob in this template — thinking is either on or off. `{"enable_thinking": false}` emits an empty
`<think>\n\n</think>` block and skips the chain of thought. Note also that this template
*preserves* prior-turn reasoning: it re-emits `message.reasoning_content` for every past assistant
turn, and falls back to splitting it out of `content` if the client didn't send it separately. That
is not the Qwen default, and it means multi-turn agent traces grow faster here than you might
expect.

**Serve it multimodal.** The model card is silent on vision — its frontmatter even says
`pipeline_tag: text-generation` — but the architecture is `Qwen3_5MoeForConditionalGeneration`, the
config carries a real 27-layer `vision_config` (patch 16, merge 2, `out_hidden_size` 2048) plus
`image_token_id` and `video_token_id`, there are 333 `model.visual.*` tensors in the index, and the
repo ships a `Qwen3VLVideoProcessor` config. `--limit-mm-per-prompt '{"image":4,"video":1}'` bounds
preprocessing, which allocates outside the KV budget and is the cheaper thing to cap than debug.

**`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.** `mtp_num_hidden_layers: 1`,
same shape as the Qwen3.8-27B recipe, and the head is in the same three shards — no second model to
fetch. 3 is the vLLM default; a Spark measurement on the Qwen3.8 checkpoint reported roughly doubled
decode at `num_speculative_tokens: 5`, so try 5 if your workload is decode-heavy and
low-concurrency. Note the draft head here is a full 256-expert MoE layer, so it is a real 1.8 GB —
still cheap against 90 GiB of slack.

**`--attention-backend flashinfer`, `--load-format fastsafetensors`,
`VLLM_MARLIN_USE_ATOMIC_ADD=1`, `FLASHINFER_DISABLE_VERSION_CHECK=1`** — standard Spark tuning, same
as the rest of the registry. fastsafetensors is safe here: 3 shards, largest 10.0 GB, so the host
bounce buffer peaks well under the failure that killed the KAT-Coder FP8 recipe.

## The memory budget, and why `gpu_memory_utilization` is 0.70

Per node, of the ~120 GiB vLLM sees:

| | 262K recipe | 1M recipe |
|---|---|---|
| Weights | 21.8 GiB | 21.8 GiB |
| Paged KV for one full window | 2.6 GiB | 10.5 GiB |
| Gated DeltaNet state | 2.0 GiB (32 seqs) | 0.5 GiB (8 seqs) |
| Activations + profiling | ~5 GiB | ~5 GiB |
| **Total** | **~31 GiB** | **~38 GiB** |

0.70 gives an ~84 GiB budget, so in both cases vLLM sizes the KV pool at *many times* one context
window — roughly 5.5M tokens of cache on the 262K recipe. That is the point. With
`--enable-prefix-caching` on, a pool that holds twenty-odd full-length sessions is what lets a
coding agent keep long repository contexts warm across requests. Context length is not the
constraint on this model; concurrent live context is.

Two knobs follow from that:

- **`--max-num-seqs 32`.** The 30 linear-attention layers don't use the KV cache; each holds a fixed
  recurrent state per *sequence*: 32 v-heads × 128 v-dim × 128 k-dim × 4 bytes
  (`mamba_ssm_dtype: float32`) ≈ 2.1 MB per layer, ≈ 63 MB per sequence across all 30. At 32 that is
  ~2.0 GiB — invisible here, where on the Qwen3.8-27B recipe the same arithmetic forced
  `max_num_seqs: 8`. Push it higher if you have the request volume.
- **Drop `gpu_memory_utilization` if you're sharing the box.** 0.50 still leaves ~28 GiB of KV pool,
  which is ~2.9M tokens, and gives back 24 GiB to your desktop session or a second workload.

`metadata.num_layers` is 10 rather than 40 for the same reason as in the Qwen3.8 recipes: only the
full-attention layers (`layer_types` indices 3, 7, 11 … 39) touch the paged cache, and reporting 40
would overstate KV by 4x. `model_vram` is pinned too — the estimator applies nominal bits-per-param
to a flat 35B and has no way to know this is a `MIXED_PRECISION` quant with a BF16 vision tower and
a BF16 MTP head inside it. With both set, `sparkrun recipe vram` reports:

```
                        262K recipe          1M recipe
  Model weights          23.40 GB             23.40 GB
  KV cache                2.50 GB (262,144)    9.54 GB (1,000,000)
  Per-GPU total          25.90 GB             32.94 GB
  DGX Spark fit          YES                  YES
  Context multiplier     24.5x                 6.4x
```

That "context multiplier" is the headroom point restated: at these settings the KV pool could hold
24 full 262K windows. The estimator doesn't count the Gated DeltaNet state or activations, so add
~7 GiB to its per-GPU totals to get the table above.

**Do not also set `metadata.kv_vram_per_token`.** The format reference calls it "KV cache bytes per
token", but the implementation (`sparkrun/models/vram.py`) treats it as *gibibytes* per token, so an
honest byte count inflates the estimate by 2^30 and the fit line flips to `EXCEEDS 121 GB`. Left
unset, `num_layers × num_kv_heads × head_dim × kv_dtype` computes the same 10,240 B/token the layer
math gives.

## The 1M recipe, and a correction to the model card

`ornith-1.5-35b-a3b-nvfp4-1m-vllm-felixlokananta` is the same checkpoint with rope overridden at
launch. **The model card's snippet will not work on this checkpoint as written.** It shows:

```json
{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}
```

Two problems. First, `rope_scaling` is the pre-transformers-5 key; this `config.json` is stamped
`transformers_version: 5.5.4` and carries a `rope_parameters` dict instead. Second, the override
*replaces* the dict rather than merging into it, and the existing one is doing real work:

```json
"rope_parameters": {
  "mrope_interleaved": true, "mrope_section": [11, 11, 10],
  "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"
}
```

Drop those and you lose multimodal position encoding along with the model's actual rope base. So
the recipe restates all of them, nested under `text_config` because the top-level config is the
multimodal wrapper:

```
--hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
  "original_max_position_embeddings":262144,"mrope_interleaved":true,
  "mrope_section":[11,11,10],"rope_theta":10000000,"partial_rotary_factor":0.25}}}'
```

Same shape as the Qwen3.8-27B 1M recipe, and for the same reasons. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
is in `env` because `max_position_embeddings` is 262144.

Only the 10 full-attention layers have RoPE at all — the 30 Gated DeltaNet layers carry a recurrent
state with no positional encoding to scale — which is a large part of why this extension is cheap.

**Don't make this your default endpoint.** Every open-source framework implements *static* YaRN:
factor 4.0 is applied to every prompt regardless of length, so a 3K-token chat pays the same
interpolation as a 900K-token document and measurably loses quality for it. The model card says this
too, and adds the right rule — size `factor` to your actual workload. If your requests top out
around 524,288 tokens, `factor: 2.0` is the better setting:

```bash
sparkrun run …-1m-… --solo \
  -o max_model_len=524288 \
  -o hf_overrides='{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_theta":10000000,"partial_rotary_factor":0.25}}}'
```

Run the 262K recipe day to day and bring this one up when you actually have the document.

## Sampling

`generation_config.json` ships `temperature: 1.0, top_p: 0.95, top_k: 20` — that is the model card's
**benchmark-reproduction** setting, not its general-use one. vLLM will pick it up as the default, so
the card's recommendation for ordinary work has to be set client-side:

| Mode | Settings |
|---|---|
| General tasks (card's recommendation) | `temperature=0.6, top_p=0.95, top_k=20` |
| Reproducing the reported benchmarks | `temperature=1.0, top_p=0.95, top_k=20` |

`eos_token_id` is a list — `[248046, 248044]` — so a client that hardcodes a single stop token from
`config.json` (which lists only 248046) will occasionally overrun. Let the server handle stopping.

## Tuning notes

- The model card asks for transformers ≥ 5.8.1, vLLM ≥ 0.19.1, SGLang ≥ 0.5.9. The
  `dgx-vllm-eugr-nightly` sentinel image resolves to the current published nightly and is well past
  all three. (The `transformers_version: 5.5.4` stamped in `config.json` is just what ModelOpt 0.45.0
  wrote at quantization time — go by the card.)
- `max_nodes: 2`. There is no memory reason to go multi-node, but `-tp 2` is available if you want
  the extra bandwidth for a heavily concurrent agent workload. Don't go past 2:
  `num_key_value_heads` is 2, and beyond that vLLM has to replicate KV heads to fill the ranks,
  which costs memory to buy nothing.
- 256 experts at `moe_intermediate_size: 512` is a fine-grained MoE. On one GPU there is no expert
  parallelism to configure; if you do run `-tp 2`, `--enable-expert-parallel` is the flag to
  benchmark against plain TP.
- If startup OOMs at these settings, something else is on the box — this recipe needs about a
  quarter of a Spark. Check `sparkrun cluster monitor` before touching `max_model_len`.
- GGUF builds exist (`ornith-ai/Ornith-1.5-35B-A3B-GGUF`) if you'd rather run this under llama.cpp
  like the Qwen3.6-35B recipes in this registry. NVFP4 under vLLM is the faster path on GB10.

## Sources

- Model card: <https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4>
- BF16 weights: <https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B>
- Ornith-1.5 blog: <https://ornith.ai/ornith_1_5.html>
- Recipe format: <https://sparkrun.dev/recipes/format/>
- vLLM runtime: <https://sparkrun.dev/runtimes/vllm/>

# KAT-Coder-V2.5-Dev on DGX Spark

[Kwaipilot/KAT-Coder-V2.5-Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev) —
35B total / 3B active MoE agentic coding model, Apache 2.0, 262,144-token native context.

## Which recipe

| Recipe | Weights | `gpu_memory_utilization` | Notes |
|---|---|---|---|
| `kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta` | ~22 GB | 0.70 | **Start here.** NVFP4 is native to GB10 Blackwell. |
| `kat-coder-v2.5-dev-fp8-vllm-felixlokananta` | ~36 GB | 0.75 | Quality-safe middle tier. |
| `kat-coder-v2.5-dev-bf16-vllm-felixlokananta` | ~69 GB | 0.85 | Reference quality; tight on a 128 GB Spark. |

All three serve the model as `kat-coder-v2.5-dev` on port 8000, single node, `-tp 1`.

```bash
sparkrun recipe validate recipes/kat-coder-v2.5-dev/felixlokananta/kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta.yaml
sparkrun run recipes/kat-coder-v2.5-dev/felixlokananta/kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta.yaml --solo --dry-run
sparkrun run recipes/kat-coder-v2.5-dev/felixlokananta/kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta.yaml --solo
```

## Why the flags look the way they do

**`--tool-call-parser qwen3_coder` + `--reasoning-parser qwen3`** — straight from the
model card. KAT's `chat_template.jinja` emits the nested
`<tool_call><function=name><parameter=x>…` XML form that the `qwen3_coder` parser expects,
and wraps reasoning in `<think>…</think>`.

**`--language-model-only`** (NVFP4 and BF16 recipes) — `config.json` declares a
`vision_config` and the architecture is `Qwen3_5MoeForConditionalGeneration`, but the
open-weight release ships language-model weights only. Without the flag vLLM tries to build
the vision tower. The FP8 quant was republished as `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text`,
so it doesn't need it.

**`--kv-cache-dtype fp8` with 256K context is cheap here.** The architecture is a
Qwen3.5-MoE hybrid: of 40 decoder layers only every 4th is full attention, so just 10 layers
feed the paged KV cache. With `num_key_value_heads: 2` and `head_dim: 256` that's
~10 KB/token at fp8 — 2.5 GB for the full 262,144-token window. The other 30 layers
are linear (gated-delta) attention with a constant-size recurrent state per sequence, which is
why `metadata.num_layers` is set to 10 rather than 40: sparkrun's VRAM estimate would
otherwise overshoot the KV figure by 4x.

**`--default-chat-template-kwargs '{"preserve_thinking":true}'`** — KAT's template drops
thinking traces from prior turns unless this is set. The model card recommends preserving them
for multi-turn agentic work, which is what it was RL-trained on. Drop it if you want shorter
prompts; add `"enable_thinking":false` to turn reasoning off entirely.

**`--attention-backend flashinfer`, `--load-format fastsafetensors`,
`VLLM_MARLIN_USE_ATOMIC_ADD=1`** — the standard Spark tuning that the official
`qwen3.5-35b-a3b` / `qwen3.6-35b-a3b` recipes use for this same architecture family.

**No `--load-format` on the FP8 recipe.** The Spark has no GPUDirect Storage, so
fastsafetensors falls back to a host bounce buffer — it reads a whole file into host memory
before copying it to the device. All three checkpoints matter here:

| Recipe | `.safetensors` files | Largest file | Bounce-buffer peak |
|---|---|---|---|
| NVFP4 | 1 | 21.9 GB | ~44 GB |
| FP8 | 1 | **35.8 GB** | **~72 GB** |
| BF16 | 13 | ~5.4 GB | ~11 GB above weights |

Host and device are the same 128 GB LPDDR5X pool on a Spark, so the FP8 single-file case
pays for the weights twice at once and the engine core gets killed mid-load. The failure is
silent — no Python traceback, just
`Engine core initialization failed … Failed core proc(s): {}` right after
`Loading fastsafetensors checkpoint shards: 0/1`. vLLM's default loader mmaps the file and
copies tensor-by-tensor, so its peak overhead is one tensor. Sharded BF16 and the smaller
NVFP4 file both stay under the limit, which is why only FP8 drops the flag. Use
`--load-format runai_streamer` instead if you want the load speed back.

## Tuning notes

- If startup OOMs, first check *where*. An OOM after the memory-profiling step is a KV-cache
  budget problem: lower `max_model_len` before touching `gpu_memory_utilization`, since KV
  scales linearly with it. A silent death *during* `Loading … checkpoint shards` is the
  loader, not the budget — drop `--load-format fastsafetensors` (see above) and leave
  `gpu_memory_utilization` alone.
- `sparkrun show` reports the BF16 recipe at 65.2 GB of weights against a 102.8 GB budget
  (0.85 of the ~121 GB usable pool), leaving ~37 GB. That's comfortable in isolation but leaves
  little for the host — if your Spark is running a desktop session, drop
  `gpu_memory_utilization` to 0.80 and `max_model_len` to 131072.
- The upstream `eugr` mods `mods/fix-qwen3-coder-next` (Triton allocator + hybrid-attention
  crash/slowness fixes) may help this architecture. They're deliberately not wired in here so
  the recipes stand alone — add them via `mods:` if you have the eugr registry installed.
  Do **not** apply `mods/fix-qwen3.5-chat-template`: it overwrites the chat template with
  Qwen's, which breaks KAT's tool-call format.

## Sources

- Model card: <https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev>
- Recipe format: <https://sparkrun.dev/recipes/format/>
- vLLM runtime: <https://sparkrun.dev/runtimes/vllm/>

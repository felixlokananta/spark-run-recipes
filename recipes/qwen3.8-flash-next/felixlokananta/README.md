# Qwen3.8-Flash-Next on DGX Spark

[nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) —
NVIDIA's quantization-aware-distilled FP4 build of Qwen3.8-Flash-Next. 125B parameters with ~6B
active, plus a 51B n-gram embedding table and a 4B MTP head. Text, image and video in; text out.
262,144-token native context. Single node.

> **Untested here.** It validates and dry-runs, but has not been launched on this registry's
> hardware. Sizes are measured from safetensors headers; performance numbers are other people's and
> are attributed below.

## Why not FP8

The request was `unsloth/Qwen3.8-Flash-Next-FP8`. It does not fit one Spark:

| | unsloth FP8 | FP4-Local (this recipe) |
|---|---|---|
| Total weights | 172.8 GiB | 95.1 GiB |
| N-gram (PLE) table | 47.7 GiB | 26.8 GiB |
| **Resident with the table on NVMe** | **125.1 GiB** | **68.3 GiB** |

A Spark exposes about 121 GiB. Leaving out MTP (2.5 GiB) and vision (0.8 GiB) as well still leaves the
FP8 build at ~122 GiB, with no KV cache. FP8 needs two nodes.

NVIDIA reports GPQA Diamond 91.5% for FP4-Local against 92.2% for BF16, and ToolEvalBench 88.7%
against 89.4%.

## Run it

```bash
sparkrun recipe validate recipes/qwen3.8-flash-next/felixlokananta/qwen3.8-flash-next-fp4-local-vllm-felixlokananta.yaml
sparkrun run recipes/qwen3.8-flash-next/felixlokananta/qwen3.8-flash-next-fp4-local-vllm-felixlokananta.yaml --solo --dry-run
sparkrun run recipes/qwen3.8-flash-next/felixlokananta/qwen3.8-flash-next-fp4-local-vllm-felixlokananta.yaml --solo
```

Serves as `qwen3.8-flash-next` on port 8000. Needs ~96 GiB of disk for the checkpoint, on local NVMe:
the n-gram table is read from the checkpoint files during serving, so a network mount will hurt.

Stop other GPU workloads first. A separate single-Spark setup with the same image and checkpoint
wants ~90 GiB `MemAvailable` before launch and sees it dip to 3-4 GiB for up to 10 seconds during load.

## Why the flags look the way they do

**The container is pinned.** The checkpoint compresses its NVFP4 scales in a format NVIDIA says
upstream vLLM cannot read yet, so `dgx-vllm-eugr-nightly` is out. `eugr/spark-vllm-b12x:nightly-20261015`
is the tag NVIDIA measured with. The kernel and loader flags (`b12x`, `modelopt_mixed`,
`--mamba-cache-mode align`) and the whole `env:` block are NVIDIA's Spark command unchanged.
On sparkrun versions with a b12x sentinel, naming `:latest` instead can be redirected to spark-arena's
own b12x nightly; the dated tag is not a sentinel.

**`seccomp=unconfined`.** With `VLLM_PLE_TABLE_MEMORY=disk` the table is read through io_uring, which
Docker's default seccomp profile blocks. NVIDIA's command disables the profile, and so does this recipe;
`no-new-privileges` and the non-root user stay. This does widen the container's syscall surface.

- sparkrun 0.2.40 needs it: rootless mode otherwise sets only `no-new-privileges`.
- sparkrun 0.4.2 allows io_uring by default, so the block can be deleted there. If it stays, running
  through a registry (`@felix/…`) may need `--trust`; running from a local path does not.
- NVIDIA also sets `apparmor=unconfined`. Left out because the known blocker is seccomp. If startup fails
  on io_uring with a permission error, add `apparmor=unconfined` to `security_opt`.

To rule io_uring out altogether, remove `VLLM_PLE_TABLE_MEMORY` and `B12X_DISK_BACKEND`. NVIDIA says the
Spark can run without offload, at the cost of 26.8 GiB of KV room.

**What differs from NVIDIA's command:** a served model name, `--limit-mm-per-prompt`, and
`reasoning_effort: medium` (the template default is `xhigh`; this matches the `qwen3.8-27b` recipes).
For NVIDIA's behavior: `-o default_chat_template_kwargs='{"reasoning_effort":"xhigh"}'`.

**`CUTE_DSL_ARCH`** draws a `restated-platform-env` suggestion from sparkrun 0.4.2, which sets it
itself. 0.2.40 does not, so it stays.

## Memory

`sparkrun recipe vram` reports 68.3 GB weights, 3.0 GB KV for one 262K window, 28.5 GB for KV at 0.80, and
about 2.5M tokens of capacity. **The token figure is optimistic**: it counts 12 full-attention layers at
12,288 B/token and omits the sparse-attention indexer, the MTP layer, Gated DeltaNet state, and activations.

That other Spark setup measured 1.05-1.10M tokens at `gpu_memory_utilization: 0.8` with 3 MTP tokens, or about
four full windows. Pinning a 31.15 GB pool got it 1.58M tokens, but `MemAvailable` fell to 6.9 GiB.

If loading pushes the box into memory pressure, eugr's current recipe adds
`PYTORCH_ALLOC_CONF: "per_process_memory_fraction:0.8"` to `env:`. It is not set here because that fix
postdates the pinned image and has not been checked against it.

## Performance, as reported by others

| Source | Setup | Result |
|---|---|---|
| [NVIDIA](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html) | this image and engine flags, MTP 4, SPEED-Bench coding, 1 Spark | 37.8 tok/s decode |
| [devopsfortunato](https://github.com/devopsfortunato/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark) | image derived from the same digest with a deeper PLE read queue, MTP 3, thinking off | 41.9 tok/s prose, 53.8 code, ~2,100 tok/s prefill; 119 tok/s total at 8 streams |

The two use different workloads, so they are not comparable with each other. devopsfortunato also found 3
MTP tokens as fast as 4 with 4-9% more KV cache; try
`-o speculative_config='{"method":"mtp","num_speculative_tokens":3}'`.

## Sampling

`generation_config.json` ships the thinking-mode settings. Qwen's recommendations:

| Mode | Settings |
|---|---|
| Thinking (default) | `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0` |
| Non-thinking (`{"enable_thinking":false}`) | `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5` |

`reasoning_effort` accepts only `xhigh`, `medium` and `low`; the template raises on anything else.
Prior-turn reasoning is kept by default; pass `{"preserve_thinking":false}` to drop it.

## License

The quant is under NVIDIA's Open Model Agreement; the base model is under the Qwen Community License 1.0.

## Sources

- Model card and Spark command: <https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local>
- FP8 checkpoint: <https://huggingface.co/unsloth/Qwen3.8-Flash-Next-FP8>
- NVIDIA playbook script: <https://github.com/NVIDIA/dgx-spark-playbooks>
- eugr's solo recipe: <https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-flash-next-nvfp4-solo.yaml>
- sparkrun recipe reference: <https://github.com/spark-arena/sparkrun/blob/main/RECIPES.md>

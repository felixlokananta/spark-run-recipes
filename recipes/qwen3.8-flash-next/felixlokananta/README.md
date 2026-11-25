# Qwen3.8-Flash-Next on DGX Spark

[nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) —
NVIDIA's quantization-aware-distilled FP4 build of Qwen3.8-Flash-Next. 125B parameters with ~6B
active, plus a 51B n-gram embedding table and a 4B MTP head. Text, image and video in; text out.
262,144-token native context. Single node.

> **Measured on a single Spark** on 2026-11-24/25 with sparkrun 0.2.40. Figures are from this registry's
> hardware unless attributed to someone else.

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

Stop other GPU workloads first. The runs here started with 107-116 of 121.6 GiB free. A separate single-Spark
setup with the same image and checkpoint wants ~90 GiB `MemAvailable` before launch and sees it dip to
3-4 GiB for up to 10 seconds during load.

### Startup time

| Phase | First launch, no cache paths | Filling the cache | Relaunch |
|---|---|---|---|
| Model load | 3 min 4 s | 2 min 40 s | 2 min 34 s |
| b12x weights preparation | 4 min 48 s | 4 min 50 s | **7 s** |
| b12x state preparation | 4 min 46 s | 4 min 49 s | **2 s** |
| JIT kernel warmup | 45 s | 45 s | **0.24 s** |
| **vLLM banner to server start** | **15 min 56 s** | **15 min 30 s** | **4 min 0 s** |

The very first launch also downloads 102 GB (23 min here, 27 min for the whole distribution step). On a
launch without a warm cache the log sits in `b12x autotuning` / `b12x compiling` for minutes at a time;
that is normal.

**The cache paths in `env:`.** NVIDIA's command mounts `~/.cache/{vllm,flashinfer,b12x}` and `~/.triton`.
From its source, rootless sparkrun 0.2.40 sets `HOME=/tmp`, mounts only the Hugging Face cache, and removes
the container when it stops, so without these paths compiled kernels and autotuning results are lost. The
recipe points them at `/cache/huggingface/sparkrun-jit`, inside the mount that persists. The relaunch
reported 207 and 9 `cached`, with 0 `measured` and 0 `compilations`. The cache is ~470 MB; nothing was
written under `CUTE_DSL_CACHE_DIR`.

On the host it is `sparkrun-jit` inside your Hugging Face home: `$HF_HOME` if set, else
`~/.cache/huggingface`, unless sparkrun's own `cache_dir` / `hf_cache_dir` settings override it. Set
`HF_HOME` with `$HOME`, not a quoted `~`, which the shell does not expand. Deleting the directory clears
the cache, and the next launch is slow again.

Remove the five paths on sparkrun 0.4.2: it keeps these caches itself and flags them as
`managed-cache-env`.

## Why the flags look the way they do

**The container is pinned.** The checkpoint compresses its NVFP4 scales in a format NVIDIA says
upstream vLLM cannot read yet, so `dgx-vllm-eugr-nightly` is out. `eugr/spark-vllm-b12x:nightly-20261015`
is the tag NVIDIA measured with. The kernel and loader flags (`b12x`, `modelopt_mixed`,
`--mamba-cache-mode align`) and the `env:` block, apart from the cache paths, are NVIDIA's Spark command
unchanged.
On sparkrun versions with a b12x sentinel, naming `:latest` instead can be redirected to spark-arena's
own b12x nightly; the dated tag is not a sentinel.

**`seccomp=unconfined`.** With `VLLM_PLE_TABLE_MEMORY=disk` the table is read through io_uring, which
Docker's default seccomp profile blocks. NVIDIA's command disables the profile, and so does this recipe;
`no-new-privileges` and the non-root user stay. This does widen the container's syscall surface.

- sparkrun 0.2.40 needs it: rootless mode otherwise sets only `no-new-privileges`.
- sparkrun 0.4.2 allows io_uring by default, so the block can be deleted there. If it stays, running
  through a registry (`@felix/…`) may need `--trust`; running from a local path does not.
- 0.2.40 ran it through the registry (`@felix/…`) without `--trust`.
- NVIDIA also sets `apparmor=unconfined`. Left out because the known blocker is seccomp, and startup here
  completed without it. If io_uring fails with a permission error, add it to `security_opt`.

To rule io_uring out altogether, remove `VLLM_PLE_TABLE_MEMORY` and `B12X_DISK_BACKEND`. NVIDIA says the
Spark can run without offload, at the cost of 26.8 GiB of KV room.

**`table_memory='device'` in the log.** vLLM prints that in its Engram config even though the recipe asks
for `disk`. The sizes say the table is not resident: the loader read 72.7 GB of the 102 GB checkpoint
for the main model, and vLLM counted 70-77 GiB used, against ~97 GiB or more with the table in memory. No
log line names the io_uring backend, so that part is inferred.

**`block_size: 16` does not survive.** vLLM raises the attention block to about 3,000 tokens (3,024 at
MTP 4, 3,008 at MTP 3) to match the recurrent-state page, so ordinary prefix-cache hits land on those
boundaries. It also keeps
request-boundary checkpoints for supported chat requests.

**What differs from NVIDIA's command:** the cache paths, 3 MTP tokens instead of 4, a served model name,
`--limit-mm-per-prompt`, and `reasoning_effort: medium` (the template default is `xhigh`; this matches the `qwen3.8-27b` recipes).
For NVIDIA's behavior: `-o default_chat_template_kwargs='{"reasoning_effort":"xhigh"}'`.

**`CUTE_DSL_ARCH`** draws a `restated-platform-env` suggestion from sparkrun 0.4.2, which sets it
itself. 0.2.40 does not, so it stays.

## Memory

Measured here at `gpu_memory_utilization: 0.8`, MTP 4, over three launches:

| | First launch | Second | Third |
|---|---|---|---|
| Free at start | 107.1 GiB | 115.4 GiB | 115.5 GiB |
| Model load, including the MTP draft | 70.32 GiB | 70.32 GiB | 70.32 GiB |
| Counted as used | 70.4 GiB | 77.1 GiB | 77.2 GiB |
| KV cache | 24.2 GiB | 17.6 GiB | 17.4 GiB |
| **KV capacity, tokens** | 1,288,711 | 935,273 | **928,100** |
| Full 262K windows | 4.92 | 3.57 | 3.54 |

**Plan on about 970K tokens.** The default MTP 3 launch got 967,152 (3.69 windows); the last two MTP 4
launches above agree to within 1%. The first is probably inflated: it started with ~8 GiB more in use, and
if that was released during startup vLLM would have counted too little as its own. That cause is a guess. Cost per token is the same in all three, about 20,200 B.

`sparkrun recipe vram` predicts 28.5 GB for KV and 2.49M tokens, **about 2.6x too many**: it counts 12
full-attention layers at 12,288 B/token and omits the sparse-attention indexer, the MTP layer, Gated
DeltaNet state, and activations.

vLLM's log suggests a `--kv-cache-memory` of 34-36 GiB to "fully utilize" memory. Don't: another Spark
setup pinned a 31.15 GB pool for 1.58M tokens and `MemAvailable` fell to 6.9 GiB. That setup got 1.05-1.10M
tokens at utilization 0.8 with 3 MTP tokens.

If loading pushes the box into memory pressure, eugr's current recipe adds
`PYTORCH_ALLOC_CONF: "per_process_memory_fraction:0.8"` to `env:`. It is not set here because that fix
postdates the pinned image and has not been checked against it.

## Performance

Measured over the LAN against an idle server: unique prompts, streaming, rates from first to last token.

**Single stream**, median of 3 runs of 1,024 output tokens:

| tok/s | MTP 4 | MTP 3 (default) | Change |
|---|---|---|---|
| Prose, thinking off | 31.5 | **35.4** | +12% |
| Prose, temperature 0 | 34.3 | **37.2** | +8% |
| Code, thinking off | **57.5** | 54.3 | -6% |
| Code, temperature 0 | **63.4** | 58.9 | -7% |
| Code, thinking on | 56.4 | 57.4 | within noise |

Sampling is the shipped default (temperature 1.0) unless stated. MTP 3 and 4 ranges do not overlap except
in the last row.

**Draft acceptance by position** (temperature 1.0, thinking off):

| | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Prose, MTP 4 | 59% | 28% | 11% | 4% |
| Prose, MTP 3 | 59% | 29% | 12% | - |
| Code, MTP 4 | 88% | 75% | 61% | 47% |
| Code, MTP 3 | 86% | 74% | 59% | - |

**Why 3.** Each step is 10-13% faster, and on prose the fourth token was hardly ever accepted. Code still
lands it 47% of the time, so for single-stream code with thinking off go back to 4:
`-o speculative_config='{"method":"mtp","num_speculative_tokens":4}'`.

**Concurrent streams**, total tok/s, half code and half prose, 768 tokens each:

| Streams | MTP 4 (one run) | MTP 3 (mean of two) |
|---|---|---|
| 2 | 52.5 | 56.0 |
| 4 | 72.8 | 84.2 |
| 8 | 105.8 | 112.8 |

**Prefill** (MTP 4, prefix cache not hit; 8K was the same at MTP 3):

| Prompt | Time to first token | tok/s |
|---|---|---|
| 2K | 1.2 s | 1,770 |
| 8K | 3.9 s | 2,074 |
| 32K | 15.5 s | 2,116 |
| 131K | 65 s | 2,031 |

Decode does not slow with context: prose after a 32K and 131K prompt ran at 33.0 and 30.7 tok/s with MTP 4,
32.5 and 34.3 with MTP 3 (one run each, 768 tokens).

**Reported by others**, on different workloads, so only a rough comparison:

| Source | Setup | Result |
|---|---|---|
| [NVIDIA](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html) | this image and engine flags, MTP 4, SPEED-Bench coding, 1 Spark | 37.8 tok/s decode |
| [devopsfortunato](https://github.com/devopsfortunato/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark) | image derived from the same digest with a deeper PLE read queue, MTP 3, thinking off | 41.9 tok/s prose, 53.8 code, ~2,100 tok/s prefill; 119 tok/s total at 8 streams |

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

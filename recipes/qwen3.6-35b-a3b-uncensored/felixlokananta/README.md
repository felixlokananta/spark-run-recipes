# Qwen3.6-35B-A3B-Uncensored (HauhauCS "Aggressive") on DGX Spark

[HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive) —
an abliterated variant of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
Apache 2.0, 35B total / 3B active MoE, 262,144-token native context. The publisher states no
dataset or capability changes were made — only refusal behaviour was removed.

## Why llama.cpp and not vLLM

This repo ships **GGUF only** — twelve quant files plus an `mmproj`, with no safetensors and
no `config.json`. vLLM has nothing to load, so these recipes use `runtime: llama-cpp` and the
`repo:quant` colon syntax so sparkrun downloads just the one quant file it needs.

## Which recipe

| Recipe | Quant | File size | `sparkrun show` |
|---|---|---|---|
| `…-aggressive-q4kp-llama-cpp-felixlokananta` | `Q4_K_P` | 23.4 GB | 18.3 GB est. — fit: YES |
| `…-aggressive-q6kp-llama-cpp-felixlokananta` | `Q6_K_P` | 30.6 GB | fit: YES |

Q4_K_P is the publisher's recommended balanced default; Q6_K_P is their "higher fidelity if
resources permit" pick. Both fit a 128 GB Spark with room to spare — on this hardware the
binding constraint is memory bandwidth, not capacity, and since only ~3B params are active per
token even `Q8_K_P` (43.6 GB) is a reasonable option if you want it. The full ladder in the repo
runs `IQ2_M` (11.7 GB) through `Q8_K_P` (43.6 GB); swap the string after the colon in `model:`.

```bash
sparkrun recipe validate recipes/qwen3.6-35b-a3b-uncensored/felixlokananta/qwen3.6-35b-a3b-uncensored-aggressive-q4kp-llama-cpp-felixlokananta.yaml
sparkrun run recipes/qwen3.6-35b-a3b-uncensored/felixlokananta/qwen3.6-35b-a3b-uncensored-aggressive-q4kp-llama-cpp-felixlokananta.yaml --solo
```

## Why the flags look the way they do

**`--jinja`** — required. The GGUF carries Qwen's chat template with its thinking blocks and
tool-call format; without `--jinja` llama-server falls back to a generic template and the
`<think>` handling breaks. It's in the publisher's own launch line.

**`--ctx-size 131072`** — the card warns to keep context at **128K or above** or the model
loses its thinking behaviour, so this is a floor rather than a tuning knob. Raising it is cheap:
the architecture is a Qwen3.5-MoE hybrid where only 10 of the 40 decoder layers are full
attention (2 KV heads, 256 head dim), so KV runs ~20 KB/token at f16 — about 2.7 GB at 131072
and 5.4 GB at the native 262144 max. Bump `ctx_size` to `262144` if you want the full window.

**Sampling defaults `temp 1.0 / top_p 0.95 / top_k 20`** — the card's general/thinking-mode
recommendation. For coding it recommends `temp 0.6`; override per-run with
`-o temp=0.6` rather than editing the recipe.

**`--flash-attn on`, `--n-gpu-layers 99`** — full GPU offload, per the sparkrun llama.cpp
runtime docs.

## Vision

The repo includes `mmproj-…-f16.gguf` (0.9 GB), so the model is multimodal-capable under
llama.cpp. These recipes don't wire it in explicitly — recent llama.cpp auto-resolves an
`mmproj` alongside `-hf` when the repo has one, and sparkrun's colon syntax filters the
download to the matching quant. **I haven't verified which way that lands inside the sparkrun
container**, so if you need image input, check the server log for an `mmproj` line on startup
and if it's absent, add an explicit `--mmproj` path. Add `--no-mmproj` if you want to be sure
it stays text-only.

## Notes

- `host: 0.0.0.0` binds the OpenAI-compatible endpoint to every interface on the Spark. This
  model has had its refusal behaviour removed, so if the Spark is on a shared network, consider
  setting `-o host=127.0.0.1` and tunnelling instead.
- KV cache quantization (`--cache-type-k q8_0 --cache-type-v q8_0`) would halve KV memory, but
  KV is only a couple of GB here so it isn't worth the quality risk. Left off deliberately.

## Sources

- Model card: <https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive>
- llama.cpp runtime: <https://sparkrun.dev/runtimes/llama-cpp/>
- Recipe format: <https://sparkrun.dev/recipes/format/>

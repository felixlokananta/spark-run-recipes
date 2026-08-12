# spark-run-recipes

Personal [sparkrun](https://github.com/spark-arena/sparkrun) recipe registry for NVIDIA DGX Spark.

## Layout

```
.sparkrun/registry.yaml            # registry manifest (name: felix)
recipes/{model}/{user}/{model}-{quant}-{runtime}-{user}.yaml
```

## Recipes

- [`kat-coder-v2.5-dev`](recipes/kat-coder-v2.5-dev/felixlokananta/) — Kwaipilot KAT-Coder-V2.5-Dev,
  35B/3B-active agentic coding MoE, in NVFP4 / FP8 / BF16. vLLM.
- [`qwen3.6-35b-a3b-uncensored`](recipes/qwen3.6-35b-a3b-uncensored/felixlokananta/) —
  HauhauCS abliterated Qwen3.6-35B-A3B, GGUF Q4_K_P / Q6_K_P. llama.cpp.

## Use

Run a recipe straight from its path:

```bash
sparkrun recipe validate recipes/kat-coder-v2.5-dev/felixlokananta/kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta.yaml
sparkrun run recipes/kat-coder-v2.5-dev/felixlokananta/kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta.yaml --solo
```

Or register this directory once and refer to recipes by name:

```bash
sparkrun registry add <url-or-path-of-this-repo>
sparkrun list @felix
sparkrun show @felix/kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta
sparkrun run  @felix/kat-coder-v2.5-dev-nvfp4-vllm-felixlokananta
```

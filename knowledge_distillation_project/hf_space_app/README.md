---
title: Ignatiuz Student Chat And Benchmark
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
---

# Ignatiuz Student Chat And Benchmark

This Space exposes:

- A chat UI for your fine-tuned student model
- A benchmark tab comparing `fp16`, `int8`, and `int4_gguf`
- Metrics for memory usage, latency per token, and ROUGE

## Required Environment Variables

- `HF_BASE_MODEL_ID`: base model, for example `Qwen/Qwen2.5-3B-Instruct`
- `HF_ADAPTER_REPO_ID`: LoRA adapter repo or local adapter folder
- `HF_GGUF_MODEL_FILE`: local GGUF file path or downloaded GGUF filename

Optional:

- `HF_SYSTEM_PROMPT`
- `HF_MAX_NEW_TOKENS`
- `HF_TEMPERATURE`

## Notes

- `fp16` and `int8` use Transformers + PEFT
- `int4_gguf` uses `llama-cpp-python`
- On CPU Spaces, latency will be much slower

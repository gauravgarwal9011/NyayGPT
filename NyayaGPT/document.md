# NyayaGPT — Session Context Summary

**Date range:** 2026-04-17 → 2026-04-23
**Project root:** `/home/ubuntu/Fine-tuning/NyayaGPT/`
**Goal:** 7-day Indian legal fine-tune of Mistral 7B with QLoRA → evaluation → quantization → A/B dashboard → HF deployment.

---

## 1. Hardware & Environment

| Resource | Value | Notes |
|---|---|---|
| GPU | NVIDIA RTX 5090 (Blackwell, sm_120) | 32 GB VRAM |
| OS | Ubuntu via WSL2 | Kernel 6.6.87.2-microsoft-standard-WSL2 |
| Python | 3.10 (venv at `/home/ubuntu/Fine-tuning/.venv/`) | Venv IS on native ext4 — earlier `/mnt/e/Fine-tuning/.venv/` path in tracebacks was stale/misleading |
| Torch | 2.10.0+cu128 | Blackwell support |
| Unsloth | 2026.3.8 | |
| Transformers | 4.56.2 | |
| llama-cpp-python | 0.3.17 | Rebuilt from source with `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120"` for Blackwell |

### Disk layout (critical)

| Mount | Filesystem | Free | Role |
|---|---|---|---|
| `/` (`/dev/sdf`) | Native ext4 (WSL VHD) | 836 GB | Project code, venv, HF cache |
| `/mnt/e` | Windows E: | **7.6 GB** | Hosts the WSL VHD file (`/mnt/e/WSL/`, 125 GB) — VHD growth limited by this free space |
| `/mnt/f` | Windows F: | 741 GB | Scratch for GGUF intermediates |
| Swap | 11 GB (4 GB original + 8 GB new swapfile) | — | Expanded after OOM during Day 4 merge |

**Key insight:** The "836 GB free" inside Linux is misleading — the ext4 VHD file dynamically grows on `/mnt/e`, which only has 7.6 GB of slack. Writing >7.6 GB of new data on `/home/ubuntu/` can crash WSL.

---

## 2. Day-by-Day Progress

### Day 1 — Dataset collection (COMPLETE)

- Scraped Indian Kanoon + synthetic generation via Azure GPT-4o + gpt-oss-20B GGUF teacher blend.
- Final output: **1,690 Q&A pairs** → `output/train.jsonl` (1,521) + `output/eval.jsonl` (169).
- **Fixes applied during:**
  - Azure content-filter silent NoneType crash → added explicit None handling in `synthetic_generator.py`.
  - gpt-oss-20B GGUF prompt format corrected from `<|im_start|>` to `<|start|>...<|message|>...<|end|>`.
  - System prompt softened (removed "Insufficient excerpt" instruction that caused model to refuse valid excerpts).
  - C++ stderr suppression during llama-cpp load to prevent Jupyter IO deadlock.

### Day 2 — QLoRA training (COMPLETE)

- Two runs preserved: **`adapters-2ep/`** and **`adapters-3ep/`**.
- Config: `r=16, alpha=32, lr=2e-4, batch=2, grad_accum=4, bf16=True, seq=2048`.
- Adapter weights stored as bf16 (verified via safetensors inspection — all 448 LoRA params in `torch.bfloat16`).
- MLflow experiment: `nyayagpt-training`.

### Day 3 — Evaluation (COMPLETE, winner: 2ep)

- ROUGE (local) + RAGAS (Azure gpt-4o-mini judge + text-embedding-3-small embeddings).

| Metric | Base | NyayaGPT-2ep | NyayaGPT-3ep |
|---|---|---|---|
| ROUGE-L | — | **0.4118** | 0.4028 |
| RAGAS Faithfulness | — | **0.7213** | 0.6573 |

- **2ep declared winner.** Symlink `adapters/` → `adapters-2ep/` created to make `config.ADAPTER_DIR` resolve correctly for downstream modules.
- RAGAS 0.4.x quirks handled:
  - Per-sample list returns wrapped with `_mean()` helper.
  - Broken `__contains__` on `EvaluationResult` bypassed with try/except `__getitem__`.

### Day 4 — Quantization benchmark (IN PROGRESS, partially complete)

See Section 3 below for the full story — this was the hardest day.

---

## 3. Day 4 Deep Dive — GGUF Build + Benchmark

### 3.1 GGUF build (COMPLETE)

**Goal:** Produce `nyayagpt-q4km.gguf` (~4 GB) from the 2ep adapter.

**Pipeline (`scripts/build_gguf.py`):**
1. Load base + adapter via Unsloth — pass adapter dir as `model_name` so Unsloth auto-wraps as PeftModel (earlier `model.load_adapter()` silently failed to merge).
2. `save_pretrained_merged(..., "merged_16bit")` → 14 GB FP16 weights.
3. `convert_hf_to_gguf.py` → 14 GB FP16 GGUF.
4. `llama-quantize` → 4 GB Q4_K_M.

**Crises along the way:**

| # | Problem | Resolution |
|---|---|---|
| 1 | OOM killed first attempt (14 GB download + 900 MB free RAM) | Expanded swap from 4 GB → 12 GB via `dd`-based swapfile on `/swapfile` |
| 2 | `sudo swapoff -a` got OOM-killed itself | Abandoned; added 8 GB swap file instead of replacing |
| 3 | `model.load_adapter()` silently skipped merge ("not a PeftModel") | Passed adapter dir as `model_name` to Unsloth |
| 4 | Unsloth's `save_pretrained_gguf()` tried to `sudo apt update` to install llama.cpp, misread sudo failure as "no internet" | Pre-built llama.cpp manually at `~/.unsloth/llama.cpp/` (CPU-only, `-DGGML_CUDA=OFF`, only `llama-quantize` target) |
| 5 | `pip install -r requirements-convert_hf_to_gguf.txt` catastrophically downgraded torch 2.10+cu128 → torch 2.6+cpu AND upgraded transformers 4.56 → 5.5 | Rolled back with explicit pinned reinstalls. Lesson: **never install convert script's full requirements; only install `gguf` module** |
| 6 | Merged FP16 + FP16 GGUF writes (~28 GB) would grow WSL VHD, overflowing 7.6 GB free on `/mnt/e` and crashing WSL (happened in a prior session) | Redirected all scratch outputs to `/mnt/f/NyayaGPT-scratch/`. Final Q4_K_M GGUF symlinked into `adapters-2ep/nyayagpt-q4km.gguf` so notebooks discover it |

**Final artifacts:**
```
/mnt/f/NyayaGPT-scratch/
  merged_model/            14 GB (reclaimable)
  nyayagpt-fp16.gguf       14 GB (reclaimable — also useful for FP16 inference)
  nyayagpt-q4km.gguf       4 GB  (final)

adapters-2ep/nyayagpt-q4km.gguf -> /mnt/f/NyayaGPT-scratch/nyayagpt-q4km.gguf
adapters -> adapters-2ep  (symlink, winning adapter)
```

### 3.2 Quantization benchmark — CURRENT BLOCKER

**Notebook:** `notebooks/04_quantization_benchmark.ipynb`
**Module:** `src/nyaya_pipeline/benchmark.py`

**Target table:**

| Variant | Method | Status |
|---|---|---|
| FP16 | HF Transformers bf16 + PEFT | **FAILING** (Blackwell cuBLAS bug) |
| INT8 | bitsandbytes 8-bit + PEFT | **FAILING** (Blackwell cuBLAS bug + Unsloth class pollution) |
| INT4 GGUF | llama-cpp-python Q4_K_M | **WORKING** (memory 4.37 GB, latency 5.0 ms/tok, ROUGE-L 0.3706) |

**Blocker diagnosis — two separate Blackwell/CUDA 12.8 issues:**

1. **cuBLAS `CUBLAS_STATUS_INVALID_VALUE` on 16-bit gemm.**
   Fails for both `CUDA_R_16F` (fp16) AND `CUDA_R_16BF` (bf16) GEMM operations during decoding. Reproducible with:
   - Raw `AutoModelForCausalLM` + `PeftModel.from_pretrained` (fp16)
   - Same stack with `torch_dtype=torch.bfloat16` (bf16)
   - Unsloth's `FastLanguageModel.from_pretrained(adapter_dir, load_in_4bit=False)` (also bf16)

   All three loader paths hit the same `cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)` crash during `.generate()`. The bug is in CUDA 12.8's cuBLAS tensor-op kernels for sm_120 with certain Mistral gemm shapes.

2. **Unsloth class pollution.**
   Once `FastLanguageModel.from_pretrained` runs, it patches `MistralAttention` globally (adds `apply_qkv`). Subsequent `AutoModelForCausalLM` loads pick up the patched class. If INT8 then tries `attn_implementation="eager"`, the eager attention path calls forward on the patched class expecting stock attributes → `AttributeError: 'MistralAttention' object has no attribute 'apply_qkv'`.

**Attempts made (all failed):**
- fp16 → bf16 dtype switch
- `bnb_8bit_compute_dtype=torch.bfloat16` for INT8
- `attn_implementation="eager"` for INT8
- Unsloth `FastLanguageModel` loader for FP16

**What does work:**
- llama-cpp-python GGUF inference (uses its own CUDA kernels, not cuBLAS GEMM) — fully verified end-to-end.

### 3.3 Subprocess isolation (FIX APPLIED)

The GGUF benchmark was ALSO crashing the Jupyter kernel initially — not from the cuBLAS bug, but from a different problem: **torch's CUDA caching allocator and llama-cpp-python's direct CUDA calls conflict** in the same process. Running FP16+INT8 first leaves torch holding GPU context, then llama.cpp's `Llama()` constructor segfaults.

**Fix:** created `scripts/_bench_gguf_worker.py` — a standalone script that does the full GGUF benchmark in a fresh Python process. `bench_int4_gguf` in `benchmark.py` now invokes it via `subprocess.run`, reading results from a JSON temp file. This gives llama.cpp a clean CUDA context with no torch pollution.

**Result:** INT4 GGUF benchmark runs reliably every time.

---

## 4. Proposed Path Forward (awaiting user decision)

Given the persistent Blackwell cuBLAS bug affects any 16-bit gemm through PyTorch, the pragmatic pivot is:

### All-GGUF benchmark

| Variant | Engine | File | Status |
|---|---|---|---|
| "FP16" | llama.cpp + GGUF F16 | `/mnt/f/NyayaGPT-scratch/nyayagpt-fp16.gguf` (14 GB) | Already built |
| "INT8" | llama.cpp + GGUF Q8_0 | Needs quantization from FP16 GGUF (~5 min) | Not built |
| "INT4" | llama.cpp + GGUF Q4_K_M | `adapters/nyayagpt-q4km.gguf` (4 GB) | Already built, working |

**Refactor required:**
- Rewrite `bench_fp16` and `bench_int8` in `benchmark.py` to call the existing subprocess worker with different GGUF paths.
- Update `04_quantization_benchmark.ipynb` variant table to reflect "all via llama.cpp".
- Run `llama-quantize` once to produce Q8_0 GGUF.

**Trade-off:** Plan originally called for HF+bnb path for FP16/INT8. Pivoting to GGUF loses the "framework diversity" comparison but is the only path that works on this hardware, AND gives a consistent apples-to-apples quantization comparison (same inference engine).

### Alternative (not recommended)

Keep fighting cuBLAS: try `TORCH_CUBLAS_PREFER_CUBLASLT=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disable tensor-op matmul flags. May or may not fix the bug, likely hours of iteration.

---

## 5. File Inventory

### Source code
```
src/nyaya_pipeline/
  config.py               — centralized config, env-var overridable
  logger.py               — logger setup (reused from KD project)
  exceptions.py           — domain errors
  data_collector.py       — Indian Kanoon scraper
  synthetic_generator.py  — Azure GPT-4o + gpt-oss-20B blend
  quality_filter.py       — length + keyword-overlap filters
  trainer.py              — QLoRA + MLflow
  evaluator.py            — ROUGE + RAGAS (Azure judge)
  benchmark.py            — FP16/INT8/INT4 benchmark (INT4 working, others blocked)
  infer.py                — inference + A/B routing
  dashboard.py            — Streamlit A/B UI
```

### Scripts
```
scripts/
  build_gguf.py           — Unsloth merge → FP16 GGUF → Q4_K_M (WORKING)
  _bench_gguf_worker.py   — isolated GGUF benchmark subprocess (WORKING)
```

### Notebooks
```
notebooks/
  01_dataset_collection.ipynb       — Day 1 (complete)
  02_qlora_training.ipynb           — Day 2 (complete)
  03_evaluation.ipynb               — Day 3 (complete)
  04_quantization_benchmark.ipynb   — Day 4 (blocked on FP16/INT8)
  05_ab_test_dashboard.ipynb        — Day 5 (not started)
  06_hf_hub_deployment.ipynb        — Day 6 (not started)
  07_hf_spaces_demo.ipynb           — Day 7 (not started)
```

### Adapters & artifacts
```
adapters-2ep/                        — winning adapter (2 epochs)
  adapter_config.json
  adapter_model.safetensors          83 MB (bf16, 448 LoRA params)
  nyayagpt-q4km.gguf -> /mnt/f/...   4 GB symlink

adapters-3ep/                        — runner-up adapter (3 epochs)

adapters -> adapters-2ep             — winner symlink

/mnt/f/NyayaGPT-scratch/
  merged_model/                      14 GB (FP16 merged HF weights)
  nyayagpt-fp16.gguf                 14 GB
  nyayagpt-q4km.gguf                 4 GB

~/.unsloth/llama.cpp/
  build/bin/llama-quantize           quantization tool
  convert_hf_to_gguf.py              HF → GGUF converter
```

### Dependencies
```
/home/ubuntu/Fine-tuning/.venv/        — main venv (on native ext4)
  torch 2.10.0+cu128
  transformers 4.56.2
  unsloth 2026.3.8
  peft 0.18.1
  llama-cpp-python 0.3.17 (custom CUDA build, Blackwell sm_120)
  gguf (for convert script)
  ragas 0.1.x
  rouge-score
  mlflow, streamlit, fastapi, gradio (later days)
```

---

## 6. Key Lessons Learned (for future sessions)

1. **NEVER run `pip install -r` from a third-party requirements file** — will trample versioned deps (torch, transformers). Install individual packages you need (e.g. `pip install gguf`) with exact pins.
2. **WSL disk space has two levels** — the ext4 VHD reports internal free space, but the VHD file itself lives on a Windows mount with its own free space. Heavy writes inside Linux can overflow the Windows filesystem and crash WSL.
3. **`model.load_adapter()` in Unsloth ≠ PeftModel wrapping.** Pass adapter dir as `model_name` to `FastLanguageModel.from_pretrained` for proper merge semantics.
4. **Unsloth patches are sticky across loads** — once `FastLanguageModel.from_pretrained` runs in a kernel, `MistralAttention` is globally patched. Subsequent raw `AutoModelForCausalLM` loads of Mistral see the patched class.
5. **torch + llama-cpp-python don't share a process well** on WSL2 Blackwell — subprocess isolation is mandatory for mixed-engine benchmarks.
6. **Blackwell cuBLAS has a 16-bit gemm bug** in CUDA 12.8 affecting both fp16 and bf16. GGUF via llama.cpp bypasses it entirely.
7. **`sudo swapoff -a` can OOM-kill itself** if insufficient free RAM to absorb the swap contents. Always add a new swap file instead of replacing existing swap.

---

## 7. Current Session State (2026-04-23)

- Day 4 benchmark: INT4 GGUF working (4.37 GB, 5.0 ms/tok, ROUGE-L 0.3706). FP16 + INT8 blocked on Blackwell cuBLAS bug.
- Proposal on the table: pivot to all-GGUF benchmark (quantize existing FP16 GGUF → Q8_0, refactor `bench_fp16`/`bench_int8` to reuse the subprocess worker).
- Awaiting user approval to execute pivot.
- Swap: 11 GB (healthy).
- `/mnt/e`: 7.6 GB free (watch for VHD growth).
- `/mnt/f`: 709 GB free (plenty of headroom for Q8_0 quantization).

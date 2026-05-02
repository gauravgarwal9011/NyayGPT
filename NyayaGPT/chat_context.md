# NyayaGPT — Session Context Document

**Date:** 2026-04-25
**Project root:** `/home/ubuntu/Fine-tuning/NyayaGPT/`
**User:** Gaurav Garwal (gauravgarwal9011@gmail.com)
**GitHub:** `https://github.com/gauravgarwal9011/NyayGPT` (note: typo in repo name — "NyayGPT" not "NyayaGPT")

---

## TL;DR

NyayaGPT is a 7-day Indian legal LLM project that fine-tunes Mistral-7B-Instruct-v0.3 on 1,690 Q&A pairs using QLoRA, then benchmarks quantization (FP16/INT8/INT4) and ships an MLflow + Streamlit MLOps stack. Days 1–5 are complete. Day 6 (HF Hub deployment) is in progress with a leaked-token cleanup pending. Day 7 (HF Spaces demo) is pending. A LinkedIn post and one-page LaTeX resume have been drafted for project promotion.

---

## Hardware & Environment

- **Platform:** WSL2 Ubuntu 22.04 on Windows host
- **GPU:** NVIDIA RTX 5090 (32 GB, Blackwell **sm_120**)
- **CUDA:** 12.8
- **Python:** 3.10 venv at `/home/ubuntu/Fine-tuning/.venv/`
- **Key packages:** torch 2.10.0+cu128, transformers 4.56.2, bitsandbytes 0.49.2, llama-cpp-python (GPU build)
- **Storage layout:**
  - Project + WSL VHD: `/home/ubuntu/Fine-tuning/` (limited free space — `/mnt/e` is the VHD location)
  - GGUF scratch: `/mnt/f/NyayaGPT-scratch/` (~700 GB free, all model intermediates live here)

---

## Project Plan & Progress

| Day | Notebook | Status |
|-----|----------|--------|
| 1 | `01_dataset_collection.ipynb` | ✅ DONE — 1,690 Q&A pairs (1,521 train / 169 eval) from IndianKanoon + Azure OpenAI GPT-4o synthetic |
| 2 | `02_qlora_training.ipynb` | ✅ DONE — QLoRA, LoRA r=16/α=32, 2 epochs, RTX 5090 (`adapters-2ep` won over `adapters-3ep`) |
| 3 | `03_evaluation.ipynb` | ✅ DONE — eval results in MLflow `nyayagpt-evaluation`: ROUGE-1 0.5683, ROUGE-L 0.4028, RAGAS Faithfulness 0.6573 |
| 4 | `04_quantization_benchmark.ipynb` | ✅ DONE — see results below |
| 5 | `05_ab_test_dashboard.ipynb` | ✅ DONE — pivoted to **vanilla Mistral Q4 vs NyayaGPT Q4** for apples-to-apples comparison |
| 6 | `06_hf_hub_deployment.ipynb` | ⚠️ IN PROGRESS — token leak needs cleanup before push |
| 7 | `07_hf_spaces_demo.ipynb` | ⬜ PENDING |

---

## Day 4 — Quantization Benchmark Results

`output/benchmark_results.json`:

| Variant | Memory (GB) | Latency (ms/tok) | ROUGE-L |
|---------|-------------|------------------|---------|
| FP16 | 14.50 | 10.8 | 0.3668 |
| INT8 | 7.70 | 6.8 | 0.3711 |
| INT4 (GGUF) | 4.37 | 4.9 | 0.3706 |

**Headline:** INT4 = 3.3× memory reduction + 2.2× faster inference + zero ROUGE-L degradation vs FP16.

**Note:** "Memory" is GGUF file size on disk (used as VRAM proxy because nvidia-smi unavailable in WSL). All three variants run via llama.cpp (not transformers) — see Critical Findings below for why.

---

## Day 5 — The "Real Base vs Fine-tuned" Pivot (deep dive)

**Original Day 5:** compared FP16 NyayaGPT vs INT4 NyayaGPT — really a quantization comparison, not a fine-tuning comparison. Useless for LinkedIn narrative.

**Goal:** compare **vanilla Mistral-7B-Instruct-v0.3** (no fine-tuning) vs **NyayaGPT** (fine-tuned).

**Attempt 1: HF-native Mistral via transformers** — failed.
- `notebooks/99_hf_base_inference_test.ipynb` was created to test 4 approaches in a fresh kernel:
  1. BnB 4-bit nf4 + bf16 compute → **FAILED** (CUBLAS_STATUS_INVALID_VALUE on `cublasGemmEx` with CUDA_R_16BF)
  2. FP32 full precision → **FAILED** (CUBLAS_STATUS_INVALID_VALUE on `cublasSgemmStridedBatched` — different kernel, same outcome)
  3. BnB 4-bit + eager attention → **FAILED** (same cuBLAS error)
  4. FP16 default (negative control) → **FAILED** as expected

  **Conclusion:** the cuBLAS 12.8 bug on Blackwell sm_120 affects **every gemm code path** (FP16, BF16, FP32, mixed-precision through bitsandbytes). It's not a dtype problem — the entire cuBLAS library is broken on this hardware. Anything that goes through PyTorch's `nn.Linear` → cuBLAS dies.

**Attempt 2: GGUF base Mistral via llama.cpp** — succeeded.
- Created `scripts/build_base_mistral_gguf.py` to convert the HF-cached `mistralai/Mistral-7B-Instruct-v0.3` safetensors → FP16 GGUF → Q4_K_M GGUF using llama.cpp tools at `~/.unsloth/llama.cpp/`.
- Output: `/mnt/f/NyayaGPT-scratch/mistral-base-q4km.gguf` (~4.4 GB)
- No new download — reused existing HF cache.

**Wired into Day 5:**
- `src/nyaya_pipeline/infer.py` — added `MISTRAL_BASE_GGUF` constant; `ab_generate` now compares base Mistral Q4 vs NyayaGPT Q4
- `src/nyaya_pipeline/dashboard.py` — column labels updated to "Base Mistral (vanilla)" / "NyayaGPT (fine-tuned)"
- `notebooks/05_ab_test_dashboard.ipynb` — cells 0, 2, 9 updated (architecture diagram, prereq checks, UI preview)
- `notebooks/99_base_vs_finetuned_comparison.ipynb` — separate ipynb-only A/B with 5 legal questions + MLflow logging

**MLflow A/B run params:**
- `base_engine: "gguf-q4_k_m-vanilla-mistral"`
- `finetuned_engine: "gguf-q4_k_m-nyaya"`
- `quantization: "Q4_K_M (same for both)"`

This isolates fine-tuning as the only variable.

---

## Critical Engineering Findings

### Finding 1: Blackwell cuBLAS is fully broken on CUDA 12.8

- Affects: FP16, BF16, FP32 — both `cublasGemmEx` (16-bit) and `cublasSgemmStridedBatched` (32-bit)
- Affects: bitsandbytes 4-bit (still routes some ops through cuBLAS)
- Workaround: **only llama.cpp's custom CUDA kernels work** (GGUF format)
- Implication: every variant in `04_quantization_benchmark.ipynb` runs through llama.cpp via `_bench_gguf_worker.py` subprocess (torch + llama-cpp-python can't share CUDA context safely on WSL2)

### Finding 2: WSL2 ↔ Windows networking quirks

- Services bound to `127.0.0.1` (loopback only) sometimes unreachable from Windows browser via `localhost:PORT` after WSL restart.
- **Fix:** bind to `0.0.0.0` instead.
  - MLflow: `python -m mlflow ui --backend-store-uri ... --host 0.0.0.0 --port 5000`
  - Streamlit: `python -m streamlit run ... --server.address 0.0.0.0 --server.port 8501 --server.headless true`
- WSL eth0 IP is `172.28.232.101` — direct IP access works as fallback.
- For SSH from another computer: use `ssh -L 5000:localhost:5000 -L 8501:localhost:8501 user@host` to tunnel both ports.

### Finding 3: WSL VHD overflow trap

- `/mnt/e` (Windows side where WSL ext4.vhdx lives) had only 7.6 GB free — not enough for the 28 GB of GGUF intermediates.
- **Fix:** all GGUFs and merged FP16 weights live in `/mnt/f/NyayaGPT-scratch/` (Windows F: drive, ~700 GB free); only symlinks live in the project tree.

### Finding 4: PATH issues even with venv "activated"

- `(.venv)` prompt label appears, but `python`, `streamlit`, `huggingface-cli` all fall through to system Python 2 / not found.
- **Fix:** always use full path `/home/ubuntu/Fine-tuning/.venv/bin/python` (or `python -m streamlit`, `python -m mlflow ui`, etc.)

---

## Files Created/Modified This Session

### Created
- `scripts/build_base_mistral_gguf.py` — converts HF-cached Mistral safetensors → Q4_K_M GGUF
- `notebooks/99_hf_base_inference_test.ipynb` — diagnostic for HF-native inference attempts (all 4 failed)
- `notebooks/99_base_vs_finetuned_comparison.ipynb` — text-only A/B comparison notebook (5 legal Q&As + MLflow logging)
- `chat_context.md` — this document

### Modified
- `src/nyaya_pipeline/infer.py` — added `MISTRAL_BASE_GGUF`, rewrote `ab_generate` for base-vs-finetuned
- `src/nyaya_pipeline/dashboard.py` — updated column labels and stats panel
- `notebooks/05_ab_test_dashboard.ipynb` — cells 0, 2, 9 updated
- `notebooks/06_hf_hub_deployment.ipynb` — cell 4 (model card auto-pulls live numbers), cell 5 (uncomment + ignore_patterns), cell 6 (rewritten merged-model push). **Cell 3 was added by user with hardcoded HF token — needs removal.**

### Filesystem additions (in `/mnt/f/NyayaGPT-scratch/`)
- `mistral-base-fp16.gguf` (~14 GB, intermediate, can be deleted)
- `mistral-base-q4km.gguf` (~4.4 GB, **keep** — required for Day 5)

---

## Outstanding Issues

### URGENT: Leaked HF token

- Cell-3 of `06_hf_hub_deployment.ipynb` contains `login(token="")` — hardcoded.
- Token was pushed to GitHub on commit `275df9e65c907cf7bc79d7424561be510f75b3ef` — push was rejected by GitHub Push Protection but the commit existed locally and the token is in GitHub's block-message logs.
- User has run `git reset HEAD~1` to undo the local commit (changes still in working tree).
- **Required actions:**
  1. **Revoke the token at https://huggingface.co/settings/tokens** (it's compromised — was visible in chat transcript and GitHub error message).
  2. **Delete cell-3 from the notebook** (use NotebookEdit with `edit_mode=delete`, `cell_id=cell-3`).
  3. Re-stage, re-commit, re-push.
- Auth alternative going forward: use `os.getenv('HF_TOKEN')` in cell-2 (already supported), set token via `export HF_TOKEN=hf_xxxxx` before launching Jupyter.

### Token has not yet been deleted

User declined the proposed `NotebookEdit delete cell-3` operation. Reason unclear — possibly wanted to do it manually. Verify cell-3 is gone before next push attempt.

---

## Promotional Materials Drafted

### LinkedIn post prompt
- Full LLM-feedable prompt with all real metrics baked in (1,690 pairs, ROUGE/RAGAS scores, quantization table, FP16→INT4 deltas).
- Asks for 3 hooks, full body, suggested first comment, hashtag options.
- Includes carousel slide designs (4 slides: Streamlit A/B screenshot, MLflow quality bars, latency chart, full quantization 3-panel).

### Resume — single-page LaTeX
- Charter font (`\usepackage{charter}`)
- 0.45in margins, parskip 1pt, itemsep 1pt, topsep 0pt
- Maruti Suzuki internship dropped (saved space)
- Best 2 projects (NyayaGPT, Due Diligence Agent) embedded **under** Ignatiuz role
- Other 3 projects (English Speaking, Multimodal RAG, Route Guidance) in separate Projects section
- New Key Skills line: **"LLM Fine-Tuning & MLOps: QLoRA, LoRA, PEFT, Unsloth, Quantization (FP16/INT8/INT4 GGUF), llama.cpp, MLflow, Hugging Face Hub Deployment, Model Evaluation (ROUGE, RAGAS)"**
- One Ignatiuz role bullet dropped (the RAG-focused one) to fit single page — RAG is already covered in the Due Diligence project bullet
- Compile via Overleaf recommended.

---

## Pending / Next Steps

In priority order:

1. **Revoke leaked HF token** (URGENT, security)
2. **Delete cell-3 from `06_hf_hub_deployment.ipynb`** (NotebookEdit, edit_mode=delete, cell_id=cell-3)
3. Set `HF_TOKEN` env var, re-stage notebook 06 + other modified files, commit, push to `https://github.com/gauravgarwal9011/NyayGPT`
4. (Optional) Rename GitHub repo from `NyayGPT` to `NyayaGPT` to fix typo — update resume URL accordingly
5. **Day 7:** build `07_hf_spaces_demo.ipynb` — Gradio app with INT4 GGUF for free-tier CPU Space
6. Generate LinkedIn post via the drafted prompt; post with carousel
7. Final-compile resume via Overleaf, verify single page, send

---

## Useful Commands (paste-ready)

```bash
# Launch MLflow on all interfaces (Windows browser can reach via localhost:5000)
nohup /home/ubuntu/Fine-tuning/.venv/bin/python -m mlflow ui \
    --backend-store-uri /home/ubuntu/Fine-tuning/NyayaGPT/mlruns \
    --host 0.0.0.0 --port 5000 > /tmp/mlflow.log 2>&1 &

# Launch Streamlit dashboard (base Mistral vs NyayaGPT, both Q4_K_M)
/home/ubuntu/Fine-tuning/.venv/bin/python -m streamlit run \
    src/nyaya_pipeline/dashboard.py \
    --server.address 0.0.0.0 --server.port 8501 --server.headless true

# Build base Mistral GGUF (one-time, ~8-10 min, no download)
/home/ubuntu/Fine-tuning/.venv/bin/python scripts/build_base_mistral_gguf.py

# SSH from another local computer with port forwarding for both dashboards
ssh -L 5000:localhost:5000 -L 8501:localhost:8501 ubuntu@<wsl-host-ip>
```

---

## Key Configuration Constants

In `src/nyaya_pipeline/config.py`:
- `STUDENT_MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"`
- `LORA_R = 16`, `LORA_ALPHA = 32`, `LORA_DROPOUT = 0.0`
- `NUM_TRAIN_EPOCHS = 2`
- `TRAIN_BATCH_SIZE = 2`, `GRAD_ACCUM_STEPS = 4` (effective batch = 8)
- `LEARNING_RATE = 2e-4`
- `MLFLOW_EXPERIMENT_TRAIN/EVAL/AB = "nyayagpt-training"/"-evaluation"/"-ab-test"`

In `src/nyaya_pipeline/infer.py`:
- `NYAYAGPT_FP16_GGUF = /mnt/f/NyayaGPT-scratch/nyayagpt-fp16.gguf`
- `NYAYAGPT_Q8_GGUF = /mnt/f/NyayaGPT-scratch/nyayagpt-q8_0.gguf`
- `NYAYAGPT_Q4_GGUF = adapters/nyayagpt-q4km.gguf` (symlink to `/mnt/f/...`)
- `MISTRAL_BASE_GGUF = /mnt/f/NyayaGPT-scratch/mistral-base-q4km.gguf` ← new

---

## End of context document

Hand this whole file to the next chat as context. Key facts the next session needs to know cold:
1. Blackwell cuBLAS is dead — only GGUF/llama.cpp works for inference
2. Day 5 compares **vanilla Mistral Q4_K_M vs NyayaGPT Q4_K_M** (apples-to-apples)
3. Cell-3 of notebook 06 has a leaked HF token that must be removed before any git push
4. All GGUFs live in `/mnt/f/NyayaGPT-scratch/`, not in the project tree
5. Always use `/home/ubuntu/Fine-tuning/.venv/bin/python` — never bare `python`

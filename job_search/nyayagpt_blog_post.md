---
title: "Blackwell killed cuBLAS — and accidentally taught me the right way to ship a domain LLM"
subtitle: "How a broken CUDA stack on the RTX 5090 forced NyayaGPT through llama.cpp end-to-end, and why the quantization benchmark is the most useful thing I built in 7 days."
tags: [llm, fine-tuning, qlora, quantization, gguf, llama-cpp, mistral, blackwell, indian-legal-tech]
---

## TL;DR

I gave myself 7 days to fine-tune **Mistral-7B-Instruct-v0.3** into **NyayaGPT**, a specialized Indian legal assistant — QLoRA on a 1,690-pair dataset, full evaluation, quantization benchmark, MLflow + Streamlit dashboard. Day 1 to Day 5 are done. Day 6 (HF Hub) and Day 7 (HF Spaces demo) are in progress.

The technical story I didn't expect to tell: my **RTX 5090 (Blackwell, sm_120) on CUDA 12.8 is unable to run a single transformer forward pass through PyTorch.** Every cuBLAS gemm path — FP16, BF16, FP32, bitsandbytes 4-bit — dies with `CUBLAS_STATUS_INVALID_VALUE`. The only inference stack that works on this hardware is **llama.cpp's custom CUDA kernels via GGUF**.

This wasn't a setback. It forced me into a discipline I should have adopted anyway: **the model gets quantized on day 4, and every comparison after that runs at the same quantization level.** Apples-to-apples. No transformers-to-llama.cpp ambiguity.

The headline numbers (RTX 5090, llama.cpp, same 169-item legal eval set):

| Variant | Size on disk | Latency (ms/tok) | ROUGE-L |
|---|---|---|---|
| FP16 GGUF | 14.50 GB | 10.8 | 0.367 |
| INT8 (Q8_0) | 7.70 GB | 6.8 | 0.371 |
| INT4 (Q4_K_M) | 4.37 GB | 4.9 | 0.371 |

**3.3× smaller, 2.2× faster, no measurable quality loss.** Plus separate full-eval results on the FP16 model: ROUGE-1 = 0.57, ROUGE-L = 0.40, RAGAS Faithfulness = 0.66.

Repo: [github.com/gauravgarwal9011/NyayGPT](https://github.com/gauravgarwal9011/NyayGPT)
Adapter: [huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter](https://huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter)

---

## Why Indian legal? Why not just use GPT-4?

Two reasons.

**Cost and sovereignty.** A typical Indian district lawyer does ~30 case-law lookups a day. At GPT-4-class API pricing that adds up fast — more than the median monthly software spend of small Indian law firms. A 4-bit quantized 7B model running on a ₹40,000 laptop, with no data leaving the office, is a structurally different product.

**Vocabulary drift.** Indian law speaks in IPC sections, CrPC, Constitutional Articles, Contract Act, IT Act, plus a lot of Hindi-English code-switching ("FIR file karna", "bail granted in CrPC 437"). Generic models hallucinate citations confidently — they'll make up an "IPC Section 412(a)" that doesn't exist. The fine-tune was less about teaching the model law and more about teaching it to refuse, cite, and hedge in the right places.

## The dataset (this is where most of the work was)

I scraped **1,690 instruction pairs**, split 1,521 train / 169 eval, focusing on:

- Section-wise Q&A across IPC, CrPC, Constitution, Contract Act, IT Act
- Case-summary tasks with binding-precedent extraction
- Bail / anticipatory bail reasoning chains
- Consumer-court and labor-law scenarios

Raw scrapes are useless for instruction tuning. The pipeline:

1. **Extract** judgment text + headnotes + cited sections from IndianKanoon.
2. **Synthesize** instruction-style Q/A using **Azure OpenAI GPT-4o** with a strict prompt forcing it to cite section numbers and hedge appropriately.
3. **Filter** with heuristics + manual review for the low-confidence pairs.
4. **Format** as Mistral instruction format (`<s>[INST] ... [/INST]`).

What I'd do differently: spend another two weeks pushing the dataset to ~5,000 pairs. Below ~3,000 examples for a 7B model targeting a niche domain, you're underfitting. The fine-tune is good, not great, and dataset size is the obvious bottleneck.

## The fine-tune: Unsloth + QLoRA

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
)

# Unsloth, BnB-4bit base, 2 epochs, batch=2, grad-accum=4 (effective 8), LR=2e-4
```

A few non-default decisions:

**1. r=16, not r=8.** The legal vocabulary shift is large enough that r=8 visibly underfit on the holdout — the model defaulted to generic "consult a lawyer" hedges instead of citing sections. Bumping to r=16 doubled adapter size to ~80MB and recovered several ROUGE-L points.

**2. All projection matrices, not just q/v.** Original LoRA found q/v sufficient, but for instruction-following with structured output (citations in a specific format), MLP layers (`gate/up/down_proj`) matter for free quality.

**3. 2 epochs, not 3.** I trained two checkpoints: `adapters-2ep` and `adapters-3ep`. The 3-epoch version had lower loss but worse RAGAS faithfulness — it was memorizing training citations and applying them to wrong contexts. 2 epochs won. Pick your eval metric carefully; loss is not it.

## The cuBLAS catastrophe

Here's the part I didn't plan for.

I run on a WSL2 Ubuntu 22.04 box with an **RTX 5090 (Blackwell, sm_120) on CUDA 12.8**. Day 5 of the project was supposed to be a base-vs-fine-tuned A/B comparison: vanilla Mistral-7B-Instruct-v0.3 vs NyayaGPT, side-by-side on legal questions.

Trivial, right? Load the base model with `transformers`, generate, log to MLflow.

Every approach died. I documented four attempts in a clean kernel:

1. **bitsandbytes 4-bit nf4 + bf16 compute** → `CUBLAS_STATUS_INVALID_VALUE` on `cublasGemmEx` with `CUDA_R_16BF`
2. **FP32 full precision** → `CUBLAS_STATUS_INVALID_VALUE` on `cublasSgemmStridedBatched` (different kernel, same outcome)
3. **bitsandbytes 4-bit + eager attention** → same cuBLAS error
4. **FP16 default (negative control)** → failed as expected

The pattern: this isn't a dtype problem. It isn't a bitsandbytes problem. **The entire cuBLAS library is broken on Blackwell sm_120 + CUDA 12.8.** Anything that goes through PyTorch's `nn.Linear` → cuBLAS dies.

My options:
- Wait for NVIDIA / PyTorch to fix it (no ETA at the time)
- Downgrade CUDA (high risk, likely break other things)
- Run inference on CPU (orders of magnitude too slow)
- **Use llama.cpp's custom CUDA kernels via GGUF**

llama.cpp doesn't link cuBLAS. It ships its own gemm kernels. **It worked on the first try.**

This is why every variant in my benchmark — FP16, INT8, INT4 — runs through `llama.cpp` via a subprocess wrapper. It wasn't a clever architectural decision. It was the only thing that worked.

## The benchmark (the part that turned out to matter most)

With the inference stack settled on llama.cpp, the benchmark became straightforward. Same 169-item legal eval set. Same temperature, same seed. Same hardware. Three quantization levels.

**Setup**
- Hardware: RTX 5090, WSL2 Ubuntu 22.04, CUDA 12.8
- Inference: `llama.cpp` (compiled with CUDA), `--n-gpu-layers 99`
- Eval: ROUGE-1/L vs reference answers
- "Memory" column: GGUF file size on disk (used as VRAM proxy — `nvidia-smi` is unavailable on WSL2 for GPU memory queries, so I'm reporting the deterministic on-disk number instead)

**Results**

```
Variant       Size      Latency    ROUGE-L
─────────────────────────────────────────
FP16          14.50 GB  10.8 ms    0.367
INT8 Q8_0      7.70 GB   6.8 ms    0.371
INT4 Q4_K_M    4.37 GB   4.9 ms    0.371
```

### What this means

**Q4_K_M is essentially free quality.** I expected an INT4 cliff. There isn't one. Within ROUGE-L noise, all three variants are equivalent. The fact that INT8 and INT4 measure slightly higher than FP16 is noise too — but it's the right kind of noise: zero degradation, not "tolerable loss."

**The memory delta is the real story.** 14.5 GB → 4.4 GB means the model fits on a laptop GPU, on a Mac M-series via Metal, or on cheap cloud instances (T4, L4) instead of A10/A100. The economics of *who can deploy this* shift completely. A district lawyer on a Lenovo gaming laptop is now in scope.

**INT8 is a trap.** It's strictly dominated. Same quality as FP16 and INT4, but bigger and slower than INT4. The only reason to ship INT8 is if your runtime doesn't support k-quants — and `llama.cpp`, `ollama`, and `vLLM` all do.

## Day 5: the apples-to-apples A/B

With the cuBLAS issue forcing everything through llama.cpp, my Day 5 base-vs-fine-tuned comparison had to be fair. I converted the HF-cached vanilla Mistral safetensors → FP16 GGUF → Q4_K_M GGUF using llama.cpp's tooling. Then the A/B is:

- **Base:** vanilla Mistral-7B-Instruct-v0.3, Q4_K_M GGUF
- **Fine-tuned:** NyayaGPT (merged base + LoRA), Q4_K_M GGUF
- **Quantization, hardware, inference engine, prompt template, seed: identical**

Fine-tuning is the *only* variable. This is the single most important methodological decision in the whole project. Most "look at my fine-tuned model!" demos compare an FP16 transformers run against an INT4 llama.cpp run and call any difference "the power of fine-tuning." That conflates two interventions. Don't do it.

## The MLOps stack

A fine-tune that lives only in your `~/work/` is not a project. The plumbing:

- **MLflow** — three experiments: `nyayagpt-training`, `nyayagpt-evaluation`, `nyayagpt-ab-test`. Every run logged with hyperparameters, eval metrics, and artifacts.
- **Streamlit A/B dashboard** — side-by-side base Mistral vs NyayaGPT-LoRA on the same prompt. This single artifact has done more to convince non-technical stakeholders the fine-tune was worth it than any benchmark table.
- **Hugging Face Hub** — LoRA adapter + model card with full eval methodology, live at [huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter](https://huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter). Pull it and merge it on top of any `Mistral-7B-Instruct-v0.3` base.
- **Reproducible scripts** — `scripts/build_base_mistral_gguf.py` rebuilds the comparison base from cache without re-downloading. One `bash` line from a clean machine to a trained, quantized model.

## What I'd do differently

1. **More data, less hyperparameter tuning.** Hyperparameter sweeps produced ~5 ROUGE points; doubling the dataset would have produced 10+.
2. **Build the eval set first.** I built it second, after I'd already started training. Means my early runs use a slightly different eval set than my late runs. Eval set is the spec; write the spec first.
3. **Plan for hardware uncertainty.** I assumed cuBLAS worked. It didn't. The right reflex is: **build the inference path on the artifact format you'll ship in (GGUF), not the format you train in.** Would have saved me a day.
4. **Add refusal training.** The model is too eager to answer questions outside Indian law. A small refusal dataset (300 OOD questions with "I'm specialized in Indian law") would cleanly fix this.
5. **Ship the demo before you ship the paper.** A live HuggingFace Space is the most-clicked link. Build it day one, not day seven.

## What's next

- **HF Spaces demo** — Gradio app on the free CPU tier with INT4 GGUF
- **NyayaGPT v2** — 5,000-pair dataset, refusal training, evaluation against retrieval-augmented baselines
- **Edge benchmarks** — Q4_K_M on M2 MacBook Air, Snapdragon X laptops, mid-range Windows gaming laptops

## If you're building domain LLMs

In order of leverage:

1. **60% of your time on the dataset.** Not the model. Not the trainer.
2. **Write the eval set before the training loop.**
3. **Quantize aggressively.** Q4_K_M is free quality on most modern hardware.
4. **Build the comparison at the same quantization level.** Otherwise you're not measuring what you think.
5. **Pick a niche where the *vocabulary* is specialized, not just the *knowledge*.** Knowledge is what RAG is for.

Code, dataset prep scripts, training notebooks, quantization benchmark, MLflow runs, and Streamlit dashboard are all open-source: [github.com/gauravgarwal9011/NyayGPT](https://github.com/gauravgarwal9011/NyayGPT). Trained LoRA adapter on the Hub: [huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter](https://huggingface.co/gauravgarwal/NyayaGPT-Mistral7B-adapter).

If you're working on similar domain-specific LLM projects in India — legal, medical, regional-language, financial — I'd love to hear from you. Reach me on [LinkedIn](https://www.linkedin.com/in/gauravgarwal/) or via email.

---

*Built by Gaurav Garwal, AI/ML Engineer at Ignatiuz Software Solutions, Indore. Opinions are my own.*

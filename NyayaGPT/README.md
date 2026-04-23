# NyayaGPT — Indian Legal LLM

> *Nyaya* (न्याय) — Sanskrit for **justice**

Fine-tuned Mistral 7B on 1,500 Indian legal instruction pairs via QLoRA on an NVIDIA RTX 5090.

## 7-Day Build Plan

| Day | Notebook | Goal |
|-----|----------|------|
| 1 | [01_dataset_collection](notebooks/01_dataset_collection.ipynb) | IndianKanoon scraper + GPT-4o synthetic Q&A → 1,500 pairs |
| 2 | [02_qlora_training](notebooks/02_qlora_training.ipynb) | QLoRA fine-tune Mistral 7B, log to MLflow |
| 3 | [03_evaluation](notebooks/03_evaluation.ipynb) | ROUGE + RAGAS: fine-tuned vs base Mistral |
| 4 | [04_quantization_benchmark](notebooks/04_quantization_benchmark.ipynb) | FP16 → INT8 → INT4 GGUF: memory, latency, ROUGE |
| 5 | [05_ab_test_dashboard](notebooks/05_ab_test_dashboard.ipynb) | Streamlit A/B dashboard with MLflow logging |
| 6 | [06_hf_hub_deployment](notebooks/06_hf_hub_deployment.ipynb) | Push model + detailed model card to HF Hub |
| 7 | [07_hf_spaces_demo](notebooks/07_hf_spaces_demo.ipynb) | Gradio demo on HF Spaces + PyTorch Profiler trace |

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Expected training time | ~2–4 hours |

## Quick Start

```bash
cd /home/ubuntu/Fine-tuning/NyayaGPT

# Install dependencies
pip install -r requirements.txt

# Day 1: Collect dataset
jupyter notebook notebooks/01_dataset_collection.ipynb

# Day 2: Train
jupyter notebook notebooks/02_qlora_training.ipynb

# Launch MLflow UI
mlflow ui --backend-store-uri mlruns/ --port 5000
```

## Model

- **Base:** `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- **Method:** QLoRA (LoRA r=16, α=32, 4-bit base)
- **Dataset:** 1,500 Indian legal Q&A pairs (IndianKanoon + GPT-4o synthetic)
- **Task:** Instruction following for Indian legal queries

## Project Structure

```
NyayaGPT/
├── notebooks/          # Day-by-day Jupyter notebooks
├── src/nyaya_pipeline/ # Modular Python package (post-notebook)
├── adapters/           # Trained LoRA weights
├── output/             # Datasets (train.jsonl, eval.jsonl)
├── mlruns/             # MLflow experiment store
├── assets/             # Screenshots, profiler traces
└── app.py              # HF Spaces Gradio entrypoint
```

"""
build_gguf.py — Merge the NyayaGPT LoRA adapter into Mistral 7B base weights
and quantize the merged model to Q4_K_M GGUF for INT4 benchmarking / deployment.

Pipeline:
  1. Load base + adapter via Unsloth (adapter dir as model_name → auto-merge)
  2. Save merged FP16 weights to SCRATCH_DIR/merged_model/
  3. Convert HF → FP16 GGUF via llama.cpp's convert_hf_to_gguf.py
  4. Quantize FP16 GGUF → Q4_K_M via llama.cpp's llama-quantize

IMPORTANT — storage layout:
  All ~32 GB of intermediate files (merged FP16 + FP16 GGUF + Q4_K_M) are
  written to SCRATCH_DIR (default /mnt/f/NyayaGPT-scratch) to avoid growing
  the WSL2 ext4.vhdx file, which lives on /mnt/e with limited free space.
  Only a symlink to the final Q4_K_M GGUF is placed in the project's
  adapters/ directory.

Prereq: llama.cpp built at ~/.unsloth/llama.cpp (quantize binary + convert script)

Usage:
    /home/ubuntu/Fine-tuning/.venv/bin/python scripts/build_gguf.py
    # override adapter / scratch location:
    ADAPTER_DIR=adapters-3ep SCRATCH_DIR=/mnt/f/nyaya-3ep \
      OUTPUT_NAME=nyayagpt-3ep-q4km.gguf python scripts/build_gguf.py
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", "adapters-2ep"))
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR", "/mnt/f/NyayaGPT-scratch"))
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "nyayagpt-q4km.gguf")

MERGED_DIR  = SCRATCH_DIR / "merged_model"
FP16_GGUF   = SCRATCH_DIR / "nyayagpt-fp16.gguf"
OUTPUT_PATH = SCRATCH_DIR / OUTPUT_NAME
SYMLINK_IN_PROJECT = ADAPTER_DIR / OUTPUT_NAME

LLAMA_CPP_DIR   = Path.home() / ".unsloth" / "llama.cpp"
CONVERT_SCRIPT  = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BINARY = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

if not ADAPTER_DIR.exists():
    sys.exit(f"Adapter not found at {ADAPTER_DIR.resolve()}")
if not CONVERT_SCRIPT.exists():
    sys.exit(f"llama.cpp convert script not found at {CONVERT_SCRIPT}")
if not QUANTIZE_BINARY.exists():
    sys.exit(f"llama.cpp quantize binary not found at {QUANTIZE_BINARY}")

SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

print(f"→ Source adapter:  {ADAPTER_DIR.resolve()}")
print(f"→ Scratch dir:     {SCRATCH_DIR}  (intermediates, ~32 GB)")
print(f"→ Merged FP16:     {MERGED_DIR}")
print(f"→ FP16 GGUF:       {FP16_GGUF}")
print(f"→ Q4_K_M GGUF:     {OUTPUT_PATH}  (~4 GB)")
print(f"→ Project symlink: {SYMLINK_IN_PROJECT}")
print()

# ── 1. Load base + adapter via Unsloth (adapter dir as model_name auto-wraps PeftModel) ─
print("[1/4] Loading base + adapter via Unsloth …")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),
    max_seq_length=2048,
    load_in_4bit=False,
    dtype=None,
)

# ── 2. Save merged FP16 weights to scratch ──────────────────────────────────
print("[2/4] Saving merged FP16 weights to scratch …")
print("      (slow write — /mnt/f is cross-fs, expect ~10-15 min for 14 GB)")
model.save_pretrained_merged(
    str(MERGED_DIR),
    tokenizer,
    save_method="merged_16bit",
)

# Free VRAM + RAM before heavy CPU subprocesses
import gc, torch
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── 3. Convert HF → FP16 GGUF ────────────────────────────────────────────────
print(f"[3/4] Converting HF → FP16 GGUF via {CONVERT_SCRIPT.name} …")
subprocess.run(
    [
        sys.executable, str(CONVERT_SCRIPT),
        str(MERGED_DIR),
        "--outtype", "f16",
        "--outfile", str(FP16_GGUF),
    ],
    check=True,
)

# ── 4. Quantize FP16 GGUF → Q4_K_M ───────────────────────────────────────────
print("[4/4] Quantizing FP16 GGUF → Q4_K_M …")
subprocess.run(
    [str(QUANTIZE_BINARY), str(FP16_GGUF), str(OUTPUT_PATH), "Q4_K_M"],
    check=True,
)

# ── 5. Symlink final Q4_K_M into project adapters/ dir for notebook discovery ─
SYMLINK_IN_PROJECT.parent.mkdir(parents=True, exist_ok=True)
if SYMLINK_IN_PROJECT.is_symlink() or SYMLINK_IN_PROJECT.exists():
    SYMLINK_IN_PROJECT.unlink()
SYMLINK_IN_PROJECT.symlink_to(OUTPUT_PATH.resolve())

size_gb = OUTPUT_PATH.stat().st_size / 1e9
print(f"\n✓ Built {OUTPUT_PATH}  ({size_gb:.2f} GB)")
print(f"✓ Symlinked → {SYMLINK_IN_PROJECT}")
print(f"  Notebook 04 will find the GGUF via the symlink.")
print()
print("ℹ Intermediates (merged FP16 ~14 GB, FP16 GGUF ~14 GB) remain in scratch.")
print(f"  To reclaim ~28 GB later:  rm -rf {MERGED_DIR} {FP16_GGUF}")

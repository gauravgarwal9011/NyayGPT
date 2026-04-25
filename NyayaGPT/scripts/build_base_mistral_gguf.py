"""
build_base_mistral_gguf.py — Convert the HF-cached vanilla Mistral-7B-Instruct-v0.3
to a Q4_K_M GGUF so it can be A/B-tested against NyayaGPT's Q4_K_M on the same
llama.cpp runtime. No new download required — reuses the ~14 GB safetensors
already in ~/.cache/huggingface/hub/.

Pipeline:
  1. Locate the HF snapshot dir for mistralai/Mistral-7B-Instruct-v0.3
  2. Convert HF safetensors → FP16 GGUF via llama.cpp's convert_hf_to_gguf.py
  3. Quantize FP16 GGUF → Q4_K_M via llama-quantize

Outputs:
  /mnt/f/NyayaGPT-scratch/mistral-base-fp16.gguf   (~14 GB, intermediate)
  /mnt/f/NyayaGPT-scratch/mistral-base-q4km.gguf   (~4.4 GB, A/B candidate)

Prereq: llama.cpp built at ~/.unsloth/llama.cpp (same as scripts/build_gguf.py).

Usage:
    /home/ubuntu/Fine-tuning/.venv/bin/python scripts/build_base_mistral_gguf.py
"""
import os
import subprocess
import sys
from pathlib import Path

HF_CACHE    = Path.home() / ".cache" / "huggingface" / "hub"
MISTRAL_DIR = HF_CACHE / "models--mistralai--Mistral-7B-Instruct-v0.3"
SNAPSHOTS   = MISTRAL_DIR / "snapshots"

SCRATCH   = Path(os.environ.get("SCRATCH_DIR", "/mnt/f/NyayaGPT-scratch"))
FP16_GGUF = SCRATCH / "mistral-base-fp16.gguf"
Q4KM_GGUF = SCRATCH / "mistral-base-q4km.gguf"

LLAMA_CPP_DIR   = Path.home() / ".unsloth" / "llama.cpp"
CONVERT_SCRIPT  = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BINARY = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

if not SNAPSHOTS.exists():
    sys.exit(
        f"No snapshot dir at {SNAPSHOTS}. "
        "Run 'huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3' first."
    )
snapshot_dirs = [d for d in SNAPSHOTS.iterdir() if d.is_dir()]
if not snapshot_dirs:
    sys.exit(f"No snapshot found under {SNAPSHOTS}.")
snapshot = snapshot_dirs[0]

if not CONVERT_SCRIPT.exists():
    sys.exit(f"llama.cpp convert script not found at {CONVERT_SCRIPT}")
if not QUANTIZE_BINARY.exists():
    sys.exit(f"llama.cpp quantize binary not found at {QUANTIZE_BINARY}")

SCRATCH.mkdir(parents=True, exist_ok=True)

print(f"→ HF snapshot:    {snapshot}")
print(f"→ FP16 GGUF:      {FP16_GGUF}")
print(f"→ Q4_K_M GGUF:    {Q4KM_GGUF}")
print()

if Q4KM_GGUF.exists():
    size_gb = Q4KM_GGUF.stat().st_size / 1e9
    print(f"✓ Q4_K_M GGUF already exists ({size_gb:.2f} GB). Nothing to do.")
    sys.exit(0)

# ── 1. HF safetensors → FP16 GGUF ────────────────────────────────────────────
if not FP16_GGUF.exists():
    print("[1/2] Converting HF safetensors → FP16 GGUF (CPU-bound, ~5-8 min) …")
    subprocess.run(
        [
            sys.executable, str(CONVERT_SCRIPT),
            str(snapshot),
            "--outtype", "f16",
            "--outfile", str(FP16_GGUF),
        ],
        check=True,
    )
else:
    size_gb = FP16_GGUF.stat().st_size / 1e9
    print(f"[1/2] FP16 GGUF already exists ({size_gb:.2f} GB) — skipping conversion.")

# ── 2. FP16 GGUF → Q4_K_M ────────────────────────────────────────────────────
print("\n[2/2] Quantizing FP16 GGUF → Q4_K_M (~1-2 min) …")
subprocess.run(
    [str(QUANTIZE_BINARY), str(FP16_GGUF), str(Q4KM_GGUF), "Q4_K_M"],
    check=True,
)

size_gb = Q4KM_GGUF.stat().st_size / 1e9
print(f"\n✓ Built {Q4KM_GGUF}  ({size_gb:.2f} GB)")
print()
print(f"ℹ Reclaim ~14 GB intermediate later:  rm {FP16_GGUF}")

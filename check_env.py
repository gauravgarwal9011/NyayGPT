# check_env.py — Full environment check for LLM finetuning
# Run: python check_env.py

import sys, importlib, subprocess

PASS = " PASS"
FAIL = " FAIL"
WARN = "  WARN"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}: {e}")
        return False

print("\n========================================")
print("   LLM Finetuning — Environment Check")
print("========================================\n")

# ── 1. Python version ──────────────────────────────
print("[1] Python")
check("Python version", lambda: sys.version.split()[0])

# ── 2. PyTorch + CUDA ──────────────────────────────
print("\n[2] PyTorch + CUDA")
import torch
check("PyTorch version",   lambda: torch.__version__)
check("CUDA available",    lambda: "YES" if torch.cuda.is_available() else (_ for _ in ()).throw(Exception("CUDA not found")))
check("CUDA version",      lambda: torch.version.cuda)
check("GPU name",          lambda: torch.cuda.get_device_name(0))
check("GPU VRAM",          lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
check("bf16 supported",   lambda: "YES" if torch.cuda.is_bf16_supported() else "NO (use fp16 instead)")
check("Flash Attention",  lambda: "YES" if torch.backends.cuda.flash_sdp_enabled() else "not enabled")

# ── 3. Core libraries ──────────────────────────────
print("\n[3] Core finetuning libraries")
for lib in ["transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes", "sentencepiece"]:
    check(lib, lambda l=lib: importlib.import_module(l).__version__)

# ── 4. bitsandbytes CUDA check ─────────────────────
print("\n[4] bitsandbytes GPU support")
def check_bnb():
    import bitsandbytes as bnb
    from bitsandbytes.cuda_setup.main import get_compute_capability
    cc = get_compute_capability(torch.cuda.current_device())
    return f"compute capability {cc} — 4-bit quantization supported"
check("bnb 4-bit (QLoRA)", check_bnb)

# ── 5. Hugging Face login ──────────────────────────
print("\n[5] Hugging Face Hub")
def check_hf():
    from huggingface_hub import whoami
    info = whoami()
    return f"logged in as: {info['name']}"
check("HF Hub login", check_hf)

# ── 6. System RAM ──────────────────────────────────
print("\n[6] System RAM")
def check_ram():
    import psutil
    gb = psutil.virtual_memory().total / 1e9
    flag = "(upgrade to 64GB recommended)" if gb < 48 else "(sufficient)"
    return f"{gb:.1f} GB {flag}"
check("System RAM", check_ram)

# ── 7. Disk space ──────────────────────────────────
print("\n[7] Disk space")
def check_disk():
    import shutil
    free = shutil.disk_usage("/").free / 1e9
    flag = "  low — need 50GB+ free for model weights" if free < 50 else "sufficient"
    return f"{free:.1f} GB free — {flag}"
check("Free disk space", check_disk)

# ── 8. Quick model load test ───────────────────────
print("\n[8] Quick model load test (tiny model)")
def check_model_load():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("hello world", return_tensors="pt")
    return f"tokenizer OK — {len(ids['input_ids'][0])} tokens"
check("Transformers pipeline", check_model_load)

# ── Summary ────────────────────────────────────────
print("\n========================================\n")
print("All checks done. Fix any  FAIL items before training.\n")
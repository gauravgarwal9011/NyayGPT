"""
config.py — Centralised configuration for NyayaGPT pipeline.
All constants are env-var overridable for CI/Docker use.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = Path(os.environ.get("NY_OUTPUT_DIR",  PROJECT_ROOT / "output"))
LOG_DIR      = Path(os.environ.get("NY_LOG_DIR",     PROJECT_ROOT / "logs"))
ADAPTER_DIR  = Path(os.environ.get("NY_ADAPTER_DIR", PROJECT_ROOT / "adapters"))
MLFLOW_URI   = os.environ.get("NY_MLFLOW_URI", str(PROJECT_ROOT / "mlruns"))

# ── Teacher model (GGUF fallback) ────────────────────────────────────────────
# Use a Linux-native path for fast loading — /mnt/c/ WSL cross-fs reads are ~10× slower.
# Copy once: cp "/mnt/c/Users/Tron/.cache/.../gpt-oss-20b-F16.gguf" ~/models/
TEACHER_GGUF_PATH = os.environ.get(
    "NY_TEACHER_GGUF",
    str(Path.home() / "models" / "gpt-oss-20b-F16.gguf"),
)

# ── Student model ────────────────────────────────────────────────────────────
STUDENT_MODEL_NAME   = os.environ.get("NY_STUDENT_MODEL", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
STUDENT_MAX_SEQ_LEN  = int(os.environ.get("NY_MAX_SEQ", 2048))
STUDENT_LOAD_IN_4BIT = os.environ.get("NY_LOAD_4BIT", "1") == "1"

# ── LoRA ─────────────────────────────────────────────────────────────────────
LORA_R       = int(os.environ.get("NY_LORA_R", 16))
LORA_ALPHA   = int(os.environ.get("NY_LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("NY_LORA_DROPOUT", 0.0))

# ── Training ─────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE  = int(os.environ.get("NY_TRAIN_BATCH", 2))
GRAD_ACCUM_STEPS  = int(os.environ.get("NY_GRAD_ACCUM", 4))
LEARNING_RATE     = float(os.environ.get("NY_LR", 2e-4))
NUM_TRAIN_EPOCHS  = int(os.environ.get("NY_EPOCHS", 2))
WARMUP_RATIO      = float(os.environ.get("NY_WARMUP", 0.1))
WEIGHT_DECAY      = float(os.environ.get("NY_WD", 0.01))

# ── Dataset ──────────────────────────────────────────────────────────────────
TRAIN_SPLIT  = float(os.environ.get("NY_TRAIN_SPLIT", 0.9))
RANDOM_SEED  = int(os.environ.get("NY_SEED", 42))
TRAIN_JSONL  = OUTPUT_DIR / "train.jsonl"
EVAL_JSONL   = OUTPUT_DIR / "eval.jsonl"

# ── Data collection ──────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.environ.get("NY_CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.environ.get("NY_CHUNK_OVERLAP", 150))
QA_PER_CHUNK    = int(os.environ.get("NY_QA_PER_CHUNK", 3))
TARGET_PAIRS    = int(os.environ.get("NY_TARGET_PAIRS", 1500))

# ── Quality filter ───────────────────────────────────────────────────────────
MIN_RESPONSE_LENGTH = int(os.environ.get("NY_MIN_RESP_LEN", 100))
MIN_KEYWORD_OVERLAP = int(os.environ.get("NY_MIN_KW_OVERLAP", 6))

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NyayaGPT, an expert Indian legal assistant trained on \
Indian Kanoon case law, IPC sections, constitutional provisions, and judicial judgments.
When answering:
1. Cite the relevant IPC section, article, or case law if known.
2. Explain legal concepts in plain language.
3. If the question is outside Indian jurisdiction or your knowledge, say so clearly.
4. Never provide advice that could replace consultation with a licensed advocate."""

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
AZURE_OPENAI_KEY      = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_TRAIN = "nyayagpt-training"
MLFLOW_EXPERIMENT_EVAL  = "nyayagpt-evaluation"
MLFLOW_EXPERIMENT_AB    = "nyayagpt-ab-test"


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

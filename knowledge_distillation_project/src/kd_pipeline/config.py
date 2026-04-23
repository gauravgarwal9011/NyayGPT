"""
config.py
=========
Centralised configuration for the entire knowledge-distillation pipeline.

WHY a separate config module?
-----------------------------
Hard-coding constants (paths, hyperparameters, prompts) inside business-logic
files makes them hard to change and impossible to override per environment.
By putting every "knob" in one place, you can:
  1. Tune the pipeline without grepping the codebase.
  2. Override values from environment variables (handy for Docker / CI).
  3. Reuse the same constants in tests, scripts, and notebooks.
"""

# `os` lets us read environment variables — the standard way to override
# config in production without editing source files.
import os

# `pathlib.Path` is the modern, object-oriented replacement for raw string
# paths. It works correctly on both Linux/WSL and Windows, and supports
# operations like `path / "subdir" / "file.txt"`.
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# `__file__` is the absolute path to *this* config.py file at runtime.
# `.resolve()` follows symlinks and returns the canonical absolute path.
# `.parents[3]` walks up three directories:
#   parents[0] = kd_pipeline/   (this file's folder)
#   parents[1] = src/            (parent)
#   parents[2] = knowledge_distillation_project/  (project root)
# This gives us a stable PROJECT_ROOT that does not depend on the user's
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder where all generated artifacts (chunks, datasets, JSON dumps) live.
# `os.environ.get("KD_OUTPUT_DIR", default)` first looks at the env var; if
# the user set KD_OUTPUT_DIR before running, that overrides the default.
OUTPUT_DIR = Path(os.environ.get("KD_OUTPUT_DIR", PROJECT_ROOT / "output"))

# Folder where rotating log files will be stored.
LOG_DIR = Path(os.environ.get("KD_LOG_DIR", PROJECT_ROOT / "logs"))

# Source PDF that the pipeline ingests. The end-user should override this
# (either via env var or by editing this file) to point at their own PDF.
PDF_PATH = os.environ.get(
    "KD_PDF_PATH",
    "/mnt/e/Ignatiuz/Ignatiuz_AIML capabilities deck.pdf",
)

# Path to the GGUF teacher model file. GGUF is the binary format used by
# llama.cpp / llama-cpp-python. /mnt/c/ is the WSL mount point for the
# Windows C: drive — adjust if your model lives elsewhere.
MODEL_PATH = os.environ.get(
    "KD_MODEL_PATH",
    "/mnt/c/Users/Tron/.cache/huggingface/hub/"
    "models--unsloth--gpt-oss-20b-GGUF/snapshots/"
    "d449b42d93e1c2c7bda5312f5c25c8fb91dfa9b4/gpt-oss-20b-F16.gguf",
)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Target size in characters for each text chunk fed to the teacher.
# 600 chars ≈ 100–120 words ≈ roughly one slide or one paragraph.
# Smaller = more focused Q&A, more samples, less per-prompt cost.
# Larger = more context per question but fewer training samples.
CHUNK_SIZE = int(os.environ.get("KD_CHUNK_SIZE", 600))

# Characters of overlap between consecutive chunks. Overlap prevents a key
# fact from being split across two chunks (e.g., "saved 1000 hours per
# month" being cut between "saved 1000" and "hours per month").
CHUNK_OVERLAP = int(os.environ.get("KD_CHUNK_OVERLAP", 100))


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Number of direct (non-CoT) Q&A pairs to generate per chunk.
# 3 is a good balance — gives ~120 direct samples from a 40-chunk deck.
QA_PER_CHUNK = int(os.environ.get("KD_QA_PER_CHUNK", 3))

# Sampling temperature for direct (factual) answer generation.
# 0.2 = nearly deterministic. We want the teacher to *quote* the document
# faithfully, not creatively rephrase it.
TEMPERATURE_DIRECT = float(os.environ.get("KD_TEMP_DIRECT", 0.2))

# Sampling temperature for chain-of-thought (CoT) reasoning generation.
# 0.4 = mild variation, so two similar reasoning traces don't end up
# identical. Above 0.7 the teacher starts to drift off-topic.
TEMPERATURE_COT = float(os.environ.get("KD_TEMP_COT", 0.4))

# Maximum new tokens to generate per direct answer.
# 350 ≈ a complete factual response without truncation.
MAX_TOKENS_DIRECT = int(os.environ.get("KD_MAX_TOK_DIRECT", 350))

# Maximum new tokens for CoT — must be larger because the model emits a
# multi-step reasoning trace before the final answer.
MAX_TOKENS_COT = int(os.environ.get("KD_MAX_TOK_COT", 700))

# Penalty applied to tokens the model has already produced. 1.0 = no
# penalty, 1.15 = gentle anti-repetition. Above ~1.3 the output gets weird.
REPEAT_PENALTY = float(os.environ.get("KD_REPEAT_PENALTY", 1.15))


# ─────────────────────────────────────────────────────────────────────────────
# TEACHER MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Number of transformer layers to offload to the GPU.
#   -1 = offload ALL layers (full GPU inference, fastest)
#    0 = pure CPU
#    N = put N layers on GPU and rest in system RAM
N_GPU_LAYERS = int(os.environ.get("KD_N_GPU_LAYERS", -1))

# Context window size in tokens — the maximum prompt + completion length
# the model can attend to. 2048 is enough for chunk + question + answer.
N_CTX = int(os.environ.get("KD_N_CTX", 2048))


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY-FILTER THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# Minimum response length (chars) before we consider the response usable.
# Anything shorter is almost certainly an empty/refusal output.
MIN_RESPONSE_LENGTH = int(os.environ.get("KD_MIN_RESP_LEN", 80))

# Minimum number of significant words (length ≥ 5) that must appear in
# BOTH the chunk and the response. Below this threshold the response is
# probably hallucinated from general knowledge instead of grounded.
MIN_KEYWORD_OVERLAP = int(os.environ.get("KD_MIN_KW_OVERLAP", 5))


# ─────────────────────────────────────────────────────────────────────────────
# DATASET-SPLIT HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Fraction of the dataset reserved for training (the rest becomes eval).
# 0.9 = 90% train / 10% eval — a common default for small datasets.
TRAIN_SPLIT = float(os.environ.get("KD_TRAIN_SPLIT", 0.9))

# Random seed used to shuffle the dataset before splitting. Fixing this
# value makes the train/eval split *reproducible* across runs.
RANDOM_SEED = int(os.environ.get("KD_RANDOM_SEED", 42))


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
#
# This string is the system instruction injected into every conversation
# we ask the teacher model to respond to. It does the heavy lifting of
# grounding the teacher to the document, and the same prompt becomes the
# system message in the student's training samples.
SYSTEM_PROMPT = """You are an expert enterprise AI/ML consultant at Ignatiuz.
You have access to the exact content from Ignatiuz's capabilities document.
When answering, you must:
1. Base your answer ONLY on the provided document excerpt — do not add external knowledge.
2. Use specific numbers, names, and results exactly as stated in the excerpt.
3. If the excerpt does not contain the answer, say "This information is not in the provided section."
"""


# ─────────────────────────────────────────────────────────────────────────────
# STUDENT TRAINING (Unsloth + LoRA) HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Hugging Face / Unsloth model id of the student to fine-tune. Unsloth's
# bnb-4bit variants are pre-quantised and load ~4x faster than vanilla HF.
STUDENT_MODEL_NAME = os.environ.get(
    "KD_STUDENT_MODEL", "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
)

# Maximum sequence length the student will see during training. 2048 is
# enough for our SYSTEM + USER + ASSISTANT samples (the longest CoT
# samples top out around 1500 tokens).
STUDENT_MAX_SEQ_LEN = int(os.environ.get("KD_STUDENT_MAX_SEQ", 2048))

# Load the base model in 4-bit (QLoRA-style). With a 3B model on a 32GB
# Blackwell card this is overkill — but it leaves headroom for higher
# batch sizes and longer sequences without OOM.
STUDENT_LOAD_IN_4BIT = os.environ.get("KD_STUDENT_4BIT", "1") == "1"

# LoRA rank (`r`). 16 is the sweet spot for small datasets — bigger r
# overfits, smaller r underfits. Alpha is conventionally 2*r.
LORA_R = int(os.environ.get("KD_LORA_R", 16))
LORA_ALPHA = int(os.environ.get("KD_LORA_ALPHA", 32))

# LoRA dropout. 0.0 is fastest (Unsloth's optimised path); raise to 0.05
# if you see overfitting on the eval split.
LORA_DROPOUT = float(os.environ.get("KD_LORA_DROPOUT", 0.0))

# Per-device training batch size. With Qwen2.5-3B in 4-bit + LoRA on a
# 32GB card, 2 samples × 2048 seq is comfortable.
TRAIN_BATCH_SIZE = int(os.environ.get("KD_TRAIN_BATCH", 2))

# Gradient accumulation: effective batch = TRAIN_BATCH_SIZE * GRAD_ACCUM.
# Effective 8 is a stable default for SFT on small datasets.
GRAD_ACCUM_STEPS = int(os.environ.get("KD_GRAD_ACCUM", 4))

# Learning rate. 2e-4 is the LoRA-canonical value used in QLoRA paper.
LEARNING_RATE = float(os.environ.get("KD_LEARNING_RATE", 2e-4))

# Number of full passes over the train set. With only ~100 samples, 3
# epochs gives the model enough exposure without overfitting hard.
NUM_TRAIN_EPOCHS = int(os.environ.get("KD_NUM_EPOCHS", 3))

# Linear warmup as a fraction of total steps — helps the optimiser
# settle before LoRA weights are actually updated.
WARMUP_RATIO = float(os.environ.get("KD_WARMUP_RATIO", 0.1))

# Weight decay applied to LoRA params. 0.01 is a mild regulariser.
WEIGHT_DECAY = float(os.environ.get("KD_WEIGHT_DECAY", 0.01))

# Where the trained LoRA adapter (and any merged checkpoints) get saved.
ADAPTER_DIR = Path(os.environ.get("KD_ADAPTER_DIR", PROJECT_ROOT / "adapters"))

# Path to the train/eval JSONL files (produced by dataset_generator).
TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
EVAL_JSONL  = OUTPUT_DIR / "eval.jsonl"


def ensure_directories() -> None:
    """
    Create OUTPUT_DIR and LOG_DIR on disk if they do not already exist.

    Why a function instead of doing it at import time?
    --------------------------------------------------
    Side-effects at import time (like creating directories) make modules
    surprising and hard to test. Instead, the entry-point script calls
    `ensure_directories()` once at startup. Tests can skip it.
    """
    # `parents=True` creates intermediate directories as needed (like `mkdir -p`).
    # `exist_ok=True` silences the error if the directory already exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

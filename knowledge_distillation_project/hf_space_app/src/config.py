import json
import os
from pathlib import Path


APP_TITLE = "Ignatiuz Student Chat And Benchmark"
APP_DESCRIPTION = (
    "Chat with a fine-tuned student model and compare FP16, INT8, and INT4 GGUF "
    "variants on memory, latency, and ROUGE."
)

BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_FILE = Path(os.environ.get("HF_BENCHMARK_FILE", BASE_DIR / "benchmark_prompts.jsonl"))

BASE_MODEL_ID = os.environ.get("HF_BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_REPO_ID = os.environ.get("HF_ADAPTER_REPO_ID", "")
GGUF_MODEL_FILE = os.environ.get("HF_GGUF_MODEL_FILE", "")
SYSTEM_PROMPT = os.environ.get(
    "HF_SYSTEM_PROMPT",
    "You are a helpful enterprise AI assistant. Answer clearly and concisely.",
)

DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", 256))
DEFAULT_TEMPERATURE = float(os.environ.get("HF_TEMPERATURE", 0.2))
DEFAULT_BENCHMARK_LIMIT = int(os.environ.get("HF_BENCHMARK_LIMIT", 3))


def load_benchmark_prompts(limit: int | None = None) -> list[dict]:
    rows = []
    with BENCHMARK_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit is not None else rows

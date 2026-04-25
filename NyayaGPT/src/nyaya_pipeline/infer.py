"""
infer.py — Inference + A/B routing for NyayaGPT.

Provides:
  1. Single-shot generation (for evaluation scripts)
  2. Interactive REPL
  3. A/B routing: randomly labels a request between two deployment variants
"""
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from . import config
from .exceptions import InferenceError, ConfigurationError
from .logger import get_logger

log = get_logger(__name__)

_model_cache: dict = {}
_gguf_cache: dict = {}

GGUF_SCRATCH_DIR = Path(os.environ.get("NY_GGUF_SCRATCH_DIR", "/mnt/f/NyayaGPT-scratch"))
NYAYAGPT_FP16_GGUF = Path(os.environ.get("NY_FP16_GGUF_PATH", GGUF_SCRATCH_DIR / "nyayagpt-fp16.gguf"))
NYAYAGPT_Q8_GGUF = Path(os.environ.get("NY_INT8_GGUF_PATH", GGUF_SCRATCH_DIR / "nyayagpt-q8_0.gguf"))
NYAYAGPT_Q4_GGUF = Path(os.environ.get("NY_INT4_GGUF_PATH", config.ADAPTER_DIR / "nyayagpt-q4km.gguf"))
MISTRAL_BASE_GGUF = Path(os.environ.get("NY_BASE_GGUF_PATH", GGUF_SCRATCH_DIR / "mistral-base-q4km.gguf"))


def _load_model(model_name: str, adapter_dir: Optional[Path] = None):
    cache_key = f"{model_name}::{adapter_dir}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as exc:
        raise InferenceError("unsloth not installed") from exc

    if not __import__("torch").cuda.is_available():
        raise InferenceError("CUDA not available — GPU required for inference.")

    log.info("Loading model: %s (adapter=%s)", model_name, adapter_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.STUDENT_MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_dir and Path(adapter_dir).exists():
        model.load_adapter(str(adapter_dir))
    FastLanguageModel.for_inference(model)

    _model_cache[cache_key] = (model, tokenizer)
    return model, tokenizer


def generate(
    question: str,
    model_name: Optional[str] = None,
    adapter_dir: Optional[Path] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate a single response from the NyayaGPT model."""
    model_name  = model_name  or config.STUDENT_MODEL_NAME
    adapter_dir = adapter_dir or config.ADAPTER_DIR

    model, tokenizer = _load_model(model_name, adapter_dir)

    messages = [
        {"role": "system", "content": system_prompt or config.SYSTEM_PROMPT},
        {"role": "user",   "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids    = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_in   = ids["input_ids"].shape[1]

    import torch
    gen_kw = {"max_new_tokens": max_new_tokens, "use_cache": True, "do_sample": temperature > 0}
    if temperature > 0:
        gen_kw["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**ids, **gen_kw)

    return tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def _load_gguf_model(gguf_path: Path):
    gguf_path = Path(gguf_path)
    cache_key = str(gguf_path.resolve())
    if cache_key in _gguf_cache:
        return _gguf_cache[cache_key]

    if not gguf_path.exists():
        raise InferenceError(f"GGUF not found: {gguf_path}")

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise InferenceError("llama-cpp-python not installed") from exc

    log.info("Loading GGUF model: %s", gguf_path)
    log.info("(silent for ~20s while weights transfer to GPU)")

    # Suppress verbose C++ stderr during load to keep notebooks responsive.
    import ctypes
    _old_stderr_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)
    try:
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=2048,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False,
        )
    finally:
        os.dup2(_old_stderr_fd, 2)
        os.close(_old_stderr_fd)

    _gguf_cache[cache_key] = llm
    return llm


def generate_gguf(
    question: str,
    gguf_path: Path,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
) -> str:
    llm = _load_gguf_model(gguf_path)
    prompt = f"[INST] {(system_prompt or config.SYSTEM_PROMPT).strip()}\n\n{question.strip()}[/INST]"

    out = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        repeat_penalty=1.1,
    )
    return out["choices"][0]["text"].strip()


def ab_generate(
    question: str,
    base_model: Optional[str] = None,
    finetuned_model: Optional[str] = None,
    adapter_dir: Optional[Path] = None,
    max_new_tokens: int = 256,
) -> Tuple[str, str, str, float, float]:
    """
    A/B routing comparing vanilla Mistral against NyayaGPT, both at Q4_K_M GGUF.

    Uses identical quantization for both sides so the only variable is
    fine-tuning itself — every quality delta is attributable to training on
    Indian legal data:
      - "base"      -> vanilla Mistral-7B-Instruct-v0.3 Q4_K_M GGUF
      - "finetuned" -> NyayaGPT Q4_K_M GGUF

    Returns:
        (assigned_variant, base_response, finetuned_response, base_latency_ms, ft_latency_ms)
    """
    import mlflow

    del base_model, finetuned_model, adapter_dir  # legacy args retained for notebook/API stability
    variant         = random.choice(["base", "finetuned"])

    # Always generate both for side-by-side display. We avoid the HF/Unsloth
    # path here because Blackwell + CUDA 12.8 hits a cuBLAS failure in decode.
    t0 = time.perf_counter()
    base_resp = generate_gguf(
        question,
        MISTRAL_BASE_GGUF,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    base_ms   = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ft_resp = generate_gguf(
        question,
        NYAYAGPT_Q4_GGUF,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    ft_ms   = (time.perf_counter() - t0) * 1000

    # Log A/B event to MLflow
    try:
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_AB)
        with mlflow.start_run(run_name=f"ab_{variant}", nested=False):
            mlflow.log_params({
                "variant": variant,
                "question_len": len(question),
                "base_engine": "gguf-q4_k_m-vanilla-mistral",
                "finetuned_engine": "gguf-q4_k_m-nyaya",
                "quantization": "Q4_K_M (same for both)",
            })
            mlflow.log_metrics({
                "base_latency_ms":      base_ms,
                "finetuned_latency_ms": ft_ms,
                "latency_delta_ms":     ft_ms - base_ms,
            })
    except Exception as exc:
        log.warning("MLflow A/B logging failed: %s", exc)

    return variant, base_resp, ft_resp, round(base_ms, 1), round(ft_ms, 1)


def interactive_repl(model_name: Optional[str] = None, adapter_dir: Optional[Path] = None) -> int:
    """Start an interactive REPL."""
    print("NyayaGPT — Indian Legal Assistant")
    print("Type your question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("You> ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            resp = generate(question, model_name, adapter_dir)
            print(f"NyayaGPT> {resp}\n")
        except Exception as exc:
            print(f"[Error] {exc}")
    return 0

"""
synthetic_generator.py — Generate synthetic Indian legal Q&A pairs.

Strategy (in order of preference):
  1. Azure OpenAI GPT-4o  — if AZURE_OPENAI_KEY is set
  2. Local GGUF teacher   — gpt-oss-20b-F16.gguf via llama-cpp-python

Output format: Alpaca JSON + chat JSONL (both produced per sample).
"""
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from . import config
from .exceptions import GenerationError
from .logger import get_logger
from .quality_filter import is_quality_response

log = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

LEGAL_QA_TEMPLATES = [
    "Based on the following Indian legal text, what is the main legal principle or ruling?",
    "What legal rights or obligations are described in this excerpt?",
    "Summarize the court's reasoning or holding in this judgment excerpt.",
    "What IPC sections, constitutional articles, or statutes are discussed here?",
    "What are the key facts and legal issues in this case excerpt?",
    "How would a judge apply the legal rule described in this passage?",
    "What remedies or reliefs are available under the law described in this excerpt?",
]

COT_TEMPLATES = [
    "A client is facing a situation similar to the one described in this excerpt. Walk through the legal analysis step by step.",
    "Using this judgment excerpt, explain how an advocate would argue for the petitioner.",
    "Analyse the legal merits of the case described, citing the relevant provisions mentioned.",
]

GENERATION_SYSTEM_PROMPT = """You are a senior Indian advocate and legal scholar preparing educational \
training material for law students. The excerpts below are from published Indian court judgments \
available in the public domain and are used strictly for legal education and AI training.
Generate clear, accurate legal Q&A pairs based ONLY on the provided judgment excerpt.
- Use specific case names, IPC sections, and constitutional articles mentioned in the text.
- Do not invent citations not present in the excerpt.
- Answers should be 2–5 sentences, accurate, and legally precise.
- Focus on legal principles, procedural aspects, and statutory interpretation.
- Always produce a useful answer from whatever legal content the excerpt contains, even if brief.
"""


def _build_generation_prompt(chunk: str, question_template: str) -> str:
    return (
        f"Judgment excerpt:\n\"\"\"\n{chunk}\n\"\"\"\n\n"
        f"Question: {question_template}\n\n"
        f"Provide a concise, accurate answer based solely on the above excerpt:"
    )


# ── Azure OpenAI generator ────────────────────────────────────────────────────

def _generate_with_azure(chunk: str, templates: List[str]) -> List[Tuple[str, str]]:
    """Generate Q&A pairs using Azure OpenAI GPT-4o."""
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise GenerationError("openai package not installed. pip install openai") from exc

    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version="2024-08-01-preview",
    )

    pairs = []
    for template in templates:
        user_msg = _build_generation_prompt(chunk, template)
        try:
            resp = client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=400,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content
            if raw is None:
                # finish_reason="content_filter" — Azure filtered silently (200 response)
                log.debug("Azure returned null content (content_filter); skipping chunk")
                continue
            answer = raw.strip()
            if answer and "Insufficient excerpt" not in answer:
                pairs.append((template, answer))
        except Exception as exc:
            log.warning("Azure OpenAI call failed: %s", exc)

    return pairs


# ── Local GGUF teacher generator ──────────────────────────────────────────────

_llm_cache = None  # module-level cache to avoid reloading


def _get_gguf_llm():
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise GenerationError("llama-cpp-python not installed.") from exc

    if not os.path.exists(config.TEACHER_GGUF_PATH):
        raise GenerationError(
            f"Teacher GGUF not found: {config.TEACHER_GGUF_PATH}\n"
            "Set NY_TEACHER_GGUF env var or configure AZURE_OPENAI_KEY."
        )

    log.info("Loading GGUF teacher: %s", config.TEACHER_GGUF_PATH)
    log.info("(silent for ~20s while weights transfer to GPU — model is 13 GB)")

    # Suppress C++ stderr during load to prevent Jupyter IO buffer deadlock
    import sys, ctypes
    _libc = ctypes.CDLL(None)
    _old_stderr_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)
    try:
        _llm_cache = Llama(
            model_path=config.TEACHER_GGUF_PATH,
            n_ctx=2048,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False,
        )
    finally:
        os.dup2(_old_stderr_fd, 2)
        os.close(_old_stderr_fd)

    log.info("GGUF teacher loaded successfully")
    return _llm_cache


def _generate_with_gguf(chunk: str, templates: List[str]) -> List[Tuple[str, str]]:
    """Generate Q&A pairs using the local GGUF teacher model."""
    llm = _get_gguf_llm()
    pairs = []

    for template in templates:
        user_msg = _build_generation_prompt(chunk, template)
        full_prompt = (
            f"<|start|>system<|message|>{GENERATION_SYSTEM_PROMPT}<|end|>\n"
            f"<|start|>user<|message|>{user_msg}<|end|>\n"
            f"<|start|>assistant<|channel|>final<|message|>"
        )
        try:
            out = llm(
                full_prompt,
                max_tokens=250,
                temperature=0.3,
                repeat_penalty=1.15,
                stop=["<|end|>", "<|return|>"],
            )
            answer = out["choices"][0]["text"].strip()
            if answer and "Insufficient excerpt" not in answer:
                pairs.append((template, answer))
            elif not answer:
                log.debug("Empty response for template: %s", template[:40])
        except Exception as exc:
            log.warning("GGUF generation failed: %s", exc)

    return pairs


# ── Main generation orchestrator ──────────────────────────────────────────────

def generate_qa_pairs(
    chunks: List[str],
    qa_per_chunk: int = None,
    include_cot_every_n: int = 2,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate Q&A pairs from a list of text chunks.

    Returns:
        (alpaca_samples, chat_samples)
        alpaca_samples: [{"instruction": ..., "input": chunk, "output": ...}]
        chat_samples:   [{"messages": [...]}]
    """
    qa_per_chunk = qa_per_chunk or config.QA_PER_CHUNK
    use_azure    = bool(config.AZURE_OPENAI_KEY)

    if use_azure:
        log.info("Using Azure OpenAI GPT-4o for generation")
        _generate = _generate_with_azure
    else:
        log.info("Using local GGUF teacher for generation (Azure key not set)")
        _generate = _generate_with_gguf

    alpaca_samples: List[Dict] = []
    chat_samples:   List[Dict] = []
    rejected = 0

    for idx, chunk in enumerate(chunks):
        if len(alpaca_samples) >= config.TARGET_PAIRS:
            break

        # Pick direct templates
        direct_templates = random.sample(LEGAL_QA_TEMPLATES, min(qa_per_chunk, len(LEGAL_QA_TEMPLATES)))
        # Add CoT every N chunks
        templates = direct_templates
        if idx % include_cot_every_n == 0:
            templates = direct_templates + [random.choice(COT_TEMPLATES)]

        pairs = _generate(chunk, templates)

        for question, answer in pairs:
            ok, reason = is_quality_response(answer, chunk)
            if not ok:
                log.debug("Rejected sample (%s): %s...", reason, answer[:60])
                rejected += 1
                continue

            alpaca_samples.append({
                "instruction": question,
                "input":       chunk[:300],   # excerpt context
                "output":      answer,
            })
            chat_samples.append({
                "messages": [
                    {"role": "system",    "content": config.SYSTEM_PROMPT},
                    {"role": "user",      "content": f"{question}\n\nContext:\n{chunk[:600]}"},
                    {"role": "assistant", "content": answer},
                ]
            })

        if (idx + 1) % 20 == 0:
            log.info("Progress: %d chunks processed → %d samples collected, %d rejected",
                     idx + 1, len(alpaca_samples), rejected)

    log.info("Generation complete: %d accepted, %d rejected (%.1f%% acceptance)",
             len(alpaca_samples), rejected,
             100 * len(alpaca_samples) / max(1, len(alpaca_samples) + rejected))

    return alpaca_samples, chat_samples


def save_datasets(
    alpaca_samples: List[Dict],
    chat_samples:   List[Dict],
    train_split: float = None,
    seed: int = None,
) -> Tuple[Path, Path]:
    """
    Shuffle, split 90/10, and save train.jsonl + eval.jsonl.
    Returns (train_path, eval_path).
    """
    import random as rng
    train_split = train_split or config.TRAIN_SPLIT
    seed        = seed        or config.RANDOM_SEED

    rng.seed(seed)
    rng.shuffle(chat_samples)

    n_train = int(len(chat_samples) * train_split)
    train_samples = chat_samples[:n_train]
    eval_samples  = chat_samples[n_train:]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _write_jsonl(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    _write_jsonl(train_samples, config.TRAIN_JSONL)
    _write_jsonl(eval_samples,  config.EVAL_JSONL)

    # Also save Alpaca JSON for human review
    alpaca_path = config.OUTPUT_DIR / "dataset_alpaca.json"
    alpaca_path.write_text(json.dumps(alpaca_samples, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Saved %d train / %d eval samples", len(train_samples), len(eval_samples))
    log.info("  train → %s", config.TRAIN_JSONL)
    log.info("  eval  → %s", config.EVAL_JSONL)
    log.info("  alpaca→ %s", alpaca_path)

    return config.TRAIN_JSONL, config.EVAL_JSONL

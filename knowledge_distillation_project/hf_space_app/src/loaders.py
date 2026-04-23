from __future__ import annotations

import time
from dataclasses import dataclass

from .config import ADAPTER_REPO_ID, BASE_MODEL_ID, GGUF_MODEL_FILE


@dataclass
class LoadedVariant:
    name: str
    model: object
    tokenizer: object | None
    backend: str


class ModelRegistry:
    def __init__(self) -> None:
        self._cache: dict[str, LoadedVariant] = {}

    def get(self, variant: str) -> LoadedVariant:
        if variant not in self._cache:
            self._cache[variant] = self._load(variant)
        return self._cache[variant]

    def _load(self, variant: str) -> LoadedVariant:
        if variant == "fp16":
            return self._load_transformers_variant(variant, load_in_8bit=False)
        if variant == "int8":
            return self._load_transformers_variant(variant, load_in_8bit=True)
        if variant == "int4_gguf":
            return self._load_gguf_variant()
        raise ValueError(f"Unsupported variant: {variant}")

    def _load_transformers_variant(self, variant: str, load_in_8bit: bool) -> LoadedVariant:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {
            "device_map": "auto",
        }
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO_ID or BASE_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
        if ADAPTER_REPO_ID:
            model = PeftModel.from_pretrained(model, ADAPTER_REPO_ID)

        return LoadedVariant(
            name=variant,
            model=model,
            tokenizer=tokenizer,
            backend="transformers",
        )

    def _load_gguf_variant(self) -> LoadedVariant:
        from llama_cpp import Llama

        if not GGUF_MODEL_FILE:
            raise ValueError("HF_GGUF_MODEL_FILE is not set for the int4_gguf variant.")

        model = Llama(
            model_path=GGUF_MODEL_FILE,
            n_ctx=2048,
            verbose=False,
        )
        return LoadedVariant(
            name="int4_gguf",
            model=model,
            tokenizer=None,
            backend="llama_cpp",
        )


def measure_peak_memory_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1_000_000_000
    except Exception:
        pass
    return 0.0


def time_call(fn, *args, **kwargs) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms

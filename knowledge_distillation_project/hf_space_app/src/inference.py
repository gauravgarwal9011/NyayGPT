from __future__ import annotations

from .config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, SYSTEM_PROMPT
from .loaders import ModelRegistry


class ChatService:
    def __init__(self) -> None:
        self.registry = ModelRegistry()

    def generate(
        self,
        prompt: str,
        variant: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        loaded = self.registry.get(variant)

        if loaded.backend == "transformers":
            return self._generate_transformers(
                loaded.model,
                loaded.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        if loaded.backend == "llama_cpp":
            return self._generate_gguf(
                loaded.model,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        raise ValueError(f"Unsupported backend: {loaded.backend}")

    def _generate_transformers(self, model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)

        prompt_tokens = inputs["input_ids"].shape[1]
        completion = outputs[0][prompt_tokens:]
        return tokenizer.decode(completion, skip_special_tokens=True).strip()

    def _generate_gguf(self, llm, prompt: str, max_new_tokens: int, temperature: float) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"].strip()

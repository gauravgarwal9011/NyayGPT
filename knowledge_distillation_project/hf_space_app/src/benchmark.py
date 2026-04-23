from __future__ import annotations

from statistics import mean

from .config import load_benchmark_prompts
from .inference import ChatService
from .loaders import measure_peak_memory_gb, time_call
from .metrics import RougeScorer


def _estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


class BenchmarkRunner:
    def __init__(self) -> None:
        self.chat = ChatService()
        self.rouge = RougeScorer()

    def run(self, limit: int = 3) -> list[dict]:
        prompts = load_benchmark_prompts(limit=limit)
        results = []

        for variant in ["fp16", "int8", "int4_gguf"]:
            predictions = []
            references = []
            latencies = []

            for row in prompts:
                response, elapsed_ms = time_call(
                    self.chat.generate,
                    row["prompt"],
                    variant,
                )
                predictions.append(response)
                references.append(row["reference"])
                latencies.append(elapsed_ms / _estimate_token_count(response))

            rouge_scores = self.rouge.score(predictions=predictions, references=references)
            results.append(
                {
                    "variant": variant,
                    "memory_gb": measure_peak_memory_gb(),
                    "ms_per_token": mean(latencies),
                    "rouge1": rouge_scores["rouge1"],
                    "rouge2": rouge_scores["rouge2"],
                    "rougeL": rouge_scores["rougeL"],
                }
            )

        return results

from __future__ import annotations

import evaluate


class RougeScorer:
    def __init__(self) -> None:
        self.metric = evaluate.load("rouge")

    def score(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        results = self.metric.compute(predictions=predictions, references=references)
        return {
            "rouge1": float(results["rouge1"]),
            "rouge2": float(results["rouge2"]),
            "rougeL": float(results["rougeL"]),
        }

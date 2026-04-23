"""
exceptions.py — Custom exception hierarchy for NyayaGPT pipeline.
"""


class NyayaError(Exception):
    """Base exception for all NyayaGPT pipeline errors."""


class ConfigurationError(NyayaError):
    """Invalid or missing configuration value."""


class ScrapingError(NyayaError):
    """IndianKanoon scraping or parsing failure."""


class GenerationError(NyayaError):
    """Synthetic Q&A generation failure (Azure OpenAI or GGUF)."""


class QualityFilterError(NyayaError):
    """Unexpected error inside the quality filter logic."""


class DatasetError(NyayaError):
    """Dataset load, save, or validation failure."""


class TrainingError(NyayaError):
    """QLoRA training failure."""


class EvaluationError(NyayaError):
    """ROUGE / RAGAS evaluation failure."""


class BenchmarkError(NyayaError):
    """Quantization benchmark failure."""


class InferenceError(NyayaError):
    """Model inference failure."""

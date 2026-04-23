"""
nyaya_pipeline — NyayaGPT: Indian Legal LLM fine-tuning pipeline.

Modules:
  config              — centralised constants (env-var overridable)
  exceptions          — custom exception hierarchy
  logger              — rotating file + console logging factory
  data_collector      — IndianKanoon scraper
  synthetic_generator — Azure OpenAI + GGUF fallback Q&A generation
  quality_filter      — legal-domain quality gate
  trainer             — QLoRA fine-tuning with MLflow logging
  evaluator           — ROUGE + RAGAS evaluation
  benchmark           — FP16 / INT8 / INT4-GGUF quantization benchmark
  infer               — inference + A/B routing
  dashboard           — Streamlit A/B test UI
"""

__version__ = "1.0.0"
__author__  = "NyayaGPT Team"

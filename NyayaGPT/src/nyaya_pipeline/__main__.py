"""
__main__.py — Entry point: python -m nyaya_pipeline

Runs the full pipeline: collect → generate → train → evaluate.
"""
import argparse
import sys
from pathlib import Path

from . import config
from .logger import get_logger

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nyaya_pipeline",
        description="NyayaGPT full pipeline: data → train → evaluate",
    )
    p.add_argument("--collect",    action="store_true", help="Run IndianKanoon scraper")
    p.add_argument("--generate",   action="store_true", help="Generate synthetic Q&A")
    p.add_argument("--train",      action="store_true", help="Run QLoRA training")
    p.add_argument("--evaluate",   action="store_true", help="Run ROUGE+RAGAS evaluation")
    p.add_argument("--all",        action="store_true", help="Run all stages")
    p.add_argument("--model",      default=config.STUDENT_MODEL_NAME)
    p.add_argument("--adapter-dir", default=str(config.ADAPTER_DIR))
    return p


def main() -> int:
    config.ensure_directories()
    args = _build_parser().parse_args()
    run_all = args.all

    if args.collect or run_all:
        from .data_collector import collect_judgments
        log.info("=== Stage 1: Collecting judgments ===")
        collect_judgments()

    if args.generate or run_all:
        import json
        from .data_collector import collect_judgments
        from .synthetic_generator import generate_qa_pairs, save_datasets

        log.info("=== Stage 2: Generating Q&A pairs ===")
        raw_path = config.OUTPUT_DIR / "raw_judgments.json"
        if not raw_path.exists():
            log.info("raw_judgments.json not found — collecting first …")
            collect_judgments()
        judgments = json.loads(raw_path.read_text(encoding="utf-8"))
        all_chunks = [c for j in judgments for c in j["chunks"]]
        alpaca, chat = generate_qa_pairs(all_chunks)
        save_datasets(alpaca, chat)

    if args.train or run_all:
        from .trainer import train
        log.info("=== Stage 3: QLoRA training ===")
        train(model_name=args.model, adapter_dir=Path(args.adapter_dir))

    if args.evaluate or run_all:
        from .evaluator import run_evaluation
        log.info("=== Stage 4: Evaluation ===")
        results = run_evaluation(adapter_dir=Path(args.adapter_dir))
        for label, metrics in results.items():
            log.info("%s: %s", label, metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
evaluator.py — ROUGE + RAGAS evaluation for NyayaGPT.

Compares fine-tuned NyayaGPT against base Mistral on the eval set.
All results logged to MLflow under the nyayagpt-evaluation experiment.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from . import config
from .exceptions import EvaluationError, ConfigurationError
from .logger import get_logger

log = get_logger(__name__)


def _load_eval_pairs(eval_path: Path, n: int = 50) -> List[Tuple[str, str]]:
    """Return (question, reference_answer) from eval.jsonl."""
    pairs = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            msgs = obj["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            pairs.append((user, asst))
            if len(pairs) >= n:
                break
    return pairs


def _load_model_for_eval(model_name: str, adapter_dir: Optional[Path] = None):
    """Load a model for inference (with optional adapter)."""
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as exc:
        raise EvaluationError("unsloth not installed") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.STUDENT_MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_dir and adapter_dir.exists():
        model.load_adapter(str(adapter_dir))
        log.info("Loaded adapter from %s", adapter_dir)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate_answer(question: str, model, tokenizer, max_new_tokens: int = 200) -> str:
    import torch
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 means."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise EvaluationError("rouge-score not installed. pip install rouge-score") from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = len(predictions)
    return {k: round(v / n, 4) for k, v in totals.items()}


def _build_ragas_judge_llm():
    """
    Return a RAGAS-compatible judge LLM. Prefer Azure OpenAI (uses existing
    AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT from the data-generation step);
    fall back to vanilla OpenAI if OPENAI_API_KEY is set; otherwise return
    None and let RAGAS skip with a clear error.
    """
    import os
    azure_key      = os.getenv("AZURE_OPENAI_KEY")       or config.AZURE_OPENAI_KEY
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  or config.AZURE_OPENAI_ENDPOINT
    # RAGAS judge can use a *different* (cheaper) deployment than data generation.
    judge_deploy   = os.getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT", "gpt-4o-mini")
    embed_deploy   = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

    if azure_key and azure_endpoint:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        log.info("RAGAS judge: Azure deployment=%s  embeddings=%s", judge_deploy, embed_deploy)

        llm = LangchainLLMWrapper(AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version="2024-08-01-preview",
            azure_deployment=judge_deploy,
            temperature=0,
        ))
        try:
            embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version="2024-08-01-preview",
                azure_deployment=embed_deploy,
            ))
        except Exception as exc:
            log.warning("Azure embeddings unavailable (%s) — answer_relevancy will be skipped", exc)
            embeddings = None
        return llm, embeddings

    if os.getenv("OPENAI_API_KEY"):
        return None, None   # RAGAS will auto-construct its OpenAI default

    raise EvaluationError(
        "No LLM credentials for RAGAS. Set AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT "
        "(recommended — same credentials used for data generation) or OPENAI_API_KEY."
    )


def compute_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
) -> Dict[str, float]:
    """Compute RAGAS faithfulness + answer relevancy."""
    try:
        from datasets import Dataset as HFDataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    except ImportError as exc:
        raise EvaluationError("ragas or datasets not installed. pip install ragas datasets") from exc

    llm, embeddings = _build_ragas_judge_llm()

    data = {
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    }
    ds = HFDataset.from_dict(data)

    metrics = [faithfulness]
    if embeddings is not None:
        metrics.append(answer_relevancy)

    kwargs = {}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    result = evaluate(ds, metrics=metrics, **kwargs)

    def _mean(v):
        """RAGAS 0.4.x returns per-sample lists; older versions return a scalar.
        Also skip NaN entries from timed-out samples."""
        import math
        if isinstance(v, (list, tuple)):
            clean = [float(x) for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
            if not clean:
                return float("nan")
            return sum(clean) / len(clean)
        return float(v)

    def _safe_get(r, key):
        """RAGAS 0.4.x EvaluationResult's `in` operator is broken (raises KeyError: 0
        due to fallback iteration), so probe with try/except instead."""
        try:
            return r[key]
        except (KeyError, AttributeError):
            return None

    out = {}
    faith = _safe_get(result, "faithfulness")
    if faith is not None:
        out["ragas_faithfulness"] = round(_mean(faith), 4)

    relev = _safe_get(result, "answer_relevancy")
    if relev is not None:
        out["ragas_answer_relevancy"] = round(_mean(relev), 4)

    return out


def run_evaluation(
    fine_tuned_model: str = None,
    adapter_dir: Path = None,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    eval_path: Path = None,
    n_samples: int = 50,
    mlflow_run_id: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Evaluate base model vs fine-tuned NyayaGPT on ROUGE + RAGAS.

    Returns:
        {"base": {metrics}, "finetuned": {metrics}, "delta": {metrics}}
    """
    import mlflow, gc, torch

    fine_tuned_model = fine_tuned_model or config.STUDENT_MODEL_NAME
    adapter_dir      = adapter_dir      or config.ADAPTER_DIR
    eval_path        = eval_path        or config.EVAL_JSONL

    if not eval_path.exists():
        raise ConfigurationError(f"eval.jsonl not found: {eval_path}")

    pairs     = _load_eval_pairs(eval_path, n=n_samples)
    questions = [q for q, _ in pairs]
    references= [r for _, r in pairs]
    contexts  = [[q] for q in questions]  # Use question as minimal context for RAGAS

    results = {}

    for label, model_name, use_adapter in [
        ("base",      base_model,         False),
        ("finetuned", fine_tuned_model,   True),
    ]:
        log.info("Evaluating: %s …", label)
        model, tokenizer = _load_model_for_eval(
            model_name, adapter_dir if use_adapter else None
        )

        t0 = time.time()
        preds = [_generate_answer(q, model, tokenizer) for q in questions]
        elapsed = time.time() - t0

        rouge_scores = compute_rouge(preds, references)
        try:
            ragas_scores = compute_ragas(questions, preds, contexts)
        except EvaluationError as exc:
            log.warning("RAGAS skipped: %s", exc)
            ragas_scores = {}

        results[label] = {
            **rouge_scores,
            **ragas_scores,
            "inference_time_secs": round(elapsed, 1),
            "ms_per_sample":       round(elapsed * 1000 / len(questions), 1),
        }
        log.info("  %s ROUGE-L=%.4f", label, rouge_scores["rougeL"])

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Delta (improvement)
    results["delta"] = {
        k: round(results["finetuned"].get(k, 0) - results["base"].get(k, 0), 4)
        for k in results["base"]
        if isinstance(results["base"][k], float)
    }

    # Log to MLflow
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_EVAL)
    with mlflow.start_run(run_name="evaluation", nested=bool(mlflow_run_id)):
        for label, metrics in results.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{label}_{k}", v)
        log.info("Evaluation metrics logged to MLflow")

    return results

"""
_bench_gguf_worker.py — Isolated GGUF benchmark worker.

Runs in a fresh Python process (no torch CUDA context) so llama-cpp-python
can claim GPU memory cleanly. Reads a spec JSON, writes a result JSON.

Invoked by benchmark.py::bench_int4_gguf via subprocess.
"""
import json
import sys
import time
from pathlib import Path


def main(spec_path: str, result_path: str) -> int:
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    name           = spec.get("name", "GGUF")
    gguf_path      = Path(spec["gguf_path"])
    samples        = spec["samples"]
    system_prompt  = spec["system_prompt"]
    max_new_tokens = spec.get("max_new_tokens", 100)

    from llama_cpp import Llama
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )

    preds, latencies = [], []
    for question, _ in samples:
        prompt = f"[INST] {system_prompt}\n\n{question}[/INST]"
        t0  = time.perf_counter()
        out = llm(prompt, max_tokens=max_new_tokens, temperature=0.0)
        elapsed = time.perf_counter() - t0
        text   = out["choices"][0]["text"].strip()
        n_new  = out["usage"]["completion_tokens"]
        if n_new > 0:
            latencies.append(elapsed * 1000 / n_new)
        preds.append(text)

    # llama-cpp + GGUF memory footprint is dominated by file size, not runtime VRAM,
    # and nvidia-smi isn't available in WSL — use file size as memory proxy.
    memory_gb = round(gguf_path.stat().st_size / 1e9, 2)
    latency   = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for pred, (_, ref) in zip(preds, samples)]
        rouge_l = round(sum(scores) / len(scores), 4) if scores else 0.0
    except ImportError:
        rouge_l = -1.0

    result = {
        "name":                  name,
        "memory_gb":             memory_gb,
        "latency_ms_per_token":  latency,
        "rouge_l":               rouge_l,
        "num_samples":           len(samples),
        "error":                 None,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: _bench_gguf_worker.py <spec.json> <result.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

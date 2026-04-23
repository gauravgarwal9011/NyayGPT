"""
benchmark.py — Quantization benchmark via GGUF: FP16 vs INT8 vs INT4.

Measures per variant:
  - Peak memory proxy (GGUF file size in GB)
  - Latency (ms per generated token)
  - ROUGE-L F1 against eval.jsonl

On this Blackwell + CUDA 12.8 setup, the original HF/Unsloth FP16 and
bitsandbytes INT8 paths are not stable due to cuBLAS and patched-attention
issues documented in document.md. To keep the benchmark working end-to-end,
all three variants are run through the same llama.cpp subprocess worker.
"""
import gc
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import torch

from . import config
from .exceptions import BenchmarkError
from .logger import get_logger

log = get_logger(__name__)

GGUF_SCRATCH_DIR = Path(os.environ.get("NY_GGUF_SCRATCH_DIR", "/mnt/f/NyayaGPT-scratch"))
FP16_GGUF_PATH = Path(os.environ.get("NY_FP16_GGUF_PATH", GGUF_SCRATCH_DIR / "nyayagpt-fp16.gguf"))
INT8_GGUF_PATH = Path(os.environ.get("NY_INT8_GGUF_PATH", GGUF_SCRATCH_DIR / "nyayagpt-q8_0.gguf"))
LLAMA_QUANTIZE = Path(
    os.environ.get(
        "NY_LLAMA_QUANTIZE",
        Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-quantize",
    )
)


@dataclass
class BenchmarkResult:
    name: str
    memory_gb: float
    latency_ms_per_token: float
    rouge_l: float
    num_samples: int
    error: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_eval_samples(n: int) -> List[Tuple[str, str]]:
    pairs = []
    with open(config.EVAL_JSONL, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            msgs = obj["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            pairs.append((user, asst))
            if len(pairs) >= n:
                break
    return pairs


def _reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def _compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for pred, ref in zip(predictions, references)]
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    except ImportError:
        log.warning("rouge-score not installed — ROUGE-L skipped")
        return -1.0


def _run_gguf_worker(name: str, gguf_path: Path, samples: List[Tuple[str, str]]) -> BenchmarkResult:
    """Run the GGUF benchmark in an isolated subprocess.

    llama-cpp-python and torch's CUDA caching allocator can't safely share a
    process on WSL2 + Blackwell — the second allocator to reach the GPU
    segfaults the kernel. Running this in a fresh Python process avoids the
    conflict entirely.
    """
    log.info("[%s] Loading %s (isolated subprocess) …", name, gguf_path.name)
    _reset_gpu()

    if not gguf_path.exists():
        return BenchmarkResult(
            name, 0, 0, 0, 0,
            error=f"GGUF not found: {gguf_path}. See notebook 04 for conversion steps.",
        )

    import sys
    import tempfile
    worker = Path(__file__).resolve().parents[2] / "scripts" / "_bench_gguf_worker.py"
    if not worker.exists():
        return BenchmarkResult(name, 0, 0, 0, 0, error=f"worker not found: {worker}")

    spec = {
        "name":           name,
        "gguf_path":      str(gguf_path),
        "samples":        [[q, r] for q, r in samples],
        "system_prompt":  config.SYSTEM_PROMPT,
        "max_new_tokens": 100,
    }
    spec_f   = tempfile.NamedTemporaryFile(mode="w", suffix="-spec.json",   delete=False, encoding="utf-8")
    result_f = tempfile.NamedTemporaryFile(mode="w", suffix="-result.json", delete=False, encoding="utf-8")
    spec_path, result_path = spec_f.name, result_f.name
    json.dump(spec, spec_f); spec_f.close(); result_f.close()

    try:
        proc = subprocess.run(
            [sys.executable, str(worker), spec_path, result_path],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            return BenchmarkResult(
                name, 0, 0, 0, 0,
                error=f"worker exited {proc.returncode}: {' | '.join(tail)[:200]}",
            )
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
        return BenchmarkResult(**data)
    except subprocess.TimeoutExpired:
        return BenchmarkResult(name, 0, 0, 0, 0, error="worker timed out (>600s)")
    except Exception as exc:
        return BenchmarkResult(name, 0, 0, 0, 0, error=str(exc))
    finally:
        for p in (spec_path, result_path):
            try: os.unlink(p)
            except OSError: pass


def _ensure_int8_gguf(fp16_gguf_path: Path, int8_gguf_path: Path) -> Optional[str]:
    if int8_gguf_path.exists():
        return None
    if not fp16_gguf_path.exists():
        return (
            f"INT8 GGUF not found: {int8_gguf_path}. "
            f"FP16 GGUF source also missing: {fp16_gguf_path}."
        )
    if not LLAMA_QUANTIZE.exists():
        return f"llama-quantize not found: {LLAMA_QUANTIZE}"

    int8_gguf_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "[INT8] Q8_0 GGUF missing; quantizing %s -> %s via llama-quantize …",
        fp16_gguf_path.name,
        int8_gguf_path.name,
    )
    try:
        subprocess.run(
            [str(LLAMA_QUANTIZE), str(fp16_gguf_path), str(int8_gguf_path), "Q8_0"],
            check=True,
            timeout=1800,
        )
        return None
    except subprocess.TimeoutExpired:
        return "llama-quantize timed out while building Q8_0 GGUF (>1800s)"
    except Exception as exc:
        return f"failed to build Q8_0 GGUF: {exc}"


# ── FP16 / INT8 / INT4 benchmarks via GGUF ───────────────────────────────────

def bench_fp16(samples: List[Tuple[str, str]], gguf_path: Path) -> BenchmarkResult:
    return _run_gguf_worker("FP16", gguf_path, samples)


def bench_int8(
    samples: List[Tuple[str, str]],
    fp16_gguf_path: Path,
    int8_gguf_path: Path,
) -> BenchmarkResult:
    err = _ensure_int8_gguf(fp16_gguf_path, int8_gguf_path)
    if err:
        return BenchmarkResult("INT8", 0, 0, 0, 0, error=err)
    return _run_gguf_worker("INT8", int8_gguf_path, samples)


def bench_int4_gguf(samples: List[Tuple[str, str]], gguf_path: Path) -> BenchmarkResult:
    return _run_gguf_worker("INT4 (GGUF)", gguf_path, samples)


# ── Run all + report ─────────────────────────────────────────────────────────

def run_benchmark(
    n_samples: int = 15,
    gguf_path: Optional[Path] = None,
    adapter_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    fp16_gguf_path: Optional[Path] = None,
    int8_gguf_path: Optional[Path] = None,
) -> List[BenchmarkResult]:
    """Run FP16 / INT8 / INT4 GGUF benchmarks and return results."""
    adapter_dir = adapter_dir or config.ADAPTER_DIR
    output_path = output_path or (config.OUTPUT_DIR / "benchmark_results.json")
    fp16_gguf_path = fp16_gguf_path or FP16_GGUF_PATH
    int8_gguf_path = int8_gguf_path or INT8_GGUF_PATH

    samples = _load_eval_samples(n_samples)
    log.info("Running benchmark on %d eval samples …", len(samples))

    results = [
        bench_fp16(samples, fp16_gguf_path),
        bench_int8(samples, fp16_gguf_path, int8_gguf_path),
    ]

    gguf_path = gguf_path or (config.ADAPTER_DIR / "nyayagpt-q4km.gguf")
    results.append(bench_int4_gguf(samples, gguf_path))

    # Print table
    print("\n" + "=" * 68)
    print(f"{'Model':<16} {'Memory (GB)':>12} {'Latency (ms/tok)':>18} {'ROUGE-L':>10}")
    print("-" * 68)
    for r in results:
        if r.error:
            print(f"{r.name:<16} {'ERROR':>12}   {r.error[:38]}")
        else:
            print(f"{r.name:<16} {r.memory_gb:>12.2f} {r.latency_ms_per_token:>18.1f} {r.rouge_l:>10.4f}")
    print("=" * 68)

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    log.info("Benchmark results saved → %s", output_path)

    return results


def plot_benchmark(results: List[BenchmarkResult], save_path: Optional[Path] = None):
    """Render 3-panel bar chart and optionally save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib not available — skipping plot")
        return None

    valid  = [r for r in results if not r.error]
    names  = [r.name for r in valid]
    colors = ["#4e79a7", "#f28e2b", "#e15759"][:len(valid)]
    x      = np.arange(len(names))
    w      = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("NyayaGPT — Quantization Benchmark (Mistral 7B)", fontsize=12)

    def _bar(ax, vals, title, ylabel):
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white")
        ax.set_title(title, pad=6)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel(ylabel)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                    f"{v:.2g}", ha="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _bar(axes[0], [r.memory_gb            for r in valid], "Memory (GB)",        "GB")
    _bar(axes[1], [r.latency_ms_per_token for r in valid], "Latency (ms/token)", "ms/tok")
    _bar(axes[2], [r.rouge_l              for r in valid], "Quality (ROUGE-L)",  "F1")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        log.info("Chart saved → %s", save_path)
    return fig

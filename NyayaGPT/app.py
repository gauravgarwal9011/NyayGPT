"""
app.py — NyayaGPT Gradio frontend for HF Spaces.

Two tabs:
  Chat       — Interactive legal Q&A with NyayaGPT
  Benchmark  — FP16 / INT8 / INT4 comparison charts

Run locally:
  python app.py            # http://localhost:7860
  python app.py --share    # public Gradio link
  python app.py --preload  # load model on startup

Deploy on HF Spaces:
  Place this file as app.py in a Gradio Space repo.
  Set HF_MODEL_REPO env var to your model repo id.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nyaya_pipeline import config
from nyaya_pipeline.logger import get_logger

log = get_logger(__name__)

BENCHMARK_CACHE = PROJECT_ROOT / "output" / "benchmark_results.json"

_DEFAULT_BENCHMARK = [
    {"name": "FP16",        "memory_gb": 14.2, "latency_ms_per_token": 22.1, "rouge_l": 0.431},
    {"name": "INT8",        "memory_gb":  7.8, "latency_ms_per_token": 28.4, "rouge_l": 0.419},
    {"name": "INT4 (GGUF)", "memory_gb":  4.6, "latency_ms_per_token": 15.3, "rouge_l": 0.402},
]

# ── Model (lazy load) ─────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    try:
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.getenv("HF_MODEL_NAME", config.STUDENT_MODEL_NAME),
            max_seq_length=config.STUDENT_MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )
        adapter_dir = os.getenv("HF_ADAPTER_DIR", str(config.ADAPTER_DIR))
        if Path(adapter_dir).exists():
            _model.load_adapter(adapter_dir)
        FastLanguageModel.for_inference(_model)
        log.info("Model loaded successfully")
    except Exception as exc:
        raise RuntimeError(f"Model load failed: {exc}") from exc

    return _model, _tokenizer


# ── Chat ─────────────────────────────────────────────────────────────────────

def _respond(message, history, system_prompt, max_tokens, temperature):
    if not message.strip():
        return history, ""
    try:
        model, tokenizer = _load_model()
        msgs = [{"role": "system", "content": system_prompt}]
        for user_msg, asst_msg in history:
            msgs += [{"role": "user", "content": user_msg},
                     {"role": "assistant", "content": asst_msg}]
        msgs.append({"role": "user", "content": message})

        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids    = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_in   = ids["input_ids"].shape[1]

        gen_kw = {"max_new_tokens": int(max_tokens), "do_sample": temperature > 0,
                  "use_cache": True}
        if temperature > 0:
            gen_kw["temperature"] = float(temperature)

        with torch.no_grad():
            out = model.generate(**ids, **gen_kw)

        response = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
    except Exception as exc:
        response = f"[Error] {exc}"

    history.append((message, response))
    return history, ""


# ── Benchmark ────────────────────────────────────────────────────────────────

def _load_benchmark():
    if BENCHMARK_CACHE.exists():
        try:
            return json.loads(BENCHMARK_CACHE.read_text())
        except Exception:
            pass
    return _DEFAULT_BENCHMARK


def _make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid  = [r for r in results if not r.get("error")]
    names  = [r["name"] for r in valid]
    colors = ["#4e79a7", "#f28e2b", "#e15759"][:len(valid)]
    x      = np.arange(len(names))
    w      = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.suptitle("NyayaGPT — Quantization Benchmark (Mistral 7B)", fontsize=11)

    def _bar(ax, vals, title, ylabel):
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white")
        ax.set_title(title, pad=5)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.03,
                    f"{v:.2g}", ha="center", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _bar(axes[0], [r["memory_gb"]            for r in valid], "Memory (GB)",       "GB")
    _bar(axes[1], [r["latency_ms_per_token"]  for r in valid], "Latency (ms/token)","ms/tok")
    _bar(axes[2], [r["rouge_l"]               for r in valid], "Quality (ROUGE-L)",  "F1")
    axes[2].set_ylim(0, 1)
    plt.tight_layout()
    return fig


def _make_table(results):
    return [[r["name"],
             f"{r['memory_gb']:.1f} GB",
             f"{r['latency_ms_per_token']:.1f} ms/tok",
             f"{r['rouge_l']:.4f}",
             r.get("error", "")] for r in results]


def _run_benchmark(n_samples, gguf_path):
    import subprocess
    yield "Running benchmark… (this may take several minutes)", None, None

    cmd = [sys.executable,
           str(PROJECT_ROOT / "src" / "nyaya_pipeline" / "benchmark.py"),
           "--n-samples", str(int(n_samples)),
           "--output", str(BENCHMARK_CACHE.with_suffix(""))]
    if gguf_path.strip():
        cmd += ["--gguf-path", gguf_path.strip()]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        yield f"Benchmark failed:\n{proc.stderr[-500:]}", None, None
        return

    results = _load_benchmark()
    yield "Benchmark complete.", _make_plots(results), _make_table(results)


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="NyayaGPT — Indian Legal Assistant",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.Markdown(
            "# ⚖️ NyayaGPT — Indian Legal Assistant\n"
            "Fine-tuned Mistral-7B on IndianKanoon case law via QLoRA. "
            "**For educational use only — not legal advice.**"
        )

        with gr.Tabs():

            # ── Chat tab ──────────────────────────────────────────────────────
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="NyayaGPT", height=430,
                                     bubble_full_width=False, show_copy_button=True)

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="e.g. What is the punishment under Section 302 IPC?",
                        label="Your question", scale=5, lines=2,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=70)

                with gr.Accordion("Settings", open=False):
                    sys_prompt = gr.Textbox(value=config.SYSTEM_PROMPT,
                                            label="System prompt", lines=4)
                    with gr.Row():
                        max_tok = gr.Slider(64, 512, 256, step=32, label="Max tokens")
                        temp    = gr.Slider(0.0, 1.0, 0.1, step=0.05, label="Temperature")

                clear = gr.Button("Clear", variant="secondary")

                send.click(_respond,
                           inputs=[msg, chatbot, sys_prompt, max_tok, temp],
                           outputs=[chatbot, msg])
                msg.submit(_respond,
                           inputs=[msg, chatbot, sys_prompt, max_tok, temp],
                           outputs=[chatbot, msg])
                clear.click(lambda: ([], ""), outputs=[chatbot, msg])

                gr.Examples(
                    examples=[
                        ["What is Section 302 of the IPC?"],
                        ["Explain Article 21 of the Constitution."],
                        ["What constitutes cheque dishonour under Section 138 NI Act?"],
                        ["What is a PIL and who can file one?"],
                        ["Explain anticipatory bail under the CrPC."],
                    ],
                    inputs=msg,
                    label="Example questions",
                )

            # ── Benchmark tab ─────────────────────────────────────────────────
            with gr.Tab("Quantization Benchmark"):
                gr.Markdown(
                    "## FP16 → INT8 → INT4 (GGUF)\n"
                    "Memory (GB) · Latency (ms/token) · ROUGE-L quality"
                )
                with gr.Row():
                    n_slider  = gr.Slider(5, 32, 15, step=1, label="Eval samples", scale=2)
                    gguf_box  = gr.Textbox(label="GGUF path (optional)", scale=3,
                                           placeholder="adapters/nyayagpt-q4km.gguf")
                    run_btn   = gr.Button("Run Benchmark", variant="primary", scale=1)

                status  = gr.Textbox(label="Status", interactive=False, lines=1,
                                     value="Showing cached/default values.")
                bplot   = gr.Plot(label="Benchmark Charts")
                btable  = gr.Dataframe(
                    headers=["Model", "Memory", "Latency", "ROUGE-L", "Error"],
                    label="Results Table", interactive=False,
                )

                def _on_load():
                    r = _load_benchmark()
                    src = "cached" if BENCHMARK_CACHE.exists() else "placeholder"
                    return f"Showing {src} results.", _make_plots(r), _make_table(r)

                demo.load(_on_load, outputs=[status, bplot, btable])
                run_btn.click(_run_benchmark, inputs=[n_slider, gguf_box],
                              outputs=[status, bplot, btable])

                with gr.Accordion("GGUF conversion steps", open=False):
                    gr.Markdown(
                        "```bash\n"
                        "# 1. Merge adapter\n"
                        "python deploy_to_hub.py --repo-id dummy --merge\n\n"
                        "# 2. Clone + build llama.cpp\n"
                        "git clone https://github.com/ggerganov/llama.cpp\n"
                        "cd llama.cpp && make -j GGML_CUDA=1\n\n"
                        "# 3. Convert to GGUF\n"
                        "python llama.cpp/convert_hf_to_gguf.py merged_model/ --outtype f16\n\n"
                        "# 4. Quantize\n"
                        "./llama.cpp/llama-quantize merged_model/ggml-model-f16.gguf \\\n"
                        "    adapters/nyayagpt-q4km.gguf Q4_K_M\n"
                        "```"
                    )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--share",   action="store_true")
    p.add_argument("--port",    type=int, default=7860)
    p.add_argument("--preload", action="store_true")
    return p


def main():
    args = _parser().parse_args()

    if args.preload and config.ADAPTER_DIR.exists():
        print("Pre-loading model …")
        try:
            _load_model()
        except Exception as e:
            print(f"[warn] Pre-load failed: {e}")

    build_ui().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()

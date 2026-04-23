import gradio as gr
import pandas as pd

from src.benchmark import BenchmarkRunner
from src.config import APP_TITLE, APP_DESCRIPTION, DEFAULT_BENCHMARK_LIMIT
from src.inference import ChatService


chat_service = ChatService()
benchmark_runner = BenchmarkRunner()


def chat_fn(message: str, history: list, variant: str, max_new_tokens: int, temperature: float):
    response = chat_service.generate(
        prompt=message,
        variant=variant,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response


def benchmark_fn(limit: int):
    results = benchmark_runner.run(limit=limit)
    df = pd.DataFrame(results)
    if not df.empty:
        numeric_cols = ["memory_gb", "ms_per_token", "rouge1", "rouge2", "rougeL"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda value: round(float(value), 4))
    return df


with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)

    with gr.Tab("Chat"):
        variant = gr.Dropdown(
            choices=["fp16", "int8", "int4_gguf"],
            value="fp16",
            label="Model Variant",
        )
        max_new_tokens = gr.Slider(
            minimum=32,
            maximum=512,
            step=32,
            value=256,
            label="Max New Tokens",
        )
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            value=0.2,
            label="Temperature",
        )
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[variant, max_new_tokens, temperature],
            type="messages",
            title="Ignatiuz Student Chat",
            description="Ask your fine-tuned model questions and switch between precision variants.",
        )

    with gr.Tab("Benchmark"):
        gr.Markdown(
            "Run a lightweight benchmark over held-out prompts and compare memory, "
            "latency, and ROUGE across FP16, INT8, and INT4 GGUF."
        )
        benchmark_limit = gr.Slider(
            minimum=1,
            maximum=20,
            step=1,
            value=DEFAULT_BENCHMARK_LIMIT,
            label="Number of Benchmark Prompts",
        )
        run_button = gr.Button("Run Benchmark", variant="primary")
        results_table = gr.Dataframe(
            headers=["variant", "memory_gb", "ms_per_token", "rouge1", "rouge2", "rougeL"],
            label="Benchmark Results",
            interactive=False,
        )
        run_button.click(fn=benchmark_fn, inputs=benchmark_limit, outputs=results_table)


if __name__ == "__main__":
    demo.launch()

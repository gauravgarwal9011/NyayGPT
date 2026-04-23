# Knowledge Distillation Pipeline

A modular Python pipeline that converts a source PDF + a verified knowledge
base into a high-quality JSONL training dataset for fine-tuning a smaller
"student" LLM. A local GGUF "teacher" model (gpt-oss-20b F16) generates the
Q&A pairs, which are then quality-filtered before being written to disk.

This is the modular refactor of the original `Knowledge_distillation.ipynb`
notebook — every cell of the notebook now lives in its own focused module
with logging, typed exceptions, and line-by-line explanatory comments.

---

## Project layout

```
knowledge_distillation_project/
├── README.md
├── requirements.txt
├── logs/                       # rotating log files (auto-created)
├── output/                     # generated chunks + train/eval JSONL
└── src/
    └── kd_pipeline/
        ├── __init__.py         # package marker + version
        ├── __main__.py         # `python -m kd_pipeline` entry point
        ├── config.py           # constants, paths, env-var overrides
        ├── logger.py           # rotating-file + console logger factory
        ├── exceptions.py       # custom exception hierarchy
        ├── knowledge_base.py   # verified DOCUMENT_KNOWLEDGE_BASE dict
        ├── pdf_extractor.py    # fitz + pdfplumber wrappers
        ├── text_cleaner.py     # regex-based text normalisation
        ├── chunker.py          # sliding-window chunkers (PDF + KB)
        ├── prompt_templates.py # direct + CoT question templates
        ├── teacher_model.py    # GGUF wrapper class (load + generate)
        ├── quality_filter.py   # response QA gate
        ├── dataset_generator.py# top-level pipeline orchestrator
        └── dataset_auditor.py  # interactive sample reviewer
```

### Why this structure?

- **One responsibility per file.** Easy to test, easy to swap out a stage.
- **`config.py` is the only place hard-coded values live.** Everything is
  overridable via environment variables.
- **Custom exceptions** (in `exceptions.py`) replace bare `raise` statements
  so callers can catch *what* went wrong, not just *that* something did.
- **`logger.py`** gives every module its own named logger that writes to
  both console (INFO+) and a rotating file (DEBUG+) under `logs/`.

---

## Pipeline stages

```
┌──────────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│ PDF + verified   │ → │ chunker  │ → │ teacher Q&A  │ → │ quality      │ → │ JSONL    │
│ knowledge base   │   │ (overlap)│   │ (direct+CoT) │   │ filter       │   │ train/   │
└──────────────────┘   └──────────┘   └──────────────┘   └──────────────┘   │ eval     │
                                                                            └──────────┘
```

1. **Build chunks** — KB sections (verified, primary) + fitz PDF pages
   (secondary, recoverable if PDF is missing).
2. **Load teacher** — wraps `llama-cpp-python` via the `TeacherModel` class.
3. **Generate** — for each chunk: `QA_PER_CHUNK` direct questions + CoT for
   every other chunk (~30 % CoT ratio).
4. **Filter** — `quality_filter.is_quality_response` drops refusals, too-short
   outputs, low-overlap hallucinations, and suspect numeric patterns.
5. **Save** — strip `_meta`, shuffle with a fixed seed, 90/10 split, write
   `output/train.jsonl` and `output/eval.jsonl`.

---

## Installation

```bash
# 1. Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python deps
pip install -r requirements.txt

# 3. Build llama-cpp-python with CUDA (Blackwell / sm_120)
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
CUDAARCHS="120" \
pip install --no-binary :all: --force-reinstall llama-cpp-python==0.3.17
```

---

## Configuration

All settings in [src/kd_pipeline/config.py](src/kd_pipeline/config.py) can
be overridden via environment variables:

| Env var          | Default                                  | Purpose                              |
|------------------|------------------------------------------|--------------------------------------|
| `KD_PDF_PATH`    | `Ignatiuz_Capabilities.pdf`              | Source PDF                           |
| `KD_MODEL_PATH`  | path to gpt-oss-20b-F16.gguf             | Teacher GGUF model                   |
| `KD_OUTPUT_DIR`  | `output/`                                | Where train/eval JSONL is written    |
| `KD_LOG_DIR`     | `logs/`                                  | Where rotating log files live        |
| `KD_N_GPU_LAYERS`| `-1` (all on GPU)                        | Override for low-VRAM machines       |
| `KD_N_CTX`       | `2048`                                   | llama.cpp context window             |

---

## Running

### Dataset generation

```bash
# Run the full pipeline (generation only)
python -m kd_pipeline

# Generate AND interactively audit 10 random samples afterwards
python -m kd_pipeline --audit

# Audit only — skip generation, review an existing dataset
python -m kd_pipeline --no-generate --audit --audit-samples 20

# Use a different PDF
python -m kd_pipeline --pdf /path/to/other.pdf
```

### Student fine-tuning (Unsloth + LoRA)

```bash
# LoRA-fine-tune the default student (Qwen2.5-3B-Instruct in 4-bit)
# on the train.jsonl produced by the previous step.
python -m kd_pipeline.train

# Use a different student model
python -m kd_pipeline.train --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit

# Custom dataset paths
python -m kd_pipeline.train --train output/train.jsonl --eval output/eval.jsonl

# Custom adapter output directory
python -m kd_pipeline.train --adapter-dir adapters/qwen-run01
```

Trained adapters land in [adapters/](adapters/) (configurable via
`KD_ADAPTER_DIR` or `--adapter-dir`). The artifact is the LoRA adapter
only — load it at inference time with the same base model.

Training hyperparameters are all in [config.py](src/kd_pipeline/config.py)
and overridable via env vars (`KD_LORA_R`, `KD_LEARNING_RATE`,
`KD_NUM_EPOCHS`, `KD_TRAIN_BATCH`, `KD_GRAD_ACCUM`, …).

Exit codes:

| Code | Meaning                                                       |
|------|---------------------------------------------------------------|
| `0`  | Success                                                       |
| `1`  | Controlled pipeline failure (missing PDF, model load error…)  |
| `2`  | Unexpected exception — see traceback in `logs/`               |

---

## Outputs

After a successful run, `output/` contains:

| File                      | Description                                          |
|---------------------------|------------------------------------------------------|
| `train.jsonl`             | 90 % of samples, OpenAI chat format                  |
| `eval.jsonl`              | 10 % held-out                                        |
| `dataset_with_meta.json`  | Un-shuffled, with `_meta` audit fields               |
| `all_chunks.json`         | Combined KB + fitz chunks (debug)                    |
| `extracted_pages.json`    | Raw fitz page extraction (debug)                     |
| `fitz_chunks.json`        | Chunks built from PDF pages (debug)                  |

---

## Logging

Each module logs to a single rotating handler at `logs/kd_pipeline.log`
(10 MB × 5 backups). Console output is INFO+; the file captures DEBUG+ so
you can post-mortem a failed run without re-running.

---

## Custom exceptions

All defined in [src/kd_pipeline/exceptions.py](src/kd_pipeline/exceptions.py):

```
KnowledgeDistillationError          # base class — catch this for "any pipeline error"
├── ConfigurationError
├── PDFExtractionError
├── ChunkingError
├── ModelLoadError
├── GenerationError
├── QualityFilterError
└── DatasetSaveError
```

The CLI entry point catches `KnowledgeDistillationError` for graceful exits
and lets anything else bubble up as a bug-report-worthy traceback.

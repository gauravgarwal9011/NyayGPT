"""
exceptions.py
=============
Custom exception hierarchy for the knowledge-distillation pipeline.

WHY define custom exceptions?
-----------------------------
Catching `Exception` (or worse: bare `except:`) hides bugs and treats
everything from typos to disk-full errors the same way. Custom exceptions
let calling code do precise error handling like:

    try:
        teacher = TeacherModel.load()
    except ModelLoadError as e:
        # only catch the specific failure mode we know how to recover from
        log.error("Falling back to CPU mode: %s", e)

A well-named exception is also self-documenting — you can tell from the
class name *what* went wrong without reading the message.

The hierarchy:

    KnowledgeDistillationError                ← root, catch-all
        ├── ConfigurationError                ← bad config / missing files
        ├── PDFExtractionError                ← PyMuPDF / pdfplumber failures
        ├── ChunkingError                     ← bad chunker input
        ├── ModelLoadError                    ← teacher GGUF load failure
        ├── GenerationError                   ← inference call failure
        ├── QualityFilterError                ← bad quality-filter input
        └── DatasetSaveError                  ← I/O failures while writing JSONL
"""


class KnowledgeDistillationError(Exception):
    """
    Base class for every exception raised by the kd_pipeline package.

    Catch this in top-level scripts when you want to recover from any
    pipeline-specific error but still let true bugs (TypeError, etc.)
    propagate as crashes.
    """
    # `pass` means "no extra attributes or methods" — this is just a
    # type-tag we can pattern-match on with `except`.
    pass


class ConfigurationError(KnowledgeDistillationError):
    """
    Raised when the pipeline is misconfigured: missing files, invalid
    paths, unreadable env vars, or hyperparameters out of valid range.
    """
    pass


class PDFExtractionError(KnowledgeDistillationError):
    """
    Raised when extracting text or tables from the source PDF fails.

    Common causes:
      • The PDF file is missing or unreadable.
      • The PDF is encrypted / password-protected.
      • PyMuPDF or pdfplumber is not installed in the environment.
    """
    pass


class ChunkingError(KnowledgeDistillationError):
    """
    Raised when chunking fails — usually because the input pages list is
    empty or every page contained too little text after cleaning.
    """
    pass


class ModelLoadError(KnowledgeDistillationError):
    """
    Raised when the teacher GGUF model fails to load.

    Common causes:
      • Wrong path / file does not exist.
      • llama-cpp-python not built with CUDA but `n_gpu_layers=-1`.
      • Out-of-memory (model too big for RAM/VRAM).
    """
    pass


class GenerationError(KnowledgeDistillationError):
    """
    Raised when calling the teacher model for inference fails.

    Wraps lower-level llama-cpp-python errors so callers don't have to
    import llama_cpp to handle them.
    """
    pass


class QualityFilterError(KnowledgeDistillationError):
    """
    Raised when the quality filter receives malformed inputs (e.g.,
    `None` instead of a string). Logical "rejects" are NOT exceptions —
    they are returned as `(False, reason)` tuples.
    """
    pass


class DatasetSaveError(KnowledgeDistillationError):
    """
    Raised when writing the final dataset to disk fails — typically a
    permission error, disk-full error, or read-only filesystem.
    """
    pass


class TrainingError(KnowledgeDistillationError):
    """
    Raised when student fine-tuning fails.

    Common causes:
      • Unsloth / transformers / trl not installed.
      • CUDA not available (Unsloth refuses to run on pure CPU).
      • Out-of-memory during the forward/backward pass.
      • Malformed train.jsonl (missing `messages` key, wrong roles).
    """
    pass

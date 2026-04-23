"""
teacher_model.py
================
Wrapper around the GGUF teacher model loaded via `llama-cpp-python`.

WHY a wrapper class?
--------------------
The notebook called the model directly via `Llama(...)` and a free
function. That works for a script, but it leaks two implementation
details into every caller:
    • the constructor's hyperparameters (model_path, n_ctx, n_gpu_layers)
    • llama-cpp-python's specific result-dict shape

Wrapping in a class lets us:
    1. Centralise model construction (one place to update if path changes)
    2. Centralise post-processing (strip echoed excerpts, etc.)
    3. Swap the backend (e.g., to vLLM or HF transformers) without
       touching the rest of the pipeline.
"""

# `re` is used to strip echoed excerpts the model may produce.
import re
# Type hints.
from typing import List, Dict, Optional

# Internal imports.
from . import config
from .logger import get_logger
from .exceptions import ModelLoadError, GenerationError

log = get_logger(__name__)


class TeacherModel:
    """
    Encapsulates a loaded GGUF teacher model.

    Use:
        teacher = TeacherModel.load()
        response = teacher.generate(messages, use_cot=False)
    """

    def __init__(self, llm) -> None:
        """
        Construct from an already-loaded `llama_cpp.Llama` instance.

        Parameters
        ----------
        llm : llama_cpp.Llama
            The underlying llama-cpp-python model object. Pass this in
            instead of constructing inside ``__init__`` so the class can
            be unit-tested with a fake/mocked model.
        """
        # We store as a private attribute (single underscore prefix is
        # the Python convention for "internal — don't touch from outside").
        self._llm = llm

    # ─────────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        n_gpu_layers: Optional[int] = None,
        n_ctx: Optional[int] = None,
    ) -> "TeacherModel":
        """
        Load the GGUF teacher model from disk.

        Parameters
        ----------
        model_path, n_gpu_layers, n_ctx
            All optional — fall back to ``config.*`` defaults if omitted.
            Allowing override is handy for tests and quick experiments.

        Returns
        -------
        TeacherModel
            A ready-to-use wrapper.

        Raises
        ------
        ModelLoadError
            If llama-cpp-python is not installed, the file is missing,
            or model construction fails for any other reason.
        """
        # Resolve defaults from config. We do this *here*, not in
        # ``__init__``, so that tests can construct from a fake `llm`
        # without triggering config lookups.
        model_path   = model_path   or config.MODEL_PATH
        n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else config.N_GPU_LAYERS
        n_ctx        = n_ctx        if n_ctx        is not None else config.N_CTX

        log.info(
            "Loading teacher model: path=%s | n_gpu_layers=%s | n_ctx=%s",
            model_path, n_gpu_layers, n_ctx,
        )

        # Lazy import: keep the heavy `llama_cpp` import inside the method
        # so importing this *module* in tests doesn't require llama-cpp.
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            log.exception("llama-cpp-python is not installed")
            raise ModelLoadError(
                "llama-cpp-python is not installed. "
                "Install with: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            ) from exc

        # Wrap the actual constructor in try/except so we can convert
        # cryptic llama.cpp errors into our typed `ModelLoadError`.
        try:
            llm = Llama(
                model_path   = model_path,
                # `-1` offloads all transformer layers to GPU.
                n_gpu_layers = n_gpu_layers,
                # Context window in tokens (input + output combined).
                n_ctx        = n_ctx,
                # Suppress llama.cpp's verbose C++ progress logs.
                # Set True if you need to debug a load failure.
                verbose      = False,
            )
        except Exception as exc:
            log.exception("Failed to construct llama_cpp.Llama")
            raise ModelLoadError(
                f"Failed to load GGUF model from {model_path}: {exc}"
            ) from exc

        log.info("Teacher model loaded successfully (n_ctx=%d)", llm.n_ctx())
        return cls(llm)

    # ─────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: List[Dict],
        use_cot: bool,
    ) -> str:
        """
        Run one chat-completion call against the teacher.

        Parameters
        ----------
        messages : List[Dict]
            OpenAI-style chat messages, as built by
            ``prompt_templates.build_generation_prompt``.
        use_cot : bool
            ``True`` to use CoT hyperparameters (more tokens, slightly
            higher temperature). ``False`` for direct answers.

        Returns
        -------
        str
            The teacher's plain-text response, with echoed excerpts
            stripped.

        Raises
        ------
        GenerationError
            If llama-cpp-python raises during inference, or if the
            response shape is unexpected.
        """
        # Pick max tokens and temperature based on the mode. The
        # ternary expression `A if cond else B` is the most concise way
        # to express "branch on a flag" without extra variables.
        max_tokens  = config.MAX_TOKENS_COT if use_cot else config.MAX_TOKENS_DIRECT
        temperature = config.TEMPERATURE_COT if use_cot else config.TEMPERATURE_DIRECT

        log.debug(
            "generate(): use_cot=%s, max_tokens=%d, temperature=%.2f",
            use_cot, max_tokens, temperature,
        )

        # Wrap the actual inference call so any error becomes a typed
        # exception. Without this, callers would have to import
        # llama_cpp just to handle inference failures.
        try:
            result = self._llm.create_chat_completion(
                messages       = messages,
                max_tokens     = max_tokens,
                temperature    = temperature,
                # Penalises tokens already produced — prevents loops
                # like "Ignatiuz provides... Ignatiuz offers...".
                repeat_penalty = config.REPEAT_PENALTY,
                # Stop sequences: if the model produces any of these
                # strings, generation halts immediately. We use them to
                # prevent the model from echoing the excerpt back or
                # hallucinating a follow-up turn.
                stop           = ["---DOCUMENT EXCERPT", "User:", "\n\nUser"],
            )
        except Exception as exc:
            log.exception("Inference call to llama-cpp-python failed")
            raise GenerationError(f"Teacher inference failed: {exc}") from exc

        # The result is a dict in OpenAI ChatCompletion format. Defensive
        # navigation: if the shape is unexpected, raise a clear error
        # rather than letting a KeyError surface from deep inside.
        try:
            response = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            log.error("Unexpected response shape from llama-cpp: %r", result)
            raise GenerationError(
                f"Unexpected response shape from teacher model: {exc}"
            ) from exc

        # Post-process: strip any echoed excerpt that slipped through
        # before the stop token could fire. `re.DOTALL` makes `.` match
        # newlines so the regex deletes everything from the marker on.
        response = re.sub(
            r'---DOCUMENT EXCERPT.*',
            '',
            response,
            flags=re.DOTALL,
        )

        return response.strip()

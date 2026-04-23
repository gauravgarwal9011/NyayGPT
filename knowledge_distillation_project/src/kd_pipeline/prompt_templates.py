"""
prompt_templates.py
===================
Question templates and the prompt builder for grounded teacher inference.

WHY a separate module?
----------------------
Prompts are *not* configuration — they're domain logic. They evolve as we
discover what kinds of questions the teacher is good/bad at. Keeping them
in their own file makes it easy to A/B-test variants without touching the
core inference code.

WHY two template lists?
-----------------------
* `DIRECT_QUESTION_TEMPLATES` → factual, "look up the answer in the doc"
* `COT_SCENARIO_TEMPLATES`    → multi-step reasoning that *uses* the doc

Direct samples teach the student to be accurate. CoT samples teach the
student to *think* about the doc. A good dataset needs both.
"""

# `re` is used in `extract_topic_from_chunk` to find capitalised phrases.
import re
# `Counter` (from collections) tallies how often each phrase appears.
from collections import Counter
# Type hints.
from typing import List, Dict

# Internal imports.
from . import config
from .logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

# Each template contains `{topic}` which is filled in at run time with a
# representative phrase extracted from the chunk. Having 10 variants means
# the same chunk can produce 10 differently-worded questions, helping the
# student generalise to question phrasings it has not seen verbatim.
DIRECT_QUESTION_TEMPLATES: List[str] = [
    "Based on the document excerpt, what is {topic}?",
    "According to the document, what are the key details about {topic}?",
    "What specific results or outcomes does the document mention regarding {topic}?",
    "What does the document say about how {topic} works?",
    "What technologies or methods does the document describe for {topic}?",
    "What problem does {topic} solve according to the document?",
    "What are the specific numbers, metrics, or statistics mentioned about {topic}?",
    "How does the document describe the implementation of {topic}?",
    "What client use case does the document present related to {topic}?",
    "According to the excerpt, what are the benefits of {topic}?",
]

# CoT (chain-of-thought) templates phrase the question as a *scenario*
# that requires multi-step reasoning, not a one-shot factual lookup.
COT_SCENARIO_TEMPLATES: List[str] = [
    "A client is evaluating whether to implement {topic}. Based on the document evidence provided, walk through the decision step by step.",
    "Using only the information in this document excerpt about {topic}, help a CTO understand whether this technology fits their enterprise context.",
    "A prospect asks: 'Can you show us a real example of {topic} working?' Using the document excerpt, reason through what evidence exists and how to present it.",
    "Based on the document section about {topic}, what questions should a consultant ask a client before recommending this solution?",
    "The document mentions {topic}. If a client had a similar situation, reason through how Ignatiuz's approach would apply.",
]


def extract_topic_from_chunk(chunk_text: str) -> str:
    """
    Detect the most representative topic phrase in `chunk_text`.

    Parameters
    ----------
    chunk_text : str
        The text body of one chunk.

    Returns
    -------
    str
        A short phrase to substitute into `{topic}` placeholders. Falls
        back to the chunk's first line, or to ``"this topic"`` if even
        that is empty.

    Strategy
    --------
    1. Find every 2–4 word capitalised phrase (likely a slide title or
       proper-noun product name).
    2. Return the most frequent one.
    3. If none, fall back to the first non-page-marker line.
    4. If even that fails, return ``"this topic"``.
    """
    # Pattern explanation:
    #   \b                     — word boundary (so we don't catch part of a word)
    #   ([A-Z][a-z]+            — Capital + lowercase (start of a Title Word)
    #     (?: [A-Z][a-z]+){1,3} — followed by 1–3 more Title Words
    #   )\b                    — closing word boundary
    # That captures e.g. "Intelligent Document Processing" but not "I" or
    # the first word of every sentence.
    capitalised_phrases = re.findall(
        r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b',
        chunk_text,
    )

    if capitalised_phrases:
        # `Counter.most_common(1)` returns a list with one (item, count)
        # tuple — we index `[0][0]` to get the phrase itself.
        most_common = Counter(capitalised_phrases).most_common(1)[0][0]
        log.debug("Topic detected: %r", most_common)
        return most_common

    # Fallback: take the first non-empty line of the chunk. We strip page
    # markers like `[Page 12]` first since they're not real content.
    first_line = chunk_text.strip().split('\n')[0]
    first_line = re.sub(r'\[Page \d+\]', '', first_line).strip()

    # Cap to 60 characters so a runaway title doesn't blow up the prompt.
    # `or "this topic"` provides the ultimate fallback if `first_line` is
    # empty after stripping.
    return first_line[:60] if first_line else "this topic"


def build_generation_prompt(
    chunk_text: str,
    question: str,
    use_cot: bool,
) -> List[Dict]:
    """
    Build the chat-format messages list to send to the teacher.

    Parameters
    ----------
    chunk_text : str
        The grounding context the teacher must answer from.
    question : str
        The question (already topic-filled) to ask the teacher.
    use_cot : bool
        ``True`` for chain-of-thought; ``False`` for direct answer.

    Returns
    -------
    List[Dict]
        OpenAI-style ``[{"role": ..., "content": ...}, ...]`` messages,
        ready to pass to ``Llama.create_chat_completion(messages=...)``.

    Notes
    -----
    This is the heart of the grounding strategy: the chunk is *injected
    into the user message* as marked context, and the system prompt
    instructs the model to answer ONLY from that context. Structurally
    identical to a RAG system at inference time — except we're doing it
    once during dataset construction so the student learns the answers
    from memory.
    """
    if use_cot:
        # f-string with triple quotes lets us embed the chunk verbatim
        # without escaping newlines or quotes. The `---DOCUMENT EXCERPT
        # START/END---` markers are stop tokens we configure on the model
        # so it doesn't echo the excerpt back into its response.
        user_content = f"""Here is an excerpt from Ignatiuz's capabilities document:

---DOCUMENT EXCERPT START---
{chunk_text}
---DOCUMENT EXCERPT END---

{question}

Think through this step by step, referencing specific details from the excerpt above.
Your reasoning should show HOW you are using the document evidence to reach your conclusion.
Finish with a clear, actionable recommendation grounded in the document."""

        # Augment the system prompt for CoT mode:
        # * "Reasoning: high" is gpt-oss-20b's signal to enter extended
        #   thinking mode (more reasoning tokens before the final answer).
        # * The citation instruction makes the reasoning trace reference
        #   document evidence rather than emit floating logical chains.
        system = (
            config.SYSTEM_PROMPT
            + "\nReasoning: high\nAlways cite specific details from the provided excerpt in your reasoning."
        )

    else:
        user_content = f"""Here is an excerpt from Ignatiuz's capabilities document:

---DOCUMENT EXCERPT START---
{chunk_text}
---DOCUMENT EXCERPT END---

{question}

Answer using ONLY the information in the excerpt above.
Include exact numbers, names, and specific details as they appear in the document."""

        system = config.SYSTEM_PROMPT

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]

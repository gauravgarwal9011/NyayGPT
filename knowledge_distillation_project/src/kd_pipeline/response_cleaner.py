"""
response_cleaner.py
===================
Utilities for stripping teacher-side reasoning traces and chat-channel
markers from generated assistant responses before they are used for
student training or shown at inference time.
"""

import re
from typing import List, Dict


def sanitize_assistant_response(text: str) -> str:
    """
    Keep only the final answer portion of a teacher response.

    The teacher sometimes emits structured traces like:
        <|channel|>analysis<|message|>...
        <|end|><|start|>assistant<|channel|>final<|message|>...

    We do not want the student to learn those control tokens, so this
    helper extracts the final answer when present and otherwise removes
    the markers as best-effort cleanup.
    """
    cleaned = text.strip()

    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)",
        cleaned,
        flags=re.DOTALL,
    )
    if final_match:
        return final_match.group(1).strip()

    cleaned = re.sub(r"<\|channel\|>analysis<\|message\|>.*?(?=<\|end\|>|$)", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<\|start\|>assistant", "", cleaned)
    cleaned = re.sub(r"<\|channel\|>final<\|message\|>", "", cleaned)
    cleaned = re.sub(r"<\|channel\|>analysis<\|message\|>", "", cleaned)
    cleaned = re.sub(r"<\|end\|>", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_dataset_rows(rows: List[Dict]) -> List[Dict]:
    """
    Return dataset rows with assistant messages sanitized in-place-copy style.
    """
    cleaned_rows: List[Dict] = []
    for row in rows:
        cleaned_messages = []
        for message in row.get("messages", []):
            cleaned_message = dict(message)
            if cleaned_message.get("role") == "assistant":
                cleaned_message["content"] = sanitize_assistant_response(
                    cleaned_message.get("content", "")
                )
            cleaned_messages.append(cleaned_message)

        cleaned_row = dict(row)
        cleaned_row["messages"] = cleaned_messages
        cleaned_rows.append(cleaned_row)

    return cleaned_rows

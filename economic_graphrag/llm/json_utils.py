import json
import re
from typing import Any, Dict


class LLMJSONParseError(ValueError):
    """Raised when an LLM response cannot be parsed as a JSON object."""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_json_object(raw_response: Any) -> Dict[str, Any]:
    """
    Parse an LLM response into a JSON object.

    Handles normal JSON, fenced JSON, and responses with small amounts of
    surrounding text while still failing loudly for invalid or non-object JSON.
    """
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise LLMJSONParseError(f"Expected JSON string, got {type(raw_response).__name__}.")

    text = _strip_json_fences(raw_response)
    candidates = [text]

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if not isinstance(parsed, dict):
            raise LLMJSONParseError("Expected a JSON object from the LLM response.")
        return parsed

    raise LLMJSONParseError(f"Failed to parse LLM JSON response: {last_error}")

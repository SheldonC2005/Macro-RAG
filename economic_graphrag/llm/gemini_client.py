# economic_graphrag/llm/gemini_client.py
from typing import Any, Dict

import requests

from economic_graphrag.config import settings
from economic_graphrag.llm.json_utils import _strip_json_fences


class GeminiClient:
    """
    A lightweight client for Google's Gemini generateContent API.

    The public contract intentionally matches the previous LLM clients:
    generate(prompt, format="json") returns {"response": "..."} or {"error": "..."}.
    """

    def __init__(
        self,
        api_key: str = settings.GEMINI_API_KEY,
        model: str = settings.GEMINI_MODEL,
    ):
        self.api_key = api_key
        self.model = model
        self.endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY is not set.")

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def _build_payload(self, prompt: str, fmt: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }

        if fmt == "json":
            payload["generationConfig"] = {"responseMimeType": "application/json"}

        return payload

    def _extract_text(self, data: Dict[str, Any]) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        if not text_parts:
            raise ValueError("Gemini returned no text content.")

        return "".join(text_parts)

    def _format_error(self, response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text

        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        return str(body)

    def generate(self, prompt: str, format: str = "json") -> Dict[str, Any]:
        try:
            response = requests.post(
                self.endpoint,
                headers=self._build_headers(),
                json=self._build_payload(prompt, format),
                timeout=30,
            )
        except requests.exceptions.RequestException as exc:
            return {"error": str(exc)}

        if response.status_code != 200:
            return {"error": self._format_error(response), "status": response.status_code}

        try:
            content = self._extract_text(response.json())
        except (ValueError, KeyError, TypeError) as exc:
            return {"error": str(exc), "status": response.status_code}

        if format == "json":
            content = _strip_json_fences(content)

        return {"response": content}


if __name__ == "__main__":
    client = GeminiClient()
    result = client.generate('What is GDP? Respond in JSON with a "definition" key.')
    print(result)

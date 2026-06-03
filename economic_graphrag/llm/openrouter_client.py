# economic_graphrag/llm/openrouter_client.py
import requests
import json
import time
from typing import Dict, Any, List

from economic_graphrag.config import settings

# Ordered list of free models to try. First valid+available one wins.
# Models confirmed working are listed first.
FALLBACK_MODELS: List[str] = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "z-ai/glm-4.5-air:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "moonshotai/kimi-k2.6:free",
    "qwen/qwen3-coder:free",
]


class OpenRouterClient:
    """
    A client for interacting with the OpenRouter API.
    Automatically retries with fallback models when rate-limited (429) or
    when the configured model is unavailable (400/503).
    """

    def __init__(
        self,
        api_key: str = settings.OPENROUTER_API_KEY,
        model: str = settings.OPENROUTER_MODEL,
    ):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        if not self.api_key:
            print("WARNING: OPENROUTER_API_KEY is not set.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, model: str, prompt: str) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

    def _strip_json_fences(self, text: str) -> str:
        """Remove markdown code-fences that some models add around JSON."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _try_model(self, model: str, prompt: str, fmt: str) -> Dict[str, Any]:
        """
        Attempt a single API call with *model*.
        Returns:
          {"response": "<text>"}           on success
          {"error": "<msg>", "status": N}  on failure
        """
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._build_headers(),
                json=self._build_payload(model, prompt),
                timeout=30,
            )
        except requests.exceptions.RequestException as exc:
            return {"error": str(exc), "status": 0}

        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0]["message"]["content"]
                if fmt == "json":
                    content = self._strip_json_fences(content)
                return {"response": content}
            return {"error": "Empty choices in response", "status": 200}

        # Non-200 — surface status + raw body for the caller to decide
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {"error": str(body), "status": resp.status_code}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, format: str = "json") -> Dict[str, Any]:
        """
        Generate a response, automatically trying fallback models when the
        primary model returns 400 (invalid/retired model) or 429/503
        (rate-limited / provider error).

        Returns:
            {"response": "<text>"}  on success
            {"error": "<msg>"}      when all models fail
        """
        # Build the full candidate list: configured model first, then fallbacks
        candidates: List[str] = [self.model]
        for m in FALLBACK_MODELS:
            if m != self.model:
                candidates.append(m)

        last_error = "No models available."

        for i, model in enumerate(candidates):
            result = self._try_model(model, prompt, format)

            if "response" in result:
                if model != self.model:
                    print(f"[OpenRouterClient] Using fallback model: {model}")
                return result

            status = result.get("status", 0)
            error_msg = result.get("error", "unknown error")

            # 400 = model retired/invalid → skip immediately
            if status == 400:
                print(f"[OpenRouterClient] Model '{model}' is invalid (400). Skipping.")
                last_error = error_msg
                continue

            # 429 / 503 = rate-limited → short wait then try next
            if status in (429, 503):
                wait_sec = 2 if i < 3 else 5
                print(
                    f"[OpenRouterClient] Model '{model}' rate-limited ({status}). "
                    f"Waiting {wait_sec}s then trying next..."
                )
                time.sleep(wait_sec)
                last_error = error_msg
                continue

            # Other errors (network issues, auth, etc.) → stop immediately
            print(f"[OpenRouterClient] Unrecoverable error ({status}): {error_msg}")
            return {"error": error_msg}

        return {"error": f"All models failed. Last error: {last_error}"}


if __name__ == "__main__":
    client = OpenRouterClient()
    print(f"Primary model: {client.model}")
    result = client.generate('What is GDP? Respond in JSON with a "definition" key.')

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        try:
            parsed = json.loads(result["response"])
            print("Definition:", parsed.get("definition", result["response"]))
        except json.JSONDecodeError:
            print("Raw response:", result["response"])

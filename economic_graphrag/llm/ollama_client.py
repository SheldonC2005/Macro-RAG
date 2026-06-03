# economic_graphrag/llm/ollama_client.py
import requests
import json
from typing import Dict, Any, List

from economic_graphrag.config import settings

class OllamaClient:
    """
    A client for interacting with the Ollama API.
    """

    def __init__(self, endpoint: str = settings.OLLAMA_ENDPOINT, model: str = settings.OLLAMA_MODEL):
        self.endpoint = endpoint
        self.model = model

    def generate(self, prompt: str, format: str = "json") -> Dict[str, Any]:
        """
        Generate a response from the Ollama model.

        Args:
            prompt (str): The prompt to send to the model.
            format (str): The format of the response (e.g., "json").

        Returns:
            Dict[str, Any]: The JSON response from the model.
        """
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": format,
                    "stream": False,
                },
            )
            response.raise_for_status()
            response_text = response.text
            # The response from Ollama is a JSON object per line, we take the last one
            last_line = response_text.strip().split("\n")[-1]
            return json.loads(last_line)
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Ollama: {e}")
            print(f"Response text: {response.text}")
            return {"error": "Invalid JSON response"}

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from the Ollama API.

        Returns:
            List[Dict[str, Any]]: A list of models.
        """
        try:
            response = requests.get(f"{self.endpoint}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    client = OllamaClient()
    
    # Check if the model is available
    available_models = [model['name'] for model in client.get_models()]
    print(f"Available models: {available_models}")
    if settings.OLLAMA_MODEL not in available_models:
        print(f"Model '{settings.OLLAMA_MODEL}' not found. Please pull it using 'ollama pull {settings.OLLAMA_MODEL}'")
    else:
        print(f"Model '{settings.OLLAMA_MODEL}' is available.")
        
        # Example generation
        prompt = "What is GDP? Respond in JSON with a 'definition' key."
        result = client.generate(prompt)
        
        if "error" in result:
            print(f"An error occurred: {result['error']}")
        elif 'response' in result:
            try:
                # The actual response from the model is a string, which needs to be parsed as JSON again
                response_data = json.loads(result['response'])
                print("Generated Definition:")
                print(response_data.get("definition", "No definition found."))
            except json.JSONDecodeError:
                print("Failed to parse model response as JSON.")
                print(f"Raw response: {result['response']}")
        else:
            print("Unexpected response structure:")
            print(result)

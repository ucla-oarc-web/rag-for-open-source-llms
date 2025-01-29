import json
import requests
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

class OllamaLLM(LLM):
    endpoint_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "endpoint_url": self.endpoint_url,
            "model": self.model,
            "temperature": self.temperature,
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Send a POST request to Ollama's /generate endpoint, reading the stream until 'done' is true.
        Accumulate response text into a single output string.
        """
        generate_url = f"{self.endpoint_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
        }

        response = requests.post(generate_url, json=payload, stream=True)
        full_text = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to decode JSON: {e}")
                    print(f"Raw response: {line}")
                    continue  # Skip this line

                if "done" in data and data["done"] is True:
                    break
                if "response" in data:
                    full_text.append(data["response"])

        return "".join(full_text).strip()

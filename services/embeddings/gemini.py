from typing import List
import requests
from .base import BaseEmbedding


class GeminiEmbedding(BaseEmbedding):
    def __init__(self, api_key: str | None = None):
        if not api_key:
            raise ValueError("API key is required for Gemini embedding mode")
        self.api_key = api_key
        self.endpoint = "https://api.example.com/gemini/embeddings"  # placeholder

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        # This is a placeholder implementation; real Gemini API details vary.
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"inputs": docs}
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Expect data to have embeddings for each input
        return [item["embedding"] for item in data.get("data", [])]

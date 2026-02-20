from google import genai
from google.genai import types
from .base import EmbeddingBase

class GeminiEmbeddingService(EmbeddingBase):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="retrieval_document")
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="retrieval_query")
        )
        return result.embeddings[0].values
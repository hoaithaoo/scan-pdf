from typing import List
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding


class LocalEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        # SentenceTransformer returns numpy arrays; convert to lists
        embs = self.model.encode(docs, show_progress_bar=False)
        return [emb.tolist() for emb in embs]

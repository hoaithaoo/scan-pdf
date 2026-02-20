from sentence_transformers import SentenceTransformer
from .base import EmbeddingBase

class LocalEmbeddingService(EmbeddingBase):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        # Model này cực ngon cho tiếng Việt và rất nhẹ
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
from services.embeddings import local, gemini, base
from core.config import settings


class EmbeddingFactory:
    @staticmethod
    def get_embedding() -> base.BaseEmbedding:
        mode = (settings.EMBEDDING_MODE or "local").lower()
        if mode == "gemini":
            return gemini.GeminiEmbedding(api_key=settings.API_KEY)
        # default to local
        return local.LocalEmbedding()

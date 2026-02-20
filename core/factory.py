from .config import settings
from services.embeddings.local import LocalEmbeddingService
from services.embeddings.gemini import GeminiEmbeddingService

class EmbeddingFactory:
    @staticmethod
    def get_embedding_service():
        if settings.EMBEDDING_MODE == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is missing in .env")
            return GeminiEmbeddingService(api_key=settings.GEMINI_API_KEY)
        
        # Mặc định trả về Local nếu không cấu hình hoặc chọn local
        return LocalEmbeddingService()
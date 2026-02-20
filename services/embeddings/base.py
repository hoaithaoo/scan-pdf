from abc import ABC, abstractmethod

class EmbeddingBase(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Biến danh sách văn bản thành danh sách vector"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Biến câu hỏi của người dùng thành 1 vector duy nhất"""
        pass
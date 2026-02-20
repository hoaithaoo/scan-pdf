from typing import List, Protocol


class BaseEmbedding(Protocol):
    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        ...

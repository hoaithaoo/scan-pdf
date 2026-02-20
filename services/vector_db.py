from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from core.config import settings
import uuid


class VectorDB:
    def __init__(self, collection_name: str = "documents"):
        url = settings.QDRANT_URL
        api_key = settings.QDRANT_API_KEY
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection_name
        # Ensure collection exists (simple shape)
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(self.collection, vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE))

    def upsert(self, vectors: List[List[float]]) -> List[str]:
        ids = [str(uuid.uuid4()) for _ in vectors]
        points = [rest.PointStruct(id=_id, vector=vec) for _id, vec in zip(ids, vectors)]
        self.client.upsert(self.collection, points)
        return ids

    def search(self, vector: List[float], top: int = 5):
        res = self.client.search(self.collection, vector, limit=top)
        return res

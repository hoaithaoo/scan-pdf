from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self, host: str, port: int, api_key: str):
        # Kết nối tới Qdrant, không truyền api_key nếu để trống
        self.client = QdrantClient(
            url=f"http://{host}:{port}",
            api_key=api_key if api_key else None
        )

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int):
        """Tạo 'Bảng' lưu trữ an toàn, chỉ tạo khi bảng chưa tồn tại"""
        try:
            # Thử kiểm tra xem collection đã có chưa
            self.client.get_collection(collection_name)
            logger.info(f"📌 Collection '{collection_name}' đã tồn tại. Sẵn sàng lưu thêm dữ liệu.")
        except UnexpectedResponse as e:
            if e.status_code == 404: # Lỗi 404 nghĩa là chưa có collection
                logger.info(f"✨ Đang tạo mới Collection: '{collection_name}' (Độ dài vector: {vector_size})")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, 
                        distance=models.Distance.COSINE # Thuật toán tính độ giống nhau
                    )
                )
            else:
                raise e

    def upsert_documents(self, collection_name: str, chunks: list, embeddings: list):
        """Đẩy các đoạn chữ và dãy số vector vào lưu trữ"""
        if len(chunks) != len(embeddings):
            raise ValueError("❌ Số lượng đoạn văn và số lượng vector không khớp nhau!")

        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()), # Tạo một ID ngẫu nhiên, duy nhất cho mỗi đoạn
                vector=vector,        # Dãy số đại diện cho ý nghĩa
                payload={             # Dữ liệu gốc để con người và LLM đọc
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                }
            ))
        
        # Đẩy nguyên 1 batch lên Qdrant
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"✅ Đã lưu thành công {len(points)} đoạn văn vào cơ sở dữ liệu '{collection_name}'.")

    def search_similar(self, collection_name: str, query_vector: list, top_k: int = 5):
        """Tìm 5 đoạn văn giống với câu hỏi nhất để đưa cho AI đọc"""
        logger.info(f"🔍 Đang tìm kiếm {top_k} kết quả gần giống nhất trong '{collection_name}'...")

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        results = []
        for hit in response.points:
            results.append({
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {})
            })

        logger.info(f"✅ Tìm thấy {len(results)} đoạn văn phù hợp.")
        return results
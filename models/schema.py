from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]  # Lưu {page: 1, source: "file.pdf", ...}
    embedding: Optional[List[float]] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] # Trả về nguồn để AI không "chém gió"
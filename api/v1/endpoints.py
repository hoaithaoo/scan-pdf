from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schema import QueryRequest, QueryResponse
from core.factory import EmbeddingFactory
from services.pdf_handler import PDFHandler
from services.vector_db import VectorDBService
from core.config import settings
import os
import shutil
import traceback

router = APIRouter()

# Khởi tạo các Service cần thiết
pdf_handler = PDFHandler()
vector_db = VectorDBService(
    host=settings.QDRANT_HOST, 
    port=settings.QDRANT_PORT, 
    api_key=settings.QDRANT_API_KEY
)

@router.post("/ingest", tags=["PDF Processing"])
async def upload_pdf(file: UploadFile = File(...)):
    """API tiếp nhận file PDF, cắt nhỏ và lưu vào Qdrant"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF.")
    
    temp_path = f"temp_{file.filename}"
    try:
        # 1. Lưu file tạm thời
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Xử lý cắt nhỏ PDF
        print(">>> BƯỚC 1: Đang mổ PDF...")
        chunks = pdf_handler.process_pdf(temp_path)

        # 3. Lấy service embedding
        print(">>> BƯỚC 2: Đang tải Model AI (có thể hơi lâu)...")
        embed_service = EmbeddingFactory.get_embedding_service()

        # 4. Biến text thành vector
        print(">>> BƯỚC 3: Đang biến chữ thành số...")
        texts = [c["content"] for c in chunks]
        embeddings = embed_service.embed_documents(texts)

        # 5. Đẩy vào Qdrant
        print(">>> BƯỚC 4: Đang lưu vào Qdrant...")
        vector_size = len(embeddings[0])
        vector_db.create_collection_if_not_exists(settings.COLLECTION_NAME, vector_size)
        vector_db.upsert_documents(settings.COLLECTION_NAME, chunks, embeddings)

        return {"status": "success", "message": f"Đã xử lý {len(chunks)} đoạn văn."}

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"\n❌ LỖI RỒI:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/ask", response_model=QueryResponse, tags=["RAG Query"])
async def ask_question(request: QueryRequest):
    """Tìm các đoạn văn liên quan nhất từ Qdrant theo câu hỏi"""
    # 1. Embed câu hỏi
    embed_service = EmbeddingFactory.get_embedding_service()
    query_vector = embed_service.embed_query(request.query)

    # 2. Tìm kiếm trong Qdrant
    try:
        results = vector_db.search_similar(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vector,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm: {str(e)}")

    if not results:
        return QueryResponse(
            answer="Không tìm thấy đoạn văn nào phù hợp trong cơ sở dữ liệu.",
            citations=[]
        )

    # 3. Ghép nội dung các đoạn tìm được thành phần answer
    context_parts = [
        f"[{i+1}] (score: {r['score']:.2f}) {r['content']}"
        for i, r in enumerate(results)
    ]
    answer = "Dưới đây là các đoạn văn liên quan tìm được:\n\n" + "\n\n".join(context_parts)

    return QueryResponse(answer=answer, citations=results)
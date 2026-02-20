from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from core.config import settings
from services.pdf_handler import extract_text_from_pdf
from services.vector_db import VectorDB
from core.factory import EmbeddingFactory
from models.schema import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    text = extract_text_from_pdf(content)

    emb = EmbeddingFactory.get_embedding()
    vectors = emb.embed_documents([text])

    db = VectorDB()
    ids = db.upsert(vectors)

    return UploadResponse(filename=file.filename, indexed=len(ids))

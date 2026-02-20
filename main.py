from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.endpoints import router as api_router
from core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Hệ thống RAG hỗ trợ tra cứu PDF thông minh",
    version="1.0.0"
)

# Cấu hình CORS giúp các ứng dụng UI (React/Vue/Mobile) có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trong thực tế nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối các đầu API vào app chính
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "status": "Running",
        "mode": settings.EMBEDDING_MODE
    }
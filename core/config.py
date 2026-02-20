import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "PDF AI RAG"
    
    # Embedding Choice: "local" hoặc "gemini"
    EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "local")
    
    # Qdrant Config
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = "pdf_knowledge_base"
    
    # Gemini API Key
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

settings = Settings()
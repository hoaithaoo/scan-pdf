from pydantic import BaseSettings


class Settings(BaseSettings):
    API_KEY: str | None = None
    EMBEDDING_MODE: str = "local"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()

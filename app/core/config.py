"""
Application configuration using pydantic-settings.
All configuration is loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "products"

    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    google_api_key: str | None = None
    google_embedding_model: str = "models/gemini-embedding-001"
    use_colbert: bool = False

    # Vector Configuration
    dense_vector_size: int = 1024

    # API Configuration
    api_v1_prefix: str = "/api/v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

    # Model Configuration

settings = get_settings()

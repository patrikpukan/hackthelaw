from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "Legal RAG Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API
    api_v1_prefix: str = "/api/v1"
    
    # Database
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/legalrag"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Vector Database
    weaviate_url: str = "http://localhost:8080"
    
    # File upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_path: str = "./uploads"
    allowed_extensions: list = [".pdf", ".docx", ".txt", ".rtf"]
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # RAG
    retrieval_k: int = 5
    similarity_threshold: float = 0.7
    
    # LLM Provider settings
    llm_provider: str = "vertexai"  # groq, openai, vertexai, mock
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Vertex AI settings
    vertex_ai_project_id: Optional[str] = None
    vertex_ai_location: str = "us-central1"
    vertex_ai_model: str = "gemini-2.5-pro"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings() 
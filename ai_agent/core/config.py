import os
from os.path import dirname, join, abspath, exists
from typing import Optional

from pydantic import HttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def find_file_in_parent_folders(filename: str, start_dir: str) -> Optional[str]:
    """
    Search for a file in the current directory and all parent directories.
    Returns the path to the file if found, None otherwise.
    """
    current_dir = abspath(start_dir)
    while True:
        file_path = join(current_dir, filename)
        if exists(file_path):
            return file_path
        parent_dir = abspath(join(current_dir, '..'))
        if parent_dir == current_dir:
            # We've reached the root directory
            return None
        current_dir = parent_dir


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_file_in_parent_folders(".env", dirname(__file__)),
        env_ignore_empty=True,
        extra="ignore"
    )

    # Application Settings
    PROJECT_NAME: str = "AI Agent API"
    PROJECT_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "dev"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    OPENAPI_URL: str = "/openapi.json"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = [
        # Local development
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://localhost:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        # Production
        "http://3.6.178.255:3000",
        "http://3.6.178.255:8000",
        "https://agentiq.prismberry.com",
        "http://agentiq.techvisions.sa",
        "https://staging-ai-vision.prismberry.com",
        "https://dev-vision-iq.prismberry.com",
        # Allow all for development (remove in production)
        "*",
    ]
    
    # Database Settings
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 60
    DATABASE_MAX_OVERFLOW: int = 20
    
    # LLM Provider Settings
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    DEFAULT_LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1000
    LLM_SYSTEM_PROMPT: str | None = None
    
    # OpenAI Specific Settings
    OPENAI_API_KEY: str | None = None
    OPENAI_API_URL: str = 'https://api.openai.com/v1/chat/completions'

    # Z.AI (Zhipu AI) Settings — LiteLLM native provider (zai/ prefix)
    ZAI_API_KEY: str | None = None

    # Voice pipeline (STT via OpenAI Realtime transcription session, TTS via
    # OpenAI /v1/audio/speech). The agent brain is AgentRuntime — see
    # api/v1/voice.py.
    OPENAI_REALTIME_URL: str = "wss://api.openai.com/v1/realtime"
    OPENAI_STT_MODEL: str = "gpt-4o-transcribe"
    OPENAI_STT_VAD_THRESHOLD: float = 0.5
    OPENAI_STT_SILENCE_MS: int = 500
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
    OPENAI_TTS_VOICE: str = "alloy"

    # Vector Database Settings
    VECTOR_DB_TYPE: str | None = None
    VECTOR_CHUNK_SIZE: int = 1000
    VECTOR_CHUNK_OVERLAP: int = 200
    VECTOR_EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTOR_MAX_CHUNKS: int = 5
    EMBEDDINGS_BASE_URL: str | None = None
    EMBEDDINGS_API_KEY: str | None = None
    EMBEDDINGS_TIMEOUT: float = 180.0
    EMBEDDINGS_BATCH_SIZE: int = 32
    
    # Reranking Settings
    RERANK_BASE_URL: str | None = None
    RERANK_API_KEY: str | None = None
    RERANK_TIMEOUT: float = 15.0
    RERANK_INITIAL_K: int = 20
    RERANK_TOP_K: int = 10
    
    # File Storage
    LOCAL_STORAGE_ROOT: str = os.path.join(dirname(__file__), "..", "local_storage")
    DOC_DIR: str = os.path.join(dirname(__file__), "..", "documents")
    UPLOAD_FOLDER: str = "uploads"
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS: list[str] = ["txt", "pdf", "doc", "docx"]
    
    # Milvus Vector Database
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "documents"
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_DIMENSION: int = 1536  # text-embedding-3-small dimension
    
    # MinIO Object Storage
    MINIO_HOST: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str
    MINIO_SECURE: bool = False
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change this in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"
    
    # Rate Limiting
    RATE_LIMIT: str = "100/minute"
    
    # Caching
    CACHE_TTL: int = 300  # 5 minutes
    
    # Logging
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Pluggy
    PLUGGY_BASE_URL: str
    PLUGGY_USERNAME: str | None = None
    PLUGGY_PASSWORD: str | None = None
    PLUGGY_UNIFIED_API_URL: str = "https://pluggy-dev.prismberry.com/api/v1/pluggy/execute-api/unified"
    PLUGGY_API_KEY: str | None = None
    PLUGGY_MS_TEAMS_CONNECTOR_ID: str | None = None



    # Gallabox
    GALLABOX_SEND_URL: str
    GALLABOX_CONNECTOR_ID: str | None = None
    GALLABOX_TENANT_ID: str | None = None
    GALLABOX_API_SECRET: str | None = None
    GALLABOX_API_KEY: str | None = None
    GALLABOX_CHANNEL_ID: str | None = None
    GALLABOX_WELCOME_TEMPLATE_NAME: str
    GALLABOX_WELCOME_TEMPLATE_ID: str | None = None



    # AWS Credential
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str
    AWS_S3_BUCKET_NAME: str
    AWS_REGION: str
    
    # Email Settings (AWS SES)
    SES_FROM_EMAIL: str = ""
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None
    EMAIL_FROM: str | None = None
    EMAIL_USE_TLS: bool = True

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Celery / Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Guardrails
    ENABLE_PII_DETECTION: bool = True
    ENABLE_PROMPT_INJECTION_DETECTION: bool = True
    ENABLE_CONTENT_MODERATION: bool = True

    # Keycloak Identity Provider
    KEYCLOAK_SERVER_URL: str = "https://keycloak.prismberry.com/"
    KEYCLOAK_REALM: str = "AgentIQ"
    KEYCLOAK_BACKEND_CLIENT_ID: str = "backend"
    KEYCLOAK_BACKEND_CLIENT_SECRET: str = "5jibNRuLRdmV75KaK8hxANP6i8JSKnM7"
    SSO_CALLBACK_URL:str="https://agentiq-dev.prismberry.com/auth/callback"

    # Frontend
    FRONTEND_URL: str = "https://agentiq-dev.prismberry.com"

    # Free-Trial Token Quota
    FREE_TRIAL_QUOTA_ENABLED: bool
    FREE_TRIAL_TOKEN_LIMIT: int

    # Schema Engine
    SE_RETRIEVAL_TOP_K: int = 15
    SE_SCHEMA_CONTEXT_MAX_TOKENS: int = 3000
    SE_GRAPH_MAX_HOPS: int = 2
    SE_EMBEDDING_MODEL: str = "text-embedding-3-small"
    SE_SUMMARIZATION_MODEL: str = "gpt-4o-mini"
    SE_SQL_MAX_ROWS: int = 10000
    SE_SQL_ALLOW_WRITE: bool = False
    SE_QUERY_LOG_ENABLED: bool = True


settings = Settings()  # type: ignore

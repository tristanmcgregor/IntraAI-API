from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


def _parse_list(value: str | None) -> List[str]:
    if not value:
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_extensions(value: str | None) -> Set[str]:
    if not value:
        return {
            ".pdf",
            ".docx",
            ".pptx",
            ".txt",
            ".md",
            ".html",
            ".htm",
        }
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Settings:
    app_name: str = "IntraAI API"
    upload_directory: str = "documents"
    sqlite_path: str = "storage/index.db"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/intraai"
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_file_extensions: Set[str] = field(
        default_factory=lambda: {
            ".pdf",
            ".docx",
            ".pptx",
            ".txt",
            ".md",
            ".html",
            ".htm",
        }
    )
    max_upload_mb: int = 10
    openai_api_key: Optional[str] = None
    openai_embeddings_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 5
    disable_embeddings: bool = False
    disable_llm: bool = False
    aws_region: Optional[str] = None
    sqs_queue_url: Optional[str] = None
    api_key: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    s3_prefix: str = ""
    s3_endpoint_url: Optional[str] = None
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_exp_minutes: int = 60
    refresh_token_exp_minutes: int = 1440
    # Email/Contact settings
    resend_api_key: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[str] = None
    # Retrieval behavior
    company_only_mode: bool = False
    min_relevance: float = 0.25

    def __post_init__(self) -> None:
        _load_env_file()
        env = os.environ
        self.app_name = env.get("APP_NAME", self.app_name)
        self.upload_directory = env.get("UPLOAD_DIRECTORY", self.upload_directory)
        self.sqlite_path = env.get("SQLITE_PATH", self.sqlite_path)
        self.database_url = env.get("DATABASE_URL", self.database_url)
        self.allowed_origins = _parse_list(env.get("ALLOWED_ORIGINS"))
        self.allowed_file_extensions = _parse_extensions(
            env.get("ALLOWED_FILE_EXTENSIONS")
        )
        self.max_upload_mb = _parse_int(env.get("MAX_UPLOAD_MB"), self.max_upload_mb)
        self.openai_api_key = env.get("OPENAI_API_KEY", self.openai_api_key)
        self.openai_embeddings_model = env.get(
            "OPENAI_EMBEDDINGS_MODEL", self.openai_embeddings_model
        )
        self.openai_chat_model = env.get("OPENAI_CHAT_MODEL", self.openai_chat_model)
        self.chunk_size = _parse_int(env.get("CHUNK_SIZE"), self.chunk_size)
        self.chunk_overlap = _parse_int(env.get("CHUNK_OVERLAP"), self.chunk_overlap)
        self.top_k = _parse_int(env.get("TOP_K"), self.top_k)
        self.disable_embeddings = _parse_bool(
            env.get("DISABLE_EMBEDDINGS"), self.disable_embeddings
        )
        self.disable_llm = _parse_bool(env.get("DISABLE_LLM"), self.disable_llm)
        self.aws_region = env.get("AWS_REGION", self.aws_region)
        self.sqs_queue_url = env.get("SQS_QUEUE_URL", self.sqs_queue_url)
        self.api_key = env.get("API_KEY", self.api_key)
        self.s3_bucket_name = env.get("S3_BUCKET_NAME", self.s3_bucket_name)
        self.s3_prefix = env.get("S3_PREFIX", self.s3_prefix).strip()
        self.s3_endpoint_url = env.get("S3_ENDPOINT_URL", self.s3_endpoint_url)
        self.jwt_secret = env.get("JWT_SECRET", self.jwt_secret)
        self.jwt_algorithm = env.get("JWT_ALGORITHM", self.jwt_algorithm)
        self.access_token_exp_minutes = _parse_int(
            env.get("ACCESS_TOKEN_EXPIRE_MINUTES"), self.access_token_exp_minutes
        )
        self.refresh_token_exp_minutes = _parse_int(
            env.get("REFRESH_TOKEN_EXPIRE_MINUTES"), self.refresh_token_exp_minutes
        )
        # Email/Contact
        self.resend_api_key = env.get("RESEND_API_KEY", self.resend_api_key)
        self.email_from = env.get("EMAIL_FROM", self.email_from)
        self.email_to = env.get("EMAIL_TO", self.email_to)
        # Retrieval behavior
        self.company_only_mode = _parse_bool(
            env.get("COMPANY_ONLY_MODE"), self.company_only_mode
        )
        self.min_relevance = _parse_float(
            env.get("MIN_RELEVANCE"), self.min_relevance
        )

    def ensure_directories(self) -> None:
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


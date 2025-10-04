from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, validator, EmailStr, ConfigDict


class IngestResponse(BaseModel):
    files_indexed: int = Field(..., description="Number of files successfully indexed")
    chunks_added: int = Field(..., description="Number of chunks added to the vector store")
    errors: List[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., min_length=1, description="Message content")

    @validator("content")
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Content must not be empty")
        return cleaned


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = None
    history: Optional[List[ChatMessage]] = Field(None, description="Prior chat messages")


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    used_provider: str


class StreamChunk(BaseModel):
    type: Literal["message", "done", "error"]
    content: Optional[str] = None
    used_provider: Optional[str] = None


class TitleRequest(BaseModel):
    first_message: str = Field(..., min_length=3, description="First user message")


class TitleResponse(BaseModel):
    title: str = Field(..., description="Short chat title")
    used_provider: str = Field(..., description="Provider used for title generation")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="User password")


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Literal["admin", "employee"] = "employee"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    role: Literal["admin", "employee"]
    created_at: datetime
    display_name: str


class LoginResponse(BaseModel):
    user: UserOut
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class FileMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_name: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime
    s3_key: str
    uploader_email: Optional[EmailStr]
    download_url: Optional[str] = None

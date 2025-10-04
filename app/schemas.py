from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum("admin", "employee", name="user_roles"), nullable=False, default="employee")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    files = relationship("File", back_populates="uploader", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(512), nullable=False)
    s3_key = Column(String(1024), nullable=False, unique=True)
    content_type = Column(String(255), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    embedding_status = Column(String(50), nullable=False, default="uploaded")

    uploader = relationship("User", back_populates="files")
    accesses = relationship("FileAccess", back_populates="file", cascade="all, delete-orphan")


class FileAccess(Base):
    __tablename__ = "file_access"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    permission = Column(Enum("read", "write", name="file_permissions"), nullable=False, default="read")
    granted_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    file = relationship("File", back_populates="accesses")

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import schemas


def get_user_by_email(db: Session, email: str) -> Optional[schemas.User]:
    stmt = select(schemas.User).where(schemas.User.email == email.lower())
    return db.scalars(stmt).first()


def create_user(db: Session, email: str, password_hash: str, role: str = "employee") -> schemas.User:
    email_lower = email.lower()
    user = schemas.User(email=email_lower, password_hash=password_hash, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_file(
    db: Session,
    *,
    user_id: int,
    file_name: str,
    s3_key: str,
    content_type: str,
    size_bytes: int,
    embedding_status: str = "pending",
) -> schemas.File:
    file = schemas.File(
        user_id=user_id,
        file_name=file_name,
        s3_key=s3_key,
        content_type=content_type,
        size_bytes=size_bytes,
        embedding_status=embedding_status,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def list_files_for_user(db: Session, user_id: int) -> list[schemas.File]:
    stmt = select(schemas.File).where(schemas.File.user_id == user_id)
    return list(db.scalars(stmt))


def update_file_embedding_status(db: Session, file_id: int, status: str) -> None:
    file_obj = db.get(schemas.File, file_id)
    if file_obj:
        file_obj.embedding_status = status
        db.add(file_obj)
        db.commit()

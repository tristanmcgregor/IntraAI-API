from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt
from passlib.context import CryptContext

from .config import Settings

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def _settings() -> Settings:
    return Settings()


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    settings = _settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_exp_minutes)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    settings = _settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_refresh_exp_minutes)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    settings = _settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
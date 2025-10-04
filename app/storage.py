from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import Settings


class StorageError(Exception):
    pass


@dataclass
class StoredDocument:
    source: str
    bytes: bytes
    content_type: str
    s3_key: str


class StorageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def _get_client(self):
        if not self.settings.s3_bucket_name:
            raise StorageError("S3 bucket not configured")
        if self._client is None:
            self._client = boto3.client(
                "s3",
                region_name=self.settings.aws_region,
                endpoint_url=self.settings.s3_endpoint_url,
            )
        return self._client

    def _make_key(self, safe_name: str) -> str:
        prefix = self.settings.s3_prefix.strip()
        unique = f"{uuid4().hex}_{safe_name}"
        return os.path.join(prefix, unique) if prefix else unique

    def save(self, safe_name: str, blob: bytes, *, content_type: str = "application/octet-stream") -> StoredDocument:
        client = self._get_client()
        key = self._make_key(safe_name)
        try:
            client.put_object(
                Bucket=self.settings.s3_bucket_name,
                Key=key,
                Body=blob,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Failed to upload {safe_name} to S3: {exc}") from exc

        return StoredDocument(
            source=safe_name,
            bytes=blob,
            content_type=content_type,
            s3_key=key,
        )

    def delete_many(self, keys: List[str]) -> None:
        client = self._get_client()
        try:
            objects = [{"Key": key} for key in keys]
            client.delete_objects(Bucket=self.settings.s3_bucket_name, Delete={"Objects": objects})
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Cleanup failed for keys {keys}: {exc}") from exc

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        client = self._get_client()
        try:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.settings.s3_bucket_name, "Key": key},
                ExpiresIn=expires_in,
            )
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Failed to presign {key}: {exc}") from exc

    def download(self, key: str) -> bytes:
        client = self._get_client()
        try:
            response = client.get_object(
                Bucket=self.settings.s3_bucket_name,
                Key=key,
            )
            return response["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Failed to download {key}: {exc}") from exc

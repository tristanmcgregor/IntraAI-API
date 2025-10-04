from __future__ import annotations

from typing import List
import sqlite3
import json
from pathlib import Path
import os
import re

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from .database import init_db, get_session
from .schemas import User, File as FileModel
from .crud import get_user_by_email, create_user, create_file, list_files_for_user
from .security import (
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)

from .config import Settings
from .models import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    Citation,
    TitleRequest,
    TitleResponse,
    LoginRequest,
    LoginResponse,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    FileMetadata,
)
from .ingest import (
    load_documents_from_records,
    chunk_documents,
    ensure_schema,
    upsert_chunk,
    compute_id,
)
from .rag import answer_question, stream_answer
from .storage import StorageService, StorageError


settings = Settings()
settings.ensure_directories()
init_db()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parents[2]

oauth_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid authentication credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    authorization: str = Header(..., alias="Authorization"),
    db: Session = Depends(get_session),
) -> User:
    if not authorization.startswith("Bearer "):
        raise oauth_exception
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception:
        raise oauth_exception
    if payload.get("type") != "access":
        raise oauth_exception
    user_id = payload.get("sub")
    if not user_id:
        raise oauth_exception
    user = db.get(User, user_id)
    if not user:
        raise oauth_exception
    return user


def _user_to_out(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at,
        "display_name": user.email.split("@")[0],
    }


@app.post("/auth/register", response_model=LoginResponse)
def register_user(payload: UserCreate, db: Session = Depends(get_session)) -> LoginResponse:
    existing = get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    password_hash = hash_password(payload.password)
    user = create_user(db, payload.email, password_hash, payload.role)
    access_token = create_access_token({"sub": user.id})
    refresh_token = create_refresh_token({"sub": user.id})
    return LoginResponse(user=_user_to_out(user), access_token=access_token, refresh_token=refresh_token)


@app.post("/auth/login", response_model=LoginResponse)
def auth_login(payload: LoginRequest, db: Session = Depends(get_session)) -> LoginResponse:
    user = get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    access_token = create_access_token({"sub": user.id})
    refresh_token = create_refresh_token({"sub": user.id})
    return LoginResponse(user=_user_to_out(user), access_token=access_token, refresh_token=refresh_token)


@app.post("/auth/refresh", response_model=TokenResponse)
def refresh_tokens(payload: RefreshRequest, db: Session = Depends(get_session)) -> TokenResponse:
    try:
        data = decode_token(payload.refresh_token)
    except Exception:
        raise oauth_exception
    if data.get("type") != "refresh":
        raise oauth_exception
    user_id = data.get("sub")
    user = db.get(User, user_id)
    if not user:
        raise oauth_exception
    access_token = create_access_token({"sub": user.id})
    refresh_token = create_refresh_token({"sub": user.id})
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@app.get("/files", response_model=List[FileMetadata])
def list_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> List[FileMetadata]:
    files = list_files_for_user(db, current_user.id)
    if not settings.s3_bucket_name:
        raise HTTPException(status_code=500, detail="S3 storage is not configured")
    storage = StorageService(settings)
    result: List[FileMetadata] = []
    for file_obj in files:
        presigned = storage.generate_presigned_url(file_obj.s3_key)
        result.append(
            FileMetadata(
                id=file_obj.id,
                file_name=file_obj.file_name,
                content_type=file_obj.content_type,
                size_bytes=file_obj.size_bytes,
                uploaded_at=file_obj.uploaded_at,
                s3_key=file_obj.s3_key,
                uploader_email=current_user.email,
                download_url=presigned,
            )
        )
    return result


def _safe_filename(raw_name: str, used: set[str]) -> str:
    name = Path(raw_name).name
    if not name:
        name = "upload"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    base, ext = Path(safe).stem, Path(safe).suffix
    if not base:
        base = "file"
    candidate = f"{base}{ext}" if ext else base
    idx = 1
    while candidate in used:
        candidate = f"{base}_{idx}{ext}"
        idx += 1
    used.add(candidate)
    return candidate


def _validate_upload(
    filename: str,
    blob_len: int,
    errors: List[str],
    used: set[str],
) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext and ext not in settings.allowed_file_extensions:
        errors.append(f"Blocked file type for {filename}.")
        return None
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if blob_len > max_bytes:
        errors.append(f"{filename} exceeds {settings.max_upload_mb} MB limit.")
        return None
    return _safe_filename(filename, used)


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    landing_path = REPO_ROOT / "frontend" / "index.html"
    if landing_path.exists():
        return HTMLResponse(landing_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>IntraAI API</h1>")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.get("/stats")
async def stats(db: Session = Depends(get_session)) -> dict:
    total_files = db.scalar(select(func.count()).select_from(FileModel)) or 0
    total_bytes = db.scalar(select(func.coalesce(func.sum(FileModel.size_bytes), 0))) or 0
    by_type_rows = db.execute(
        select(FileModel.content_type, func.count())
        .group_by(FileModel.content_type)
    ).all()
    by_type = {ctype or "unknown": count for ctype, count in by_type_rows}

    try:
        db_size = os.path.getsize(settings.sqlite_path)
    except Exception:
        db_size = 0

    return {
        "app": settings.app_name,
        "totalFiles": total_files,
        "totalBytes": total_bytes,
        "documentsByContentType": by_type,
        "legacySqliteBytes": db_size,
    }


def _maybe_enqueue_chunks(
    chunks: List[tuple[str, dict]], errors: List[str]
) -> int | None:
    """If SQS is configured, enqueue one message per chunk and return number enqueued.
    Returns None if SQS not configured.
    """
    if not settings.sqs_queue_url:
        return None
    try:
        import boto3  # type: ignore

        client = (
            boto3.client("sqs", region_name=settings.aws_region)
            if settings.aws_region
            else boto3.client("sqs")
        )
        enqueued = 0
        for content, meta in chunks:
            payload = {
                "type": "embed_chunk_v1",
                "source": meta.get("source", "unknown"),
                "path": meta.get("path", ""),
                "content": content,
            }
            client.send_message(
                QueueUrl=settings.sqs_queue_url, MessageBody=json.dumps(payload)
            )
            enqueued += 1
        return enqueued
    except Exception as e:
        errors.append(f"Failed to enqueue to SQS: {e}")
        return 0


@app.post(
    "/ingest/files",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
async def ingest_files(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> IngestResponse:
    if not settings.s3_bucket_name:
        raise HTTPException(status_code=500, detail="S3 storage is not configured")
    upload_blobs = []
    errors: List[str] = []
    used_names: set[str] = set()
    for f in files:
        try:
            blob = await f.read()
            filename = f.filename or "upload"
            safe_name = _validate_upload(filename, len(blob), errors, used_names)
            if not safe_name:
                continue
            content_type = f.content_type or "application/octet-stream"
            upload_blobs.append((safe_name, blob, content_type))
        except Exception as e:
            errors.append(f"Failed to read {f.filename}: {e}")

    storage = StorageService(settings)
    stored_documents = []
    for safe_name, blob, content_type in upload_blobs:
        try:
            stored = storage.save(safe_name, blob, content_type=content_type)
            stored_documents.append({
                "source": stored.source,
                "bytes": stored.bytes,
                "content_type": stored.content_type,
                "s3_key": stored.s3_key,
            })
            file_record = create_file(
                db,
                user_id=current_user.id,
                file_name=safe_name,
                s3_key=stored.s3_key,
                content_type=content_type,
                size_bytes=len(blob),
                embedding_status="queued" if settings.sqs_queue_url else "processing",
            )
        except StorageError as e:
            errors.append(str(e))

    docs = load_documents_from_records(stored_documents)
    chunks = chunk_documents(docs, settings)

    # If SQS is configured, enqueue and return
    enqueued = _maybe_enqueue_chunks(chunks, errors)
    if enqueued is not None:
        return IngestResponse(
            files_indexed=len(docs), chunks_added=enqueued or 0, errors=errors
        )

    from openai import OpenAI

    client = (
        OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    )
    conn = sqlite3.connect(settings.sqlite_path)
    ensure_schema(conn)
    added = 0
    if client is None:
        errors.append("OPENAI_API_KEY not set; skipped embedding.")
    else:
        stop_early = False
        for content, meta in chunks:
            try:
                resp = client.embeddings.create(
                    model=settings.openai_embeddings_model, input=[content]
                )
                emb = resp.data[0].embedding
                emb_blob = json.dumps(emb).encode("utf-8")
                cid = compute_id(meta.get("source", "unknown"), content)
                upsert_chunk(
                    conn,
                    cid,
                    meta.get("source", "unknown"),
                    meta.get("path", ""),
                    content,
                    emb_blob,
                )
                added += 1
            except Exception as e:
                err_str = str(e)
                errors.append(
                    f"Embed failed for {meta.get('source','unknown')}: {err_str}"
                )
                if (
                    "insufficient_quota" in err_str
                    or "You exceeded your current quota" in err_str
                ):
                    errors.append("Stopping early due to quota error.")
                    stop_early = True
                    break
        if stop_early:
            pass
    conn.close()

    return IngestResponse(files_indexed=len(docs), chunks_added=added, errors=errors)


@app.post(
    "/ingest/scan",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
async def ingest_scan() -> IngestResponse:
    root = Path(settings.upload_directory)
    allowed_exts = settings.allowed_file_extensions
    all_files = []
    for p in root.rglob("*"):
        if p.is_file():
            # Enforce extension allowlist here as well
            if p.suffix.lower() and p.suffix.lower() not in allowed_exts:
                continue
            all_files.append(str(p))
    if not all_files:
        return IngestResponse(files_indexed=0, chunks_added=0, errors=[])

    docs = load_documents_from_records(all_files)
    chunks = chunk_documents(docs, settings)

    # If SQS is configured, enqueue and return
    errors: List[str] = []
    enqueued = _maybe_enqueue_chunks(chunks, errors)
    if enqueued is not None:
        return IngestResponse(
            files_indexed=len(docs), chunks_added=enqueued or 0, errors=errors
        )

    from openai import OpenAI

    client = (
        OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    )
    conn = sqlite3.connect(settings.sqlite_path)
    ensure_schema(conn)
    added = 0
    errors = []
    if client is None:
        errors.append("OPENAI_API_KEY not set; skipped embedding.")
    else:
        stop_early = False
        for content, meta in chunks:
            try:
                resp = client.embeddings.create(
                    model=settings.openai_embeddings_model, input=[content]
                )
                emb = resp.data[0].embedding
                emb_blob = json.dumps(emb).encode("utf-8")
                cid = compute_id(meta.get("source", "unknown"), content)
                upsert_chunk(
                    conn,
                    cid,
                    meta.get("source", "unknown"),
                    meta.get("path", ""),
                    content,
                    emb_blob,
                )
                added += 1
            except Exception as e:
                err_str = str(e)
                errors.append(
                    f"Embed failed for {meta.get('source','unknown')}: {err_str}"
                )
                if (
                    "insufficient_quota" in err_str
                    or "You exceeded your current quota" in err_str
                ):
                    errors.append("Stopping early due to quota error.")
                    stop_early = True
                    break
        if stop_early:
            pass
    conn.close()

    return IngestResponse(files_indexed=len(docs), chunks_added=added, errors=errors)


@app.post(
    "/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)]
)
async def query(req: QueryRequest) -> QueryResponse:
    answer, citations_raw, provider = answer_question(
        settings,
        req.question,
        top_k=req.top_k,
        history=req.history,
    )
    citations = [Citation(**c) for c in citations_raw]
    return QueryResponse(answer=answer, citations=citations, used_provider=provider)


@app.post(
	"/query/stream",
	dependencies=[Depends(require_api_key)],
)
async def query_stream(req: QueryRequest) -> StreamingResponse:
	generator = stream_answer(
		settings,
		req.question,
		top_k=req.top_k,
		history=req.history,
	)
	return StreamingResponse(generator, media_type="application/x-ndjson")


@app.post("/chat/title", response_model=TitleResponse, dependencies=[Depends(require_api_key)])
async def create_chat_title(req: TitleRequest) -> TitleResponse:
    """Generate a short chat title from the first user message.
    Uses OpenAI if configured, otherwise falls back to a simple heuristic.
    """
    text = req.first_message.strip()
    # Fallback heuristic: take first sentence or first 6 words
    fallback = (
        (text.split(".")[0] or text)
        .strip()
        .split()
    )
    simple = " ".join(fallback[:6]).strip()
    simple = (simple[:48] + "…") if len(simple) > 48 else simple

    if not settings.openai_api_key:
        return TitleResponse(title=simple or "New Chat", used_provider="simple")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        prompt = (
            "Write a very short, 3-6 word title for this chat. "
            "No quotes, no punctuation, title case.\n\nMessage: " + text[:2000]
        )
        resp = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": "You generate concise titles only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16,
        )
        title = (resp.choices[0].message.content or simple).strip().strip('"')
        title = title[:60]
        return TitleResponse(title=title or simple or "New Chat", used_provider="openai")
    except Exception:
        # On any error, return fallback
        return TitleResponse(title=simple or "New Chat", used_provider="simple")


@app.post("/contact", dependencies=[Depends(require_api_key)])
async def contact(
    payload: dict = Body(...),
):
    """Send a contact message via Resend.
    Expects JSON: { name, email, company?, phone?, message }
    """
    required = ["name", "email", "message"]
    missing = [k for k in required if not str(payload.get(k, "")).strip()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing)}")

    if not settings.resend_api_key or not settings.email_from or not settings.email_to:
        raise HTTPException(status_code=500, detail="Email service not configured")

    # Build email content
    name = str(payload.get("name", "")).strip()
    email = str(payload.get("email", "")).strip()
    company = str(payload.get("company", "")).strip()
    phone = str(payload.get("phone", "")).strip()
    message = str(payload.get("message", "")).strip()

    subject = f"[IntraAI Contact] {name} — {company or 'No company'}"
    text = (
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Company: {company or '-'}\n"
        f"Phone: {phone or '-'}\n\n"
        f"Message:\n{message}\n"
    )

    try:
        import requests  # already in requirements

        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {settings.resend_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": settings.email_from,
                "to": [settings.email_to],
                "subject": subject,
                "text": text,
                # Optionally: add reply_to to make replies go to the submitter
                "reply_to": email,
            },
            timeout=10,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Email send failed: {resp.text}")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email error: {e}")

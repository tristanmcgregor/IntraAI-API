from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import sqlite3
import json
import math
from collections import defaultdict

from openai import OpenAI

from .config import Settings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def retrieve_chunks(settings: Settings, question: str, top_k: int | None = None) -> List[tuple[str, dict, float]]:
    conn = sqlite3.connect(settings.sqlite_path)
    cur = conn.cursor()
    cur.execute("SELECT id, source, path, content, embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return []

    # If embeddings disabled or no key, skip vector scoring
    if settings.disable_embeddings or not settings.openai_api_key:
        # Simple keyword scoring as a fallback
        q_terms = [t.lower() for t in question.split() if t]
        results: List[tuple[str, dict, float]] = []
        seen_contents = set()
        for _id, source, path, content, embedding_blob in rows:
            key = (source, content.strip().lower())
            if key in seen_contents:
                continue
            seen_contents.add(key)
            text = content.lower()
            score = sum(text.count(t) for t in q_terms)
            results.append((content, {"source": source, "path": path}, float(score)))
        results.sort(key=lambda x: x[2], reverse=True)
        cap_per_source = 2
        per_source: dict[str, int] = defaultdict(int)
        filtered: List[tuple[str, dict, float]] = []
        for c, m, s in results:
            if per_source[m["source"]] >= cap_per_source:
                continue
            per_source[m["source"]] += 1
            filtered.append((c, m, s))
        return filtered[: (top_k or settings.top_k)]

    client = OpenAI(api_key=settings.openai_api_key)
    [q_vec] = _embed_texts(client, settings.openai_embeddings_model, [question])
    results: List[tuple[str, dict, float]] = []
    seen_contents = set()
    for _id, source, path, content, embedding_blob in rows:
        # Stored as JSON bytes; decode safely before loading
        vec = json.loads(embedding_blob.decode("utf-8"))
        score = _cosine_similarity(q_vec, vec)
        # Deduplicate identical content blocks
        key = (source, content.strip().lower())
        if key in seen_contents:
            continue
        seen_contents.add(key)
        results.append((content, {"source": source, "path": path}, float(score)))
    results.sort(key=lambda x: x[2], reverse=True)

    # Cap per-source to reduce repetition
    cap_per_source = 2
    per_source: dict[str, int] = defaultdict(int)
    filtered: List[tuple[str, dict, float]] = []
    for c, m, s in results:
        if per_source[m["source"]] >= cap_per_source:
            continue
        per_source[m["source"]] += 1
        filtered.append((c, m, s))
    return filtered[: (top_k or settings.top_k)]


def _format_history(history: Optional[Sequence[object]]) -> List[dict[str, str]]:
    formatted: List[dict[str, str]] = []
    if not history:
        return formatted
    for entry in history:
        role = getattr(entry, "role", None)
        content = getattr(entry, "content", None)
        if isinstance(entry, dict):
            role = entry.get("role", role)
            content = entry.get("content", content)
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        clean = content.strip()
        if not clean:
            continue
        formatted.append({"role": role, "content": clean})
    return formatted


def _filter_by_relevance(
    chunks: List[tuple[str, dict, float]],
    threshold: float,
) -> List[tuple[str, dict, float]]:
    # For keyword fallback, scores can be large integers; treat any positive as relevant
    if not chunks:
        return []
    if chunks and chunks[0][2] > 1.0 and any(isinstance(x[2], float) and x[2] <= 1.0 for x in chunks):
        pass
    return [c for c in chunks if c[2] >= threshold]


def answer_question(
    settings: Settings,
    question: str,
    top_k: int | None = None,
    history: Optional[Sequence[object]] = None,
) -> tuple[str, list[dict], str]:
    chunks = retrieve_chunks(settings, question, top_k)
    # Apply relevance threshold when embeddings available
    if chunks and isinstance(chunks[0][2], float):
        chunks = _filter_by_relevance(chunks, settings.min_relevance)
    citations: List[dict] = []
    context = ""

    if chunks:
        context = "\n\n".join([f"[Source: {m['source']}]\n{c}" for c, m, s in chunks])
        citations = [
            {"source": m["source"], "page": None, "score": s}
            for c, m, s in chunks
        ]

    # Cost control: disable LLM entirely if configured
    if settings.disable_llm or not settings.openai_api_key:
        if chunks:
            return (
                f"Most relevant context (LLM disabled):\n\n{chunks[0][0][:800]}...",
                citations,
                "context-only",
            )
        msg = (
            "No relevant company context found."
            if settings.company_only_mode
            else "LLM disabled and no context retrieved."
        )
        return (msg, [], "context-only")

    client = OpenAI(api_key=settings.openai_api_key)
    system_content = (
        "You are an internal assistant. When company context is provided, use it and ground all"
        " claims in that context with accurate citations. If no relevant company context is"
        " available, answer from general knowledge helpfully and concisely."
        if not settings.company_only_mode
        else "You are an internal company assistant. Use only the provided company context when"
        " answering. If there is no relevant context, politely say you don't have company"
        " information on that topic and do not answer from general knowledge."
    )
    messages = [
        {
            "role": "system",
            "content": system_content,
        },
    ]
    messages.extend(_format_history(history))
    prompt = f"Question: {question}"
    if context:
        prompt += f"\n\nContext:\n{context}"
    else:
        if settings.company_only_mode:
            # Explicit instruction for refusal path
            prompt += "\n\nNo relevant company context was found."
    messages.append({"role": "user", "content": prompt})
    # If company-only mode and no context, short-circuit with refusal without calling LLM
    if settings.company_only_mode and not context:
        return (
            "I couldn't find relevant company files for that question.",
            [],
            "context-only",
        )
    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0,
    )
    answer = resp.choices[0].message.content
    return (answer, citations, "openai")


def stream_answer(
    settings: Settings,
    question: str,
    top_k: int | None = None,
    history: Optional[Sequence[object]] = None,
) -> Iterable[bytes]:
    chunks = retrieve_chunks(settings, question, top_k)
    if chunks and isinstance(chunks[0][2], float):
        chunks = _filter_by_relevance(chunks, settings.min_relevance)
    citations: List[dict] = []
    context = ""

    if chunks:
        context = "\n\n".join([f"[Source: {m['source']}]\n{c}" for c, m, s in chunks])
        citations = [
            {"source": m["source"], "page": None, "score": s}
            for c, m, s in chunks
        ]

    if settings.disable_llm or not settings.openai_api_key:
        if chunks:
            yield json.dumps(
                {
                    "type": "message",
                    "content": f"Most relevant context (LLM disabled):\n\n{chunks[0][0][:800]}...",
                    "used_provider": "context-only",
                },
            ).encode("utf-8") + b"\n"
        else:
            msg = (
                "No relevant company context found."
                if settings.company_only_mode
                else "LLM disabled and no context was retrieved."
            )
            yield json.dumps(
                {
                    "type": "message",
                    "content": msg,
                    "used_provider": "context-only",
                },
            ).encode("utf-8") + b"\n"
        yield json.dumps({"type": "done", "citations": citations}).encode("utf-8") + b"\n"
        return

    client = OpenAI(api_key=settings.openai_api_key)
    system_content = (
        "You are an internal assistant. When company context is provided, use it and ground all"
        " claims in that context with accurate citations. If no relevant company context is"
        " available, answer from general knowledge helpfully and concisely."
        if not settings.company_only_mode
        else "You are an internal company assistant. Use only the provided company context when"
        " answering. If there is no relevant context, politely say you don't have company"
        " information on that topic and do not answer from general knowledge."
    )
    messages = [
        {
            "role": "system",
            "content": system_content,
        }
    ]
    messages.extend(_format_history(history))
    prompt = f"Question: {question}"
    if context:
        prompt += f"\n\nContext:\n{context}"
    else:
        if settings.company_only_mode:
            prompt += "\n\nNo relevant company context was found."
    messages.append({"role": "user", "content": prompt})
    if settings.company_only_mode and not context:
        yield json.dumps(
            {
                "type": "message",
                "content": "I couldn't find relevant company files for that question.",
                "used_provider": "context-only",
            }
        ).encode("utf-8") + b"\n"
        yield json.dumps({"type": "done", "citations": citations}).encode("utf-8") + b"\n"
        return
    stream = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0,
        stream=True,
    )

    for event in stream:
        choice = event.choices[0]
        text = getattr(choice.delta, "content", None)
        if text:
            yield json.dumps({"type": "message", "content": text, "used_provider": "openai"}).encode("utf-8") + b"\n"

    yield json.dumps({"type": "done", "citations": citations, "used_provider": "openai"}).encode("utf-8") + b"\n"

"""NVIDIA Insight — Q&A app over Jensen Huang's last-6-months speeches, served on DGX Spark.

Fine-tuned Nemotron-3-Nano-4B (LoRA, NeMo AutoModel) + hybrid RAG (Nemotron-3-Embed-1B + BM25).
Run: uvicorn app.server:app --host 0.0.0.0 --port 8600
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import llm
from app.retriever import Retriever

HERE = Path(__file__).resolve().parent
STATE: dict = {}

SYSTEM_PROMPT = (
    "You are NVIDIA Insight, an assistant that answers questions about NVIDIA and its CEO Jensen Huang, "
    "grounded in his speeches, keynotes, earnings calls and interviews from the last six months. "
    "Answer precisely, cite the event and date when relevant, and say so if something is not covered."
)
RAG_INSTRUCTIONS = (
    "Use the numbered sources below when they are relevant and cite them inline as [1], [2], ... "
    "If the sources do not cover the question, answer from what you learned from Jensen Huang's recent speeches "
    "and say that no source excerpt confirms it. Keep answers concise and factual."
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("DISABLE_RAG", "false").lower() != "true":
        STATE["retriever"] = Retriever()
    yield


app = FastAPI(title="NVIDIA Insight (Jensen Huang Q&A)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI-style messages (user/assistant turns)")
    use_rag: bool = True
    top_k: int = 6
    max_tokens: int = 700
    temperature: float = 0.3


def _format_sources(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h.get('title') or 'Untitled'} — {h.get('date') or 'n/a'} — {h.get('url') or ''}"
        lines.append(f"{head}\n{h['text']}")
    return "\n\n".join(lines)


def _build_messages(req: ChatRequest) -> tuple[list[dict], list[dict]]:
    history = [m for m in req.messages if m.get("role") in ("user", "assistant")][-8:]
    question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    hits: list[dict] = []
    system = SYSTEM_PROMPT
    if req.use_rag and "retriever" in STATE and question:
        hits = STATE["retriever"].search(question, top_k=req.top_k)
        if hits:
            system = f"{SYSTEM_PROMPT}\n\n{RAG_INSTRUCTIONS}\n\nSources:\n{_format_sources(hits)}"
    return [{"role": "system", "content": system}, *history], hits


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/health")
async def api_health():
    h = await llm.health()
    h["rag"] = "retriever" in STATE
    return h


@app.get("/api/sources")
async def api_sources():
    if "retriever" not in STATE:
        return {"n_docs": 0, "docs": []}
    return STATE["retriever"].stats()


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    messages, hits = _build_messages(req)

    async def gen():
        yield "data: " + json.dumps({"type": "sources", "sources": [
            {k: h.get(k) for k in ("title", "url", "date", "source_type", "chunk_id", "score")} | {"snippet": h["text"][:300]} for h in hits
        ]}) + "\n\n"
        try:
            async for piece in llm.chat_stream(messages, max_tokens=req.max_tokens, temperature=req.temperature):
                yield "data: " + json.dumps({"type": "token", "content": piece}) + "\n\n"
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps({"type": "error", "error": f"LLM backend error: {e}"}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/answer")
async def api_answer(req: ChatRequest):
    """Non-streaming variant (used by the evaluation script and simple integrations)."""
    messages, hits = _build_messages(req)
    text = await llm.chat(messages, max_tokens=req.max_tokens, temperature=req.temperature)
    return {"answer": text, "sources": [{k: h.get(k) for k in ("title", "url", "date", "chunk_id", "score")} for h in hits]}

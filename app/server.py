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
    "You are NVIDIA Insight, an analyst-grade assistant that answers questions about NVIDIA and its CEO Jensen Huang, "
    "grounded in his speeches, keynotes, earnings calls and interviews from the last six months. "
    "Answer precisely, name the event and date for every claim you attribute to Jensen, and say so if something is not covered."
)

ANSWER_FORMAT = (
    "Write a structured analyst briefing. Use exactly these markdown section headings, in this order, and write each "
    "section as one or two full paragraphs of prose (no bullet lists):\n"
    "### Summary — the direct answer in 2-3 sentences, with the event and date.\n"
    "### Technical view — what was actually announced or explained: architectures, products, numbers, how it works, what changed versus the previous generation.\n"
    "### Business view — why it matters commercially: revenue, demand, margins, guidance, market position, partners, competitive implications, and what Jensen's framing signals about NVIDIA's strategy.\n"
    "### Real-world use cases — how an enterprise, developer, or government would concretely apply this today (e.g. sovereign AI factories, agentic enterprise software, robotics/physical AI, drug discovery, telecom, manufacturing), described as specific scenarios.\n"
    "### Who is already using it — named companies, labs, countries or partners that the numbered sources explicitly say have adopted it, with the benefit they report and a citation for each name. Copy names exactly as they appear in the sources; do not add descriptions of a company that the sources do not give. If the sources name no adopter, write exactly: 'The retrieved sources do not name specific adopters.' and stop.\n"
    "### Insights — 2-3 sentences of non-obvious takeaways: what to watch, risks, second-order effects, or how this connects to Jensen's broader theses (AI factories, the five-layer cake, extreme co-design, agentic and physical AI).\n"
    "Cite the numbered sources inline as [n] after the sentences they support. Be specific and quantitative wherever the material allows; never invent figures, customers or dates."
)

RAG_INSTRUCTIONS = (
    "Use the numbered sources below as your primary evidence and cite them inline as [1], [2], ... after the sentences they support. "
    "If the sources do not cover part of the question, draw on what you learned from Jensen Huang's recent speeches "
    "and state that no source excerpt confirms it."
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
    top_k: int = 8
    max_tokens: int = 1400
    temperature: float = 0.2
    structured: bool = Field(True, description="Analyst briefing format (Summary / Technical / Business / Use cases / Adopters / Insights)")


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
        if req.structured:
            # Second pass aimed at the "Who is already using it" section: pull passages that name
            # customers/partners/deployments so the model has evidence instead of guessing.
            seen = {h["chunk_id"] for h in hits}
            extra = STATE["retriever"].search(
                f"companies, partners and customers named as deploying or adopting this: {question}", top_k=10)
            extra = [h for h in extra if h["chunk_id"] not in seen][:4]
            hits += extra
        if hits:
            system = f"{system}\n\n{RAG_INSTRUCTIONS}\n\nSources:\n{_format_sources(hits)}"
    # The 4B model follows formatting instructions far more reliably when they sit in the user turn
    # (next to the question) than in the system prompt, so append the briefing format there.
    if req.structured and history and history[-1]["role"] == "user":
        history = [*history[:-1], {"role": "user", "content": f"{history[-1]['content']}\n\n{ANSWER_FORMAT}"}]
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

"""NVIDIA Insight — Q&A app over Jensen Huang's last-6-months speeches, served on DGX Spark.

Two Nemotron generators behind one hybrid RAG index (Nemotron-3-Embed-1B + BM25 with a source-quality prior):
  * expert — Nemotron-3-Super 120B-A12B (Ollama): analyst-grade, strictly cited briefings
  * fast   — Nemotron-3-Nano-4B LoRA fine-tuned on the corpus (Ollama/vLLM): quick answers
Run: uvicorn app.server:app --host 0.0.0.0 --port 8600
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
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
log = logging.getLogger("insight")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

SYSTEM_PROMPT = """You are NVIDIA Insight, a senior analyst who covers NVIDIA and its CEO Jensen Huang. You answer only from the evidence provided: numbered source excerpts from Jensen's keynotes, earnings calls, interviews and NVIDIA press coverage from the last six months.

Rules:
- Every factual sentence must be supported by a source and end with its citation(s) like [3] or [2][5]. Prefer transcripts, earnings calls and NVIDIA's own publications over third-party commentary or live-stream chatter; ignore excerpts that are irrelevant, or that are commentators' opinions rather than Jensen's or NVIDIA's words.
- Never invent numbers, names, dates, products or customers. If the evidence does not cover part of a section, say in one sentence what is not covered rather than filling space.
- Quote Jensen's exact words briefly where they are vivid and specific.
- Do not recycle generic NVIDIA talking points (AI factories, five-layer cake, extreme co-design, etc.) unless a source for THIS question actually says them, and never repeat material from earlier answers in the conversation unless asked.
- Be concrete and quantitative: numbers, dates, product names, counterparties. Fiscal-year labels: NVIDIA's FY2027 runs Feb 2026–Jan 2027."""

BRIEFING_FORMAT = """Answer format — a structured expert briefing in markdown, written as prose paragraphs (no bullet lists), using exactly these six headings in this order:
### Summary
the direct answer in 3-4 sentences with event and date.
### Technical view
what was announced or explained, how it works, specifications, what changed versus the previous generation.
### Business view
revenue, demand, guidance, margins, pricing, partners, competitive position, and what Jensen's framing signals.
### Real-world use cases
concrete scenarios described in the sources (who would use it for what); label anything that is your own inference as such.
### Who is already using it
only companies, countries or institutions the sources name, each with a citation and the reported benefit; if none, write "The sources do not name specific adopters for this topic."
### Insights
3-5 sentences of expert, non-obvious takeaways specific to this question — implications, risks, what to watch next, contradictions between sources.
Every factual sentence ends with its source citation(s) in square brackets, e.g. [3] or [2][7]. An answer without the six headings or without bracketed citations is unacceptable."""

SHORT_FORMAT = ("Answer format — a direct, well-organised answer of one to three prose paragraphs, no headings; every factual "
                "sentence ends with its source citation(s) in square brackets, e.g. [3] or [2][7].")

CITE_RE = re.compile(r"\[(\d{1,2})\]")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("DISABLE_RAG", "false").lower() != "true":
        STATE["retriever"] = Retriever()
    yield


app = FastAPI(title="NVIDIA Insight (Jensen Huang Q&A)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI-style messages (user/assistant turns)")
    mode: str | None = Field(None, description="'expert' (Nemotron-3-Super) or 'fast' (fine-tuned Nemotron-3-Nano-4B)")
    use_rag: bool = True
    top_k: int = 8
    max_tokens: int = 1500
    temperature: float = 0.2
    structured: bool = Field(True, description="Six-section analyst briefing; false = short direct answer")


def _format_sources(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h.get('title') or 'Untitled'} — {h.get('date') or 'n/a'} — {h.get('source_type')} — {h.get('url') or ''}"
        lines.append(f"{head}\n{h['text']}")
    return "\n\n".join(lines)


FOLLOWUP_RE = re.compile(r"^(tell me more|more on|more about|what about|and |also |elaborate|expand|go deeper|why|how so|can you|could you|its |their |that |this |it )", re.I)
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b|\d")


def _conversation_context(messages: list[dict]) -> tuple[list, str, str, str]:
    """Return (prior turns as compact notes, current question, retrieval query).

    Earlier assistant briefings are NOT replayed verbatim: a long previous answer that begins with "### Summary"
    primes the model to copy it instead of answering the new question. Instead the model gets a short note of what
    was asked and the gist of each answer. For short/pronoun-heavy follow-ups the previous question is prepended to
    the retrieval query so the right passages are found."""
    hist = [m for m in messages if m.get("role") in ("user", "assistant")]
    question = next((m["content"] for m in reversed(hist) if m["role"] == "user"), "")
    prior = hist[: max(0, len(hist) - 1)]
    notes, prev_q = [], None
    pairs = []
    for m in prior:
        if m["role"] == "user":
            prev_q = m["content"]
        elif prev_q is not None:
            pairs.append((prev_q, m.get("content") or ""))
            prev_q = None
    for q, a in pairs[-4:]:
        body = re.sub(r"#{2,3} [^\n]*", " ", a)
        body = CITE_RE.sub("", body)
        gist = " ".join(body.split())[:260]
        notes.append(f"- Q: {q.strip()[:160]}\n  A (gist): {gist}…")
    context = ("Conversation so far (for context only — answer ONLY the new question below and do not repeat these answers):\n"
               + "\n".join(notes)) if notes else ""
    last_q = pairs[-1][0] if pairs else None
    # follow-up = starts like one, or is short and names no entity (so it must lean on the previous question)
    q_body = question.strip()
    has_entity = ENTITY_RE.search(q_body[1:]) is not None
    followup = FOLLOWUP_RE.match(q_body) is not None or (len(q_body.split()) < 7 and not has_entity)
    rquery = f"{last_q} {question}" if (last_q and followup) else question
    return [], question, rquery, context


def _build_messages(req: ChatRequest) -> tuple[list[dict], list[dict]]:
    _, question, rquery, context = _conversation_context(req.messages)
    hits: list[dict] = []
    if req.use_rag and "retriever" in STATE and question:
        hits = STATE["retriever"].search(question, top_k=req.top_k)
        if rquery != question:  # follow-up: add passages found with the previous question as context
            seen = {h["chunk_id"] for h in hits}
            extra = [h for h in STATE["retriever"].search(rquery, top_k=req.top_k) if h["chunk_id"] not in seen]
            hits = (hits[: max(4, req.top_k - 4)] + extra)[: req.top_k]
    fmt = BRIEFING_FORMAT if req.structured else SHORT_FORMAT
    # A per-request nonce at the very start of the prompt defeats llama.cpp/Ollama prefix caching, which with hybrid
    # Mamba models (Nemotron-3-Super) can replay a stale recurrent state; sources change every turn anyway.
    system = f"Request {uuid.uuid4().hex[:12]}\n\n{SYSTEM_PROMPT}"
    if context:
        system = f"{system}\n\n{context}"
    if hits:
        user = f"Sources:\n{_format_sources(hits)}\n\nNew question: {question}\n\n{fmt}"
    else:
        user = (f"New question: {question}\n\n(No source excerpts were retrieved; answer from Jensen Huang's recent public remarks "
                f"and say that no excerpt confirms it.)\n\n{fmt}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], hits


def _public(h: dict) -> dict:
    return {k: h.get(k) for k in ("title", "url", "date", "source_type", "chunk_id", "score", "quality")} | {"snippet": h["text"][:300]}


def _cited(text: str, n: int) -> list[int]:
    return sorted({int(m) for m in CITE_RE.findall(text) if 1 <= int(m) <= n})


def _finalize(text: str, hits: list[dict]) -> tuple[str, list[int]]:
    """Guarantee citations: keep the model's [n] marks, attribute uncited sentences by embedding similarity."""
    if hits and "retriever" in STATE:
        try:
            return STATE["retriever"].attribute(text, hits)
        except Exception:  # noqa: BLE001
            pass
    return text, _cited(text, len(hits))


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
    return STATE["retriever"].stats() if "retriever" in STATE else {"n_chunks": 0, "n_docs": 0, "docs": []}


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Server-sent events: status → sources → tokens → cited → done."""
    mode = req.mode or llm.DEFAULT_MODE
    cfg = llm.mode_cfg(mode)

    async def gen():
        t0 = time.time()
        yield "data: " + json.dumps({"type": "status", "text": f"Retrieving evidence… ({cfg['label']})"}) + "\n\n"
        messages, hits = _build_messages(req)
        yield "data: " + json.dumps({"type": "sources", "sources": [_public(h) for h in hits]}) + "\n\n"
        yield "data: " + json.dumps({"type": "status", "text": f"Reading {len(hits)} sources and writing the briefing — the first tokens take ~{'20-40' if mode == 'expert' else '3-5'} s"}) + "\n\n"
        text = ""
        try:
            async for piece in llm.chat_stream(messages, mode=mode, max_tokens=req.max_tokens, temperature=req.temperature):
                text += piece
                yield "data: " + json.dumps({"type": "token", "content": piece}) + "\n\n"
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps({"type": "error", "error": f"LLM backend error: {e}"}) + "\n\n"
        text, cited = _finalize(text, hits)
        q = next((m["content"] for m in reversed(req.messages) if m.get("role") == "user"), "")
        log.info("chat mode=%s turns=%d %.1fs q=%r sources=%s answer=%r", mode, len(req.messages), time.time() - t0, q[:120],
                 [h["chunk_id"] for h in hits[:4]], text[:160])
        yield "data: " + json.dumps({"type": "final", "content": text}) + "\n\n"
        yield "data: " + json.dumps({"type": "cited", "ids": cited, "seconds": round(time.time() - t0, 1), "mode": mode, "model": cfg["model"]}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/answer")
async def api_answer(req: ChatRequest):
    """Non-streaming variant (used by the evaluation script and simple integrations)."""
    mode = req.mode or llm.DEFAULT_MODE
    t0 = time.time()
    messages, hits = _build_messages(req)
    text = await llm.chat(messages, mode=mode, max_tokens=req.max_tokens, temperature=req.temperature)
    text, cited = _finalize(text, hits)
    return {"answer": text, "mode": mode, "model": llm.mode_cfg(mode)["model"], "seconds": round(time.time() - t0, 1),
            "cited": cited, "sources": [_public(h) | {"cited": (i + 1) in cited} for i, h in enumerate(hits)]}

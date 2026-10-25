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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import llm
from app.feedback import FeedbackStore
from app.nvidia_snippets import catalog as snippet_catalog
from app.retriever import Retriever

HERE = Path(__file__).resolve().parent
STATE: dict = {}
log = logging.getLogger("insight")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

SYSTEM_PROMPT = """You are NVIDIA Insight, a senior analyst who covers NVIDIA and its CEO Jensen Huang. You answer only from the evidence provided: numbered source excerpts from Jensen's keynotes, earnings calls, interviews and NVIDIA press coverage from the last six months.

Rules:
- Everything factual must come from the sources, but cite sparingly so the text stays readable: one citation like [3] at the end of the sentence that carries each paragraph's key fact, at most two citations per paragraph, never a stack like [1][2][5]. Prefer transcripts, earnings calls and NVIDIA's own publications over third-party commentary or live-stream chatter; ignore excerpts that are irrelevant, or that are commentators' opinions rather than Jensen's or NVIDIA's words.
- Never invent numbers, names, dates, products or customers. If the evidence does not cover part of a section, say in one sentence what is not covered rather than filling space.
- Quote Jensen's exact words briefly where they are vivid and specific.
- Do not recycle generic NVIDIA talking points (AI factories, five-layer cake, extreme co-design, etc.) unless a source for THIS question actually says them, and never repeat material from earlier answers in the conversation unless asked.
- Be concrete and quantitative: numbers, dates, product names, counterparties. Fiscal-year labels: NVIDIA's FY2027 runs Feb 2026–Jan 2027."""

BRIEFING_FORMAT = """Answer format — a structured expert briefing in markdown that is easy to read on screen. Use exactly these six section headings (### level) in this order, each followed by a blank line:
### Summary
the direct answer in 3 sentences with event and date.
### Technical view
### Business view
### Real-world use cases
### Who is already using it
### Insights

Inside Technical view, Business view, Real-world use cases and Insights, split the content into 2-4 short sub-topics. Each sub-topic is its own paragraph of 2-4 sentences, separated by a blank line, that begins with a bold lead-in label naming the sub-topic, e.g. "**Compute fabric.** ..." or "**Guidance.** ...". Never use bullet lists or numbered lists. Keep paragraphs short (under 90 words).
Who is already using it: only companies, countries or institutions the sources name, with the reported benefit; if none, write "The sources do not name specific adopters for this topic."
Insights: expert, non-obvious takeaways specific to this question — implications, risks, what to watch next, contradictions between sources.

Citations: one [n] at the end of the sentence with the key fact of each paragraph, at most two per paragraph, never stacked like [1][4][6]. An answer without the six headings or without any bracketed citations is unacceptable.
"""

ARCH_FORMAT = """Seventh section — "### Architecture & code" — this question is about a system, platform, chip, network, software stack or developer workflow, so include this section after Insights. If, and only if, the sources truly describe no system or workflow, omit the heading entirely (never write a placeholder sentence under it). It has two parts, both mandatory when the section is present:
1. A Mermaid diagram in a ```mermaid fenced block (flowchart LR or TD, 6-14 nodes) showing the architecture or data flow exactly as the sources describe it; label nodes with the real component names (GPUs, CPUs, NVLink, Spectrum-X, BlueField, software layers) and always put node labels in double quotes, e.g. A["DGX Spark (64 GB)"] --> B["NVLink switch"]. Do not invent components that the sources do not mention.
2. A short, runnable example (10-30 lines, one ```python or ```bash fenced block) using the most relevant NVIDIA library or tool — e.g. TensorRT-LLM / NVIDIA Dynamo / NIM for inference, NeMo for training, cuDF/RAPIDS for data, Isaac Sim / Cosmos for robotics, CUDA Python (numba/cupy) for kernels, DGX Spark playbooks — preceded by one sentence naming the library and why it fits the question. Take it from the approved snippet catalog below and adapt only names/prompts/sizes; never invent commands or functions."""

TECH_RE = re.compile(r"\b(architecture|fine-?tun\w*|inference|serving|software stack|full.?stack|chips?|gpus?|cpus?|racks?|nvl\d*|nvlink|spectrum-?x|infiniband|bluefield|connectx|cuda|tensorrt|dynamo|nim|nemo|nemotron|cosmos|isaac|omniverse|openshell|hyperion|rubin|blackwell|vera|grace|dgx|spark|rtx|jetson|thor|drive\s+(agx|thor|os)|kv.?cache|disaggregat\w*|robots?|robotics|physical ai|digital twins?|humanoids?|developers?|sdk|library|libraries|api|framework)\b", re.I)

SHORT_FORMAT = ("Answer format — a direct, well-organised answer of one to three short prose paragraphs separated by blank lines, "
                "no headings or bullets; one citation like [3] at the end of each paragraph's key sentence, at most two per paragraph.")

CITE_RE = re.compile(r"\[(\d{1,2})\]")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("DISABLE_RAG", "false").lower() != "true":
        STATE["retriever"] = Retriever()
        STATE["feedback"] = FeedbackStore(STATE["retriever"].embed_queries)
    yield


app = FastAPI(title="NVIDIA Insight (Jensen Huang Q&A)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI-style messages (user/assistant turns)")
    mode: str | None = Field(None, description="'expert' (Nemotron-3-Super) or 'fast' (fine-tuned Nemotron-3-Nano-4B)")
    use_rag: bool = True
    top_k: int = 8
    max_tokens: int = 2600
    temperature: float = 0.2
    structured: bool = Field(True, description="Six-section analyst briefing; false = short direct answer")


def _format_sources(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        head = (f"[{i}] {h.get('title') or 'Untitled'} — published {h.get('date') or 'n/a'} "
                f"({'video captions' if h.get('source_type') == 'youtube' else 'web'}; the event itself may be earlier) — {h.get('url') or ''}")
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
    guidance, weights = ("", {})
    if "feedback" in STATE and question:
        guidance, weights = STATE["feedback"].guidance(question)
    if req.use_rag and "retriever" in STATE and question:
        hits = STATE["retriever"].search(question, top_k=req.top_k, weights=weights)
        if rquery != question:  # follow-up: add passages found with the previous question as context
            seen = {h["chunk_id"] for h in hits}
            extra = [h for h in STATE["retriever"].search(rquery, top_k=req.top_k, weights=weights) if h["chunk_id"] not in seen]
            hits = (hits[: max(4, req.top_k - 4)] + extra)[: req.top_k]
    fmt = BRIEFING_FORMAT if req.structured else SHORT_FORMAT
    if req.structured and TECH_RE.search(question):
        fmt = f"{fmt}\n\n{ARCH_FORMAT}\n\n{snippet_catalog()}"
    # A per-request nonce at the very start of the prompt defeats llama.cpp/Ollama prefix caching, which with hybrid
    # Mamba models (Nemotron-3-Super) can replay a stale recurrent state; sources change every turn anyway.
    system = f"Request {uuid.uuid4().hex[:12]}\n\n{SYSTEM_PROMPT}"
    if context:
        system = f"{system}\n\n{context}"
    if guidance:
        system = f"{system}\n\nUser feedback on earlier answers (reinforcement):\n{guidance}"
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


_STACK_RE = re.compile(r"(\[\d{1,2}\]){3,}")
_LEADIN_RE = re.compile(r"^\*\*([^*\n]{2,60})\*\*:?\s*$", re.M)


_SHAPES = {"[[": "]]", "((": "))", "[": "]", "{": "}", "(": ")"}
_OPEN_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)(\[\[|\(\(|\[|\{|\()")
_MERMAID_FENCE_RE = re.compile(r"(```mermaid\n)(.*?)(```)", re.S)


def _fix_mermaid(block: str) -> str:
    """Quote node labels so parentheses, colons, commas etc. inside them do not break the Mermaid parser."""
    def fix_line(line: str) -> str:
        out, pos = [], 0
        while True:
            m = _OPEN_RE.search(line, pos)
            if not m:
                out.append(line[pos:]); break
            o, c = m.group(2), _SHAPES[m.group(2)]
            start = m.end()
            end = -1
            k = line.find(c, start)
            while k != -1:  # closer must end the token
                if k + len(c) >= len(line) or line[k + len(c)] in " \t-=;|":
                    end = k; break
                k = line.find(c, k + 1)
            if end == -1:
                out.append(line[pos:m.end()]); pos = m.end(); continue
            label = line[start:end]
            if label.strip() and not label.strip().startswith('"') and re.search(r"[()\[\]{}|;:,<>/#&\"']", label):
                label = '"' + label.strip().replace('"', "'") + '"'
            out.append(line[pos:m.start()] + m.group(1) + o + label + c)
            pos = end + len(c)
        return "".join(out)
    return "\n".join(fix_line(l) for l in block.split("\n"))


def _tidy(text: str) -> str:
    """Readability pass: ≤2 citations per stack, bold lead-in lines merged with their paragraph, blank line after headings."""
    text = "\n".join(l.rstrip() for l in text.split("\n"))
    text = re.sub(r"(\[\d{1,2}\])(?:\s*,\s*|\s+)(?=\[\d{1,2}\])", r"\1", text)
    text = _STACK_RE.sub(lambda m: "".join(f"[{n}]" for n in CITE_RE.findall(m.group(0))[:2]), text)
    text = re.sub(r"(\[\d{1,2}\])(?:\s*,\s*|\s+)(?=\[\d{1,2}\])", r"\1", text)
    text = _LEADIN_RE.sub(lambda m: f"**{m.group(1).rstrip('.:')}.**", text)
    text = re.sub(r"\*\*([^*\n]{2,60}?):\*\*", r"**\1.**", text)
    text = re.sub(r"\*\*([^*\n]{2,60}\.)\*\*\n(?=\S)", r"**\1** ", text)
    text = re.sub(r"^(#{2,4} .+)\n(?=\S)", r"\1\n\n", text, flags=re.M)
    text = re.sub(r"(?<=\S)\n(?=#{2,4} )", "\n\n", text)
    text = re.sub(r"(?<=\S)\n(?=\*\*[^*\n]{2,60}\*\*)", "\n\n", text)  # each bold lead-in starts its own paragraph
    text = _MERMAID_FENCE_RE.sub(lambda m: m.group(1) + _fix_mermaid(m.group(2)) + m.group(3), text)
    m = re.search(r"^### Architecture & code\s*$", text, re.M)
    if m and "```" not in text[m.end():]:  # placeholder section without diagram/code → drop it
        text = text[: m.start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _finalize(text: str, hits: list[dict]) -> tuple[str, list[int]]:
    text = _tidy(text)
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


# ---------------------------------------------------------------- feedback (reinforcement layer)
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str = Field(..., pattern="^(good|bad)$")
    comment: str | None = None
    mode: str | None = None
    model: str | None = None
    cited_chunk_ids: list[str] = []
    sources: list[dict] = []


@app.post("/api/feedback")
async def api_feedback(req: FeedbackRequest):
    if "feedback" not in STATE:
        raise HTTPException(503, "feedback store unavailable (RAG disabled)")
    rec = STATE["feedback"].add(req.model_dump())
    log.info("feedback %s mode=%s q=%r comment=%r", req.rating, req.mode, req.question[:100], (req.comment or "")[:100])
    return {"ok": True, "id": rec["id"], "stats": STATE["feedback"].stats()}


@app.get("/api/feedback/stats")
async def api_feedback_stats():
    return STATE["feedback"].stats() if "feedback" in STATE else {"n": 0}


# ---------------------------------------------------------------- PDF export
class PdfRequest(BaseModel):
    question: str
    answer: str
    sources: list[dict] = []
    cited: list[int] = []
    mode: str | None = None
    model: str | None = None
    diagrams_svg: list[str] = Field([], description="Rendered Mermaid SVGs from the browser, in order of appearance")


def _md_to_html(md: str, svgs: list[str]) -> str:
    """Small markdown → HTML for the briefing (headings, paragraphs, fenced code, mermaid → supplied SVG)."""
    import html as _h
    out, para, fence, lang, buf, svg_i = [], [], False, "", [], 0
    def flush():
        nonlocal para
        if para:
            text = _h.escape(" ".join(para))
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            text = CITE_RE.sub(lambda m: f'<span class="cite">[{m.group(1)}]</span>', text)
            out.append(f"<p>{text}</p>"); para = []
    for line in md.split("\n"):
        if line.strip().startswith("```"):
            if not fence:
                flush(); fence, lang, buf = True, line.strip()[3:].strip().lower(), []
            else:
                if lang == "mermaid" and svg_i < len(svgs):
                    out.append(f'<div class="diagram">{svgs[svg_i]}</div>'); svg_i += 1
                elif lang == "mermaid":
                    out.append(f'<pre class="code"><code>{_h.escape(chr(10).join(buf))}</code></pre>')
                else:
                    out.append(f'<pre class="code"><code>{_h.escape(chr(10).join(buf))}</code></pre>')
                fence = False
            continue
        if fence:
            buf.append(line); continue
        if line.startswith("#"):
            flush(); out.append(f"<h3>{_h.escape(line.lstrip('#').strip())}</h3>")
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return "\n".join(out)


@app.post("/api/pdf")
async def api_pdf(req: PdfRequest):
    import html as _h
    from weasyprint import HTML
    body = _md_to_html(req.answer, req.diagrams_svg)
    src_rows = "".join(
        f'<li class="{"cited" if (i + 1) in req.cited else "uncited"}"><b>[{i + 1}]</b> {_h.escape(s.get("title") or "Untitled")} '
        f'<span class="meta">({_h.escape(str(s.get("date") or ""))}, {_h.escape(str(s.get("source_type") or ""))})</span><br>'
        f'<a href="{_h.escape(s.get("url") or "")}">{_h.escape(s.get("url") or "")}</a></li>'
        for i, s in enumerate(req.sources))
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    @page {{ size: A4; margin: 18mm 16mm; @bottom-right {{ content: "NVIDIA Insight · page " counter(page) " of " counter(pages); font-size: 9px; color: #777; }} }}
    body {{ font-family: Helvetica, Arial, sans-serif; color: #111; font-size: 11px; line-height: 1.5; }}
    h1 {{ font-size: 18px; margin: 0 0 4px; }} h1 span {{ color: #76b900; }}
    .q {{ font-size: 14px; font-weight: 600; margin: 10px 0 14px; padding: 10px 12px; background: #f3f7ec; border-left: 4px solid #76b900; }}
    h3 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #5a8a00; border-top: 1px solid #ddd; padding-top: 8px; margin: 14px 0 4px; }}
    p {{ margin: 0 0 9px; }} p b:first-child {{ display: block; color: #000; margin-bottom: 1px; }} .cite {{ color: #5a8a00; font-weight: 600; font-size: 8px; vertical-align: super; }}
    pre.code {{ background: #f6f6f6; border: 1px solid #ddd; padding: 8px; font-size: 9px; white-space: pre-wrap; }}
    .diagram svg {{ max-width: 100%; height: auto; }} .diagram {{ margin: 8px 0; }}
    .meta {{ color: #666; font-size: 9px; }} ol.sources {{ padding-left: 18px; }} ol.sources li {{ margin-bottom: 6px; }} li.uncited {{ color: #888; }}
    .foot {{ margin-top: 16px; color: #666; font-size: 9px; border-top: 1px solid #ddd; padding-top: 6px; }}
    </style></head><body>
    <h1><span>NVIDIA</span> Insight — Jensen Huang Q&amp;A briefing</h1>
    <div class="meta">Generated {time.strftime('%Y-%m-%d %H:%M')} on NVIDIA DGX Spark · model {_h.escape(req.model or '')} ({_h.escape(req.mode or '')}) · corpus: Jensen Huang public remarks, Apr–Oct 2026</div>
    <div class="q">{_h.escape(req.question)}</div>
    {body}
    <h3>Sources</h3><ol class="sources">{src_rows}</ol>
    <div class="foot">Citations [n] refer to the numbered sources. Content is machine-generated from public transcripts and articles; verify against the linked sources. Nemotron models under the NVIDIA Open Model License.</div>
    </body></html>"""
    pdf = HTML(string=doc, base_url=str(HERE)).write_pdf()
    fname = re.sub(r"[^a-zA-Z0-9]+", "-", req.question)[:60].strip("-") or "briefing"
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="nvidia-insight-{fname}.pdf"'})

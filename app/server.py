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

import httpx
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import grounding, llm
from app.feedback import FeedbackStore
from app.nvidia_snippets import catalog as snippet_catalog, pick as pick_snippet
from app.retriever import Retriever
from app import verify as verifier

HERE = Path(__file__).resolve().parent
STATE: dict = {}
log = logging.getLogger("insight")
# What the reader is told the evidence spans when the question names something the sources do not cover.
CORPUS_SPAN = os.environ.get("CORPUS_SPAN", "Jensen Huang's keynotes, interviews, earnings calls and NVIDIA news, May–Oct 2026")
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

The final section of every answer (after any Architecture & code section) is:
### Use-case inspiration
3-5 short paragraphs, each with a bold lead-in naming a TYPE of use case that THIS topic specifically inspires — derive the types from what the sources say about this topic (capabilities, customers, industries, Jensen's own examples), not from a generic list, and make them different from one another: who would build it, what the user gets, and which NVIDIA capability from the sources enables it. Cite where the sources support it; where it is your extrapolation, write "(inspiration)" at the end of the paragraph instead of a citation.
Close the section with a use-case map as a Mermaid mindmap in a ```mermaid fenced block, preceded by a one-line caption starting with "Figure:": root((topic)) in the centre, one branch per use-case type (3-5, the same names as the bold lead-ins above, at most 4 words each), 2 leaves per branch naming an industry or scenario in at most 3 words. Mindmap labels are plain words only — no brackets, parentheses, colons, quotes or slashes — and indentation is two spaces per level (syntax illustration only, do not reuse its content):
```mermaid
mindmap
  root((Topic))
    First use-case type
      Industry or scenario
      Industry or scenario
    Second use-case type
      Industry or scenario
```

Citations: one [n] at the end of the sentence with the key fact of each paragraph, at most two per paragraph, never stacked like [1][4][6]. An answer without the headings or without any bracketed citations is unacceptable.
"""

ARCH_FORMAT = """Seventh section — "### Architecture & code" — this question is about a system, platform, chip, network, software stack or developer workflow, so include this section immediately after Insights and before Use-case inspiration. If, and only if, the sources truly describe no system or workflow, omit the heading entirely (never write a placeholder sentence under it). It has two parts, both mandatory when the section is present:
1. An architecture diagram in a ```mermaid fenced block, preceded by a one-line caption sentence starting with "Figure:". Rules for an expert-quality diagram:
- `flowchart TB`; 8-14 nodes grouped into 2-4 `subgraph` layers stacked top to bottom in the direction data flows (e.g. Data & networking → Compute → Software stack → Applications), with the nodes of one layer side by side (at most 4 per layer); every subgraph gets an id and a quoted title: subgraph compute["Compute layer"].
- Node labels: a short name, then `<br/>` and one spec line of at most 5 words, e.g. A["Rubin GPU<br/>HBM4, 72 per rack"]. No more than 2 lines, no sentences, labels always in double quotes.
- Edges show flow or dependency, one direction only (never A-->B and B-->A), labelled with the protocol or quantity when the sources give it: -->|NVLink 72|, -->|KV cache|. 8-14 edges, no decorative edges.
- Use only components the sources mention; if a layer has nothing in the sources, leave it out.
- Finish the block with exactly these style lines and assign every node to one class (hw = compute/memory/cooling hardware, net = networking/interconnect, sw = software/frameworks/models, app = applications/customers):
classDef hw fill:#e8f5d0,stroke:#5a8f00,stroke-width:2px,color:#0f1115
classDef net fill:#dcefff,stroke:#2b8ee6,stroke-width:2px,color:#0f1115
classDef sw fill:#ece4ff,stroke:#8a5cf6,stroke-width:2px,color:#0f1115
classDef app fill:#fff0d6,stroke:#f59e0b,stroke-width:2px,color:#0f1115
class A,B hw
(continue `class ... net/sw/app` lines for the remaining nodes).
1b. If the sources describe a request/data flow (prefill → decode, sensor → policy, training → deployment), add a `sequenceDiagram` with 3-5 participants and 4-8 messages; if they give a dated roadmap (announcement, sampling, production, shipments), add a `timeline` with 3-6 dated entries instead. Each extra diagram in its own ```mermaid block, preceded by its own "Figure:" caption line. Skip 1b if the sources give neither.
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
    max_tokens: int = 3200
    temperature: float = 0.2
    structured: bool = Field(True, description="Six-section analyst briefing; false = short direct answer")
    verify: bool | None = Field(None, description="Agentic claim verification pass (default: on for expert mode)")
    notes: str | None = Field(None, description="Verifier feedback from a previous draft (used by the NAT refine loop)")


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


def _ground(question: str, hits: list[dict], top_k: int, weights: dict | None = None) -> tuple[list[dict], dict]:
    """Coverage check + targeted re-query. The corpus may well cover the named thing even when the semantic top-k
    missed it ("GTC Paris", "Hyperion"): pull passages that literally mention each missing term (BM25-only ranking,
    so the rare word wins over the common one), put them first, and re-check. Whatever is still missing is reported."""
    cov = grounding.coverage(question, hits) if hits else {"terms": [], "missing": []}
    if cov["missing"] and "retriever" in STATE:
        seen = {h["chunk_id"] for h in hits}
        extra: list[dict] = []
        for term in cov["missing"]:
            for h in STATE["retriever"].search(term, top_k=8, weights=weights, lexical=True):
                if h["chunk_id"] not in seen and grounding.mentions(term, h):
                    extra.append(h)
                    seen.add(h["chunk_id"])
        if extra:
            hits = (extra[:4] + hits)[: max(top_k, 6)]
            cov = grounding.coverage(question, hits)
    return hits, cov


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
    if req.notes:
        system = f"{system}\n\nFact-checker feedback on your previous draft — these sentences were NOT supported by the sources; do not repeat them, state only what the sources say:\n{req.notes}"
    hits, cov = _ground(question, hits, req.top_k, weights)
    if cov["missing"] or cov.get("thin"):
        system = f"{system}\n\n{grounding.prompt_instruction(cov, CORPUS_SPAN)}"
    if hits:
        user = f"Sources:\n{_format_sources(hits)}\n\nNew question: {question}\n\n{fmt}"
    else:
        user = (f"New question: {question}\n\n(No source excerpts were retrieved; answer from Jensen Huang's recent public remarks "
                f"and say that no excerpt confirms it.)\n\n{fmt}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], hits, cov


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
    text = _MERMAID_FENCE_RE.sub(lambda m: m.group(1) + (_fix_mermaid(m.group(2)) if re.match(r"\s*(flowchart|graph)\b", m.group(2)) else m.group(2)) + m.group(3), text)
    text = re.sub(r"^\*?Figure:\*?\s*$\n?", "", text, flags=re.M)  # empty captions
    m = re.search(r"^### Architecture & code\s*$", text, re.M)
    if m and "```" not in text[m.end():]:  # placeholder section without diagram/code → drop it
        text = text[: m.start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _ensure_code_example(text: str, question: str) -> str:
    """Guarantee the Architecture & code section carries a verified NVIDIA code example."""
    m = re.search(r"^### Architecture & code\s*$", text, re.M)
    if not m:
        return text
    end = re.search(r"^### Use-case inspiration\s*$", text[m.end():], re.M)
    section_end = m.end() + end.start() if end else len(text)
    if re.search(r"```(python|bash|sh)\b", text[m.end():section_end]):
        return text
    key, sn = pick_snippet(question + " " + text[: m.start()])
    lib = {"tensorrt_llm": "TensorRT-LLM", "dynamo": "NVIDIA Dynamo", "nim": "NVIDIA NIM", "nemo_automodel": "NeMo AutoModel",
           "cuda_python": "CUDA Python (CuPy / Numba)", "rapids": "RAPIDS cuDF", "isaac_sim": "Isaac Sim", "cosmos": "NVIDIA Cosmos",
           "nemotron_hf": "Nemotron with Hugging Face Transformers", "omniverse": "OpenUSD / Omniverse"}[key]
    block = (f"\n\nCode example (verified snippet from NVIDIA's public documentation, chosen for this topic): {lib} — "
             f"fits because it is the documented path for {sn['when'].split(',')[0].strip()}.\n\n```{sn['lang']}\n{sn['code']}\n```\n\n")
    return text[:section_end].rstrip() + block + text[section_end:]


async def _verify(text: str, hits: list[dict], mode: str, question: str = "", cov: dict | None = None) -> tuple[str, dict | None]:
    """Second agentic step: judge every cited sentence against its sources and correct/remove unsupported ones."""
    judge_input, claims = verifier.build_judge_input(text, hits, question=question, missing=((cov or {}).get("missing") or []) + [t["term"] for t in (cov or {}).get("thin") or []])
    if not claims:
        return text, None
    msgs = [{"role": "system", "content": verifier.JUDGE_PROMPT}, {"role": "user", "content": judge_input}]
    try:
        verdicts = await llm.chat(msgs, mode=mode, max_tokens=1800, temperature=0.0)
    except Exception as e:  # noqa: BLE001
        log.warning("verification failed: %s", e)
        return text, {"error": str(e)}
    new_text, report = verifier.apply_verdicts(text, claims, verdicts)
    report["model"] = llm.mode_cfg(mode)["model"]
    return new_text, report


def _should_verify(req: "ChatRequest", mode: str) -> bool:
    return req.verify if req.verify is not None else (mode == "expert" and req.structured)


def _last_question(req: "ChatRequest") -> str:
    q = next((m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"), "")
    return q if req.structured and TECH_RE.search(q) else ""


def _finalize(text: str, hits: list[dict], question: str = "") -> tuple[str, list[int]]:
    text = _tidy(text)
    if question:
        text = _ensure_code_example(text, question)
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
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            h["nat"] = (await c.get(f"{NAT_URL}/health")).status_code == 200
    except Exception:  # noqa: BLE001
        h["nat"] = False
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
        messages, hits, cov = _build_messages(req)
        q = next((m["content"] for m in reversed(req.messages) if m.get("role") == "user"), "")
        note = grounding.reader_note(cov, CORPUS_SPAN)
        yield "data: " + json.dumps({"type": "sources", "sources": [_public(h) for h in hits], "coverage": cov | {"note": note}}) + "\n\n"
        yield "data: " + json.dumps({"type": "status", "text": f"Reading {len(hits)} sources and writing the briefing — the first tokens take ~{'20-40' if mode == 'expert' else '3-5'} s"}) + "\n\n"
        text = ""
        try:
            async for piece in llm.chat_stream(messages, mode=mode, max_tokens=req.max_tokens, temperature=req.temperature):
                text += piece
                yield "data: " + json.dumps({"type": "token", "content": piece}) + "\n\n"
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps({"type": "error", "error": f"LLM backend error: {e}"}) + "\n\n"
        text, cited = _finalize(text, hits, _last_question(req))
        text = grounding.ensure_disclaimer(text, cov, CORPUS_SPAN)
        yield "data: " + json.dumps({"type": "final", "content": text}) + "\n\n"
        report = None
        if _should_verify(req, mode) and hits:
            yield "data: " + json.dumps({"type": "status", "text": "Verifying every cited claim against its sources (agentic check, ~40-70 s)…"}) + "\n\n"
            text, report = await _verify(text, hits, mode, question=q, cov=cov)
            cited = sorted({int(n) for n in CITE_RE.findall(text) if 1 <= int(n) <= len(hits)})
            yield "data: " + json.dumps({"type": "final", "content": text}) + "\n\n"
        log.info("chat mode=%s turns=%d %.1fs q=%r sources=%s missing=%s verify=%s answer=%r", mode, len(req.messages), time.time() - t0, q[:120],
                 [h["chunk_id"] for h in hits[:4]], cov["missing"], {k: report[k] for k in ("score", "corrected", "removed")} if report and "score" in report else None, text[:160])
        yield "data: " + json.dumps({"type": "cited", "ids": cited, "seconds": round(time.time() - t0, 1), "mode": mode, "model": cfg["model"], "verification": report, "coverage": cov}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/answer")
async def api_answer(req: ChatRequest):
    """Non-streaming variant (used by the evaluation script and simple integrations)."""
    mode = req.mode or llm.DEFAULT_MODE
    t0 = time.time()
    messages, hits, cov = _build_messages(req)
    q = next((m["content"] for m in reversed(req.messages) if m.get("role") == "user"), "")
    text = await llm.chat(messages, mode=mode, max_tokens=req.max_tokens, temperature=req.temperature)
    text, cited = _finalize(text, hits, _last_question(req))
    text = grounding.ensure_disclaimer(text, cov, CORPUS_SPAN)
    report = None
    if _should_verify(req, mode) and hits:
        text, report = await _verify(text, hits, mode, question=q, cov=cov)
        cited = sorted({int(n) for n in CITE_RE.findall(text) if 1 <= int(n) <= len(hits)})
    return {"answer": text, "mode": mode, "model": llm.mode_cfg(mode)["model"], "seconds": round(time.time() - t0, 1),
            "cited": cited, "verification": report, "coverage": cov | {"note": grounding.reader_note(cov, CORPUS_SPAN)},
            "sources": [_public(h) | {"cited": (i + 1) in cited} for i, h in enumerate(hits)]}


# ---------------------------------------------------------------- precise mode: NeMo Agent Toolkit workflow (draft -> verify -> refine)
NAT_URL = os.environ.get("NAT_URL", "http://127.0.0.1:8601")


@app.post("/api/precise")
async def api_precise(req: ChatRequest):
    """SSE wrapper around the NAT workflow server so the UI gets the same event stream as /api/chat."""
    question = next((m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"), "")

    async def gen():
        t0 = time.time()
        yield "data: " + json.dumps({"type": "status", "text": "Precise mode · NeMo Agent Toolkit workflow: retrieving evidence…"}) + "\n\n"
        hits = STATE["retriever"].search(question, top_k=req.top_k) if "retriever" in STATE else []
        hits, cov = _ground(question, hits, req.top_k)
        yield "data: " + json.dumps({"type": "sources", "sources": [_public(h) for h in hits], "coverage": cov | {"note": grounding.reader_note(cov, CORPUS_SPAN)}}) + "\n\n"
        yield "data: " + json.dumps({"type": "status", "text": "Agent loop running: draft → fact-check every cited sentence → refine until ≥90% verified (typically 2-5 min)…"}) + "\n\n"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(1200.0, connect=5.0)) as c:
                r = await c.post(f"{NAT_URL}/generate", json={"question": question})
            r.raise_for_status()
            out = r.json()
            val = out.get("value") if isinstance(out, dict) else out
            data = json.loads(val) if isinstance(val, str) and val.strip().startswith("{") else {"answer": str(val), "cited": [], "verification": None, "round": None, "model": "nat"}
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps({"type": "error", "error": f"NAT workflow error: {e}"}) + "\n\n"
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"
            return
        text = data["answer"]
        cited = sorted({int(n) for n in CITE_RE.findall(text) if 1 <= int(n) <= len(hits)})
        rep = data.get("verification") or {}
        if rep:
            rep["rounds"] = data.get("round")
        log.info("precise %.1fs q=%r rounds=%s verify=%s", time.time() - t0, question[:120], data.get("round"),
                 {k: rep.get(k) for k in ("score", "corrected", "removed")} if rep else None)
        yield "data: " + json.dumps({"type": "final", "content": text}) + "\n\n"
        yield "data: " + json.dumps({"type": "cited", "ids": cited, "seconds": round(time.time() - t0, 1), "mode": "precise",
                                     "model": f"{data.get('model', '')} · NAT loop ×{data.get('round')}", "verification": rep or None}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- tool endpoints for the NeMo Agent Toolkit workflow (nat/)
class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 8


@app.post("/api/retrieve")
async def api_retrieve(req: RetrieveRequest):
    if "retriever" not in STATE:
        raise HTTPException(503, "RAG disabled")
    hits = STATE["retriever"].search(req.question, top_k=req.top_k)
    hits, cov = _ground(req.question, hits, req.top_k)
    return {"sources": [_public(h) | {"text": h.get("text", "")} for h in hits], "coverage": cov}


class VerifyRequest(BaseModel):
    answer: str
    sources: list[dict] = Field(..., description="[{title,date,text,...}] in citation order")
    mode: str | None = None
    question: str = ""


@app.post("/api/verify")
async def api_verify(req: VerifyRequest):
    mode = req.mode or llm.DEFAULT_MODE
    sources = []
    for src in req.sources:  # resolve full passage text (API responses carry only snippets)
        full = STATE["retriever"].chunk(src.get("chunk_id", "")) if "retriever" in STATE and src.get("chunk_id") else None
        sources.append({**src, "text": (full or {}).get("text") or src.get("text") or src.get("snippet") or ""})
    cov = grounding.coverage(req.question, sources) if req.question else None
    text, report = await _verify(req.answer, sources, mode, question=req.question, cov=cov)
    if cov:
        text = grounding.ensure_disclaimer(text, cov, CORPUS_SPAN)
    cited = sorted({int(n) for n in CITE_RE.findall(text) if 1 <= int(n) <= len(req.sources)})
    return {"answer": text, "cited": cited, "verification": report}


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
    diagrams_svg: list[str | dict] = Field([], description="Rendered Mermaid SVGs (print theme) from the browser, in order of appearance; str or {svg,label,legend}")


_LEGEND_HTML = ('<div class="legend"><span class="hw">Compute / memory hardware</span><span class="net">Networking / interconnect</span>'
                '<span class="sw">Software / models</span><span class="app">Applications / customers</span></div>')


def _md_to_html(md: str, svgs: list) -> str:
    """Small markdown → HTML for the briefing (headings, paragraphs, fenced code, mermaid → supplied SVG)."""
    import html as _h
    out, para, fence, lang, buf, svg_i = [], [], False, "", [], 0
    def flush():
        nonlocal para
        if para:
            text = _h.escape(" ".join(para))
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            text = CITE_RE.sub(lambda m: f'<span class="cite">[{m.group(1)}]</span>', text)
            cls = ' class="caption"' if text.startswith("Figure:") else ""
            out.append(f"<p{cls}>{text}</p>"); para = []
    for line in md.split("\n"):
        if line.strip().startswith("```"):
            if not fence:
                flush(); fence, lang, buf = True, line.strip()[3:].strip().lower(), []
            else:
                if lang == "mermaid" and svg_i < len(svgs):
                    d = svgs[svg_i]; svg_i += 1
                    if isinstance(d, str):
                        d = {"svg": d, "label": "", "legend": False}
                    label = f'<div class="dlabel">{_h.escape(d.get("label") or "Diagram")}</div>'
                    if d.get("png", "").startswith("data:image/png;base64,"):
                        w, hgt = float(d.get("width") or 800), float(d.get("height") or 600)
                        css = "width:100%" if w >= hgt * 0.8 else f"width:{min(100, max(45, 100 * w / hgt * 1.2)):.0f}%"
                        fig = f'<img class="dimg" src="{d["png"]}" style="{css}" alt="{_h.escape(d.get("label") or "diagram")}">'
                    else:
                        fig = d.get("svg", "")
                    out.append(f'<div class="diagram">{label}{fig}{_LEGEND_HTML if d.get("legend") else ""}</div>')
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
    .diagram {{ margin: 10px 0 14px; padding: 10px; border: 1px solid #d5dbcf; border-radius: 6px; background: #fff; page-break-inside: avoid; }}
    .diagram svg {{ width: 100%; max-width: 100%; height: auto; display: block; }}
    .diagram img.dimg {{ display: block; margin: 0 auto; max-width: 100%; max-height: 225mm; height: auto; }}
    .dlabel {{ font-size: 9px; letter-spacing: 1px; text-transform: uppercase; color: #5a8a00; font-weight: 700; margin-bottom: 6px; }}
    .legend {{ margin-top: 8px; font-size: 9px; color: #333; }} .legend span {{ display: inline-block; padding: 2px 8px; margin-right: 6px; border-radius: 10px; border: 1.5px solid; }}
    .legend .hw {{ background: #e8f5d0; border-color: #5a8f00; }} .legend .net {{ background: #dcefff; border-color: #2b8ee6; }} .legend .sw {{ background: #ece4ff; border-color: #8a5cf6; }} .legend .app {{ background: #fff0d6; border-color: #f59e0b; }}
    p.caption {{ font-style: italic; color: #444; margin: 6px 0 2px; }}
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

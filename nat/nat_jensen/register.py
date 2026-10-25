"""NeMo Agent Toolkit functions for NVIDIA Insight.

Tools (each wraps an endpoint of the running app so the toolkit, the UI and the eval harness share one code path):
  jensen_retrieve  question -> JSON list of ranked, quality-weighted source passages (hybrid RAG)
  jensen_draft     question (+ optional fact-checker notes) -> cited six-section briefing (expert or fast model)
  jensen_verify    {answer, sources} -> answer with unsupported sentences corrected/removed + verification report

Workflow:
  jensen_precise_briefing  question -> verified briefing. Draft, verify every cited sentence against its sources,
  and if the groundedness score is below `target_score`, re-draft with the fact-checker's notes (up to `max_rounds`),
  keeping the best-scoring version. The final text carries a one-line verification footer.
"""
from __future__ import annotations

import json
import logging

import httpx
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)
TIMEOUT = httpx.Timeout(900.0, connect=10.0)


class JensenRetrieveConfig(FunctionBaseConfig, name="jensen_retrieve"):
    app_url: str = Field("http://127.0.0.1:8600", description="Base URL of the NVIDIA Insight app")
    top_k: int = 8


@register_function(config_type=JensenRetrieveConfig)
async def jensen_retrieve(config: JensenRetrieveConfig, _builder: Builder):
    async def _fn(question: str) -> str:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{config.app_url}/api/retrieve", json={"question": question, "top_k": config.top_k})
        r.raise_for_status()
        return json.dumps(r.json()["sources"])

    yield FunctionInfo.from_fn(_fn, description="Retrieve the most relevant passages from Jensen Huang's keynotes, earnings calls and NVIDIA coverage (hybrid dense+BM25 with source-quality prior). Input: the question. Output: JSON list of sources with title, date, url and text.")


class JensenDraftConfig(FunctionBaseConfig, name="jensen_draft"):
    app_url: str = Field("http://127.0.0.1:8600")
    mode: str = Field("expert", description="expert (Nemotron-3-Super) or fast (fine-tuned Nemotron-Nano-4B)")
    top_k: int = 8


@register_function(config_type=JensenDraftConfig)
async def jensen_draft(config: JensenDraftConfig, _builder: Builder):
    async def _fn(question: str, notes: str | None = None) -> str:
        body = {"messages": [{"role": "user", "content": question}], "mode": config.mode, "top_k": config.top_k, "verify": False, "notes": notes}
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{config.app_url}/api/answer", json=body)
        r.raise_for_status()
        j = r.json()
        return json.dumps({"answer": j["answer"], "sources": j["sources"], "model": j["model"], "question": question, "coverage": j.get("coverage")})

    yield FunctionInfo.from_fn(_fn, description="Draft a cited, six-section analyst briefing for the question using retrieved sources. Optional notes = fact-checker feedback to respect. Output: JSON {answer, sources, model, question, coverage}.")


class JensenVerifyConfig(FunctionBaseConfig, name="jensen_verify"):
    app_url: str = Field("http://127.0.0.1:8600")
    mode: str = "expert"


@register_function(config_type=JensenVerifyConfig)
async def jensen_verify(config: JensenVerifyConfig, _builder: Builder):
    async def _fn(draft_json: str) -> str:
        d = json.loads(draft_json)
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{config.app_url}/api/verify", json={"answer": d["answer"], "sources": d["sources"], "mode": config.mode, "question": d.get("question", "")})
        r.raise_for_status()
        return json.dumps(r.json())

    yield FunctionInfo.from_fn(_fn, description="Fact-check a drafted briefing sentence by sentence against the sources it cites; corrects or removes unsupported sentences. Input: JSON {answer, sources}. Output: JSON {answer, cited, verification{score, checked, corrected, removed, changes}}.")


class JensenPreciseConfig(FunctionBaseConfig, name="jensen_precise_briefing"):
    app_url: str = Field("http://127.0.0.1:8600")
    mode: str = "expert"
    top_k: int = 8
    target_score: float = Field(0.9, description="Stop refining once this share of sentences is verified")
    max_rounds: int = Field(2, description="Draft/verify rounds")
    footer: bool = True
    output_json: bool = Field(False, description="Return JSON {answer, cited, verification, round, model} instead of text")


@register_function(config_type=JensenPreciseConfig)
async def jensen_precise_briefing(config: JensenPreciseConfig, _builder: Builder):
    async def _fn(question: str) -> str:
        best: dict | None = None
        notes: str | None = None
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            for rnd in range(1, config.max_rounds + 1):
                body = {"messages": [{"role": "user", "content": question}], "mode": config.mode, "top_k": config.top_k, "verify": False, "notes": notes}
                d = (await c.post(f"{config.app_url}/api/answer", json=body)).raise_for_status().json()
                v = (await c.post(f"{config.app_url}/api/verify", json={"answer": d["answer"], "sources": d["sources"], "mode": config.mode, "question": question})).raise_for_status().json()
                rep = v.get("verification") or {}
                score = rep.get("score", 1.0)
                log.info("round %d: score=%.2f corrected=%s removed=%s", rnd, score, rep.get("corrected"), rep.get("removed"))
                cand = {"answer": v["answer"], "sources": d["sources"], "cited": v["cited"], "verification": rep, "round": rnd, "model": d["model"]}
                if best is None or score >= best["verification"].get("score", 0):
                    best = cand
                if score >= config.target_score or not rep.get("changes"):
                    break
                notes = "\n".join(f"- {ch['from']}" for ch in rep["changes"][:12])
        assert best is not None
        if config.output_json:
            return json.dumps({k: best[k] for k in ("answer", "cited", "verification", "round", "model")})
        text = best["answer"]
        if config.footer:
            r = best["verification"]
            text += (f"\n\n---\nVerification: {round(100 * r.get('score', 1.0))}% of {r.get('checked', 0)} cited sentences confirmed against sources "
                     f"({r.get('corrected', 0)} corrected, {r.get('removed', 0)} removed) after {best['round']} round(s) · model {best['model']}")
        return text

    yield FunctionInfo.from_fn(_fn, description="Answer a question about NVIDIA / Jensen Huang with a verified analyst briefing: retrieve, draft, fact-check every cited sentence, refine until grounded.")

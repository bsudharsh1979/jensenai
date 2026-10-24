"""Thin OpenAI-compatible chat client (works with vLLM and Ollama's /v1 endpoint)."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "nemotron-nano-4b-jensen")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "none")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))
# Nemotron-3 Nano reasons by default; the fine-tune was trained in non-thinking mode.
ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "false").lower() == "true"


def _payload(messages: list[dict], stream: bool, max_tokens: int, temperature: float) -> dict:
    p = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},  # honoured by vLLM
    }
    if "11434" in LLM_BASE_URL:  # Ollama: thinking toggle is a top-level flag
        p["think"] = ENABLE_THINKING
    return p


async def chat_stream(messages: list[dict], max_tokens: int = 700, temperature: float = 0.3) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        async with client.stream(
            "POST", f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json=_payload(messages, True, max_tokens, temperature),
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if not piece and not ENABLE_THINKING:
                    # vLLM's nano_v3 reasoning parser mislabels streamed answer tokens as
                    # reasoning_content when thinking is disabled (non-streaming is correct).
                    piece = delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    yield piece


async def chat(messages: list[dict], max_tokens: int = 700, temperature: float = 0.3) -> str:
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        r = await client.post(f"{LLM_BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                              json=_payload(messages, False, max_tokens, temperature))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{LLM_BASE_URL}/models", headers={"Authorization": f"Bearer {LLM_API_KEY}"})
            ok = r.status_code == 200
            models = [m.get("id") for m in r.json().get("data", [])] if ok else []
            return {"ok": ok, "base_url": LLM_BASE_URL, "model": LLM_MODEL, "available_models": models}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "base_url": LLM_BASE_URL, "model": LLM_MODEL, "error": str(e)}

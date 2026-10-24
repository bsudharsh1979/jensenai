"""LLM client with two backends and two answer modes.

* ``ollama``  — native /api/chat (lets us set num_ctx, keep_alive and the Nemotron thinking toggle).
* ``openai``  — OpenAI-compatible /v1/chat/completions (vLLM, or Ollama's /v1 shim).

Modes (``MODES``): "expert" uses the strongest local Nemotron (Nemotron-3-Super 120B-A12B) for analyst-grade
briefings; "fast" uses the LoRA fine-tuned Nemotron-3-Nano-4B for quick answers. Each mode is an env-configurable
(backend, base_url, model) triple so the app can be re-pointed without code changes.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "900"))
LLM_API_KEY = os.environ.get("LLM_API_KEY", "none")
KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "24h")   # keep the expert model resident between questions

MODES: dict[str, dict] = {
    "expert": {
        "label": "Expert — Nemotron-3-Super 120B-A12B",
        "backend": os.environ.get("EXPERT_BACKEND", "ollama"),
        "base_url": os.environ.get("EXPERT_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        "model": os.environ.get("EXPERT_MODEL", "nemotron-3-super:latest"),
        "num_ctx": int(os.environ.get("EXPERT_NUM_CTX", "32768")),
        "thinking": os.environ.get("EXPERT_THINKING", "false").lower() == "true",
        "send_think": True,   # model declares the thinking capability; toggle it explicitly
    },
    "fast": {
        "label": "Fast — Nemotron-3-Nano-4B (LoRA fine-tuned on Jensen's speeches)",
        "backend": os.environ.get("FAST_BACKEND", "ollama"),
        "base_url": os.environ.get("FAST_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        "model": os.environ.get("FAST_MODEL", "nemotron-nano-4b-jensen"),
        "num_ctx": int(os.environ.get("FAST_NUM_CTX", "16384")),
        "thinking": False,
        "send_think": False,  # non-thinking is baked into the Modelfile template
    },
}
DEFAULT_MODE = os.environ.get("DEFAULT_MODE", "expert")


def mode_cfg(mode: str | None) -> dict:
    return MODES.get(mode or DEFAULT_MODE, MODES[DEFAULT_MODE])


def _ollama_payload(cfg: dict, messages: list[dict], stream: bool, max_tokens: int, temperature: float) -> dict:
    p = {
        "model": cfg["model"], "messages": messages, "stream": stream, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": temperature, "top_p": 0.95, "num_predict": max_tokens, "num_ctx": cfg["num_ctx"],
                    "repeat_penalty": 1.05},
    }
    if cfg.get("send_think"):
        p["think"] = cfg["thinking"]
    return p


def _same(a: str, b: str) -> bool:
    return a == b or a.split(":")[0] == b.split(":")[0]


async def _ensure_exclusive(client: httpx.AsyncClient, cfg: dict) -> None:
    """DGX Spark unified memory cannot hold Nemotron-3-Super (~91 GB) and another model next to the other
    workloads on the box, so evict any *other* mode's model from Ollama before loading this one."""
    if cfg["backend"] != "ollama":
        return
    try:
        loaded = [m["name"] for m in (await client.get(f"{cfg['base_url']}/api/ps")).json().get("models", [])]
    except Exception:  # noqa: BLE001
        return
    ours = {c["model"] for c in MODES.values() if c["backend"] == "ollama"}
    for name in loaded:
        if not _same(name, cfg["model"]) and any(_same(name, o) for o in ours):
            await client.post(f"{cfg['base_url']}/api/generate", json={"model": name, "keep_alive": 0})


def _openai_payload(cfg: dict, messages: list[dict], stream: bool, max_tokens: int, temperature: float) -> dict:
    return {"model": cfg["model"], "messages": messages, "stream": stream, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": cfg["thinking"]}}


async def chat_stream(messages: list[dict], mode: str | None = None, max_tokens: int = 1800,
                      temperature: float = 0.2) -> AsyncIterator[str]:
    cfg = mode_cfg(mode)
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        await _ensure_exclusive(client, cfg)
        if cfg["backend"] == "ollama":
            async with client.stream("POST", f"{cfg['base_url']}/api/chat",
                                     json=_ollama_payload(cfg, messages, True, max_tokens, temperature)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if d.get("error"):
                        raise RuntimeError(d["error"])
                    piece = d.get("message", {}).get("content")
                    if piece:
                        yield piece
                    if d.get("done"):
                        break
        else:
            async with client.stream("POST", f"{cfg['base_url']}/chat/completions",
                                     headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                                     json=_openai_payload(cfg, messages, True, max_tokens, temperature)) as r:
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
                    if not piece and not cfg["thinking"]:
                        # vLLM's nano_v3 reasoning parser mislabels streamed answer tokens when thinking is off
                        piece = delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        yield piece


async def chat(messages: list[dict], mode: str | None = None, max_tokens: int = 1800, temperature: float = 0.2) -> str:
    return "".join([p async for p in chat_stream(messages, mode, max_tokens, temperature)])


async def health() -> dict:
    out = {"default_mode": DEFAULT_MODE, "modes": {}}
    ok_any = False
    async with httpx.AsyncClient(timeout=5) as client:
        for name, cfg in MODES.items():
            info = {"label": cfg["label"], "backend": cfg["backend"], "base_url": cfg["base_url"], "model": cfg["model"]}
            try:
                if cfg["backend"] == "ollama":
                    r = await client.get(f"{cfg['base_url']}/api/tags")
                    models = [m["name"] for m in r.json().get("models", [])]
                    info["available"] = any(m == cfg["model"] or m.split(":")[0] == cfg["model"] for m in models)
                    loaded = await client.get(f"{cfg['base_url']}/api/ps")
                    info["loaded"] = any(m["name"] == cfg["model"] or m["name"].split(":")[0] == cfg["model"]
                                         for m in loaded.json().get("models", []))
                else:
                    r = await client.get(f"{cfg['base_url']}/models", headers={"Authorization": f"Bearer {LLM_API_KEY}"})
                    info["available"] = any(m.get("id") == cfg["model"] for m in r.json().get("data", []))
                    info["loaded"] = info["available"]
            except Exception as e:  # noqa: BLE001
                info["available"] = False
                info["error"] = str(e)
            ok_any = ok_any or info.get("available", False)
            out["modes"][name] = info
    out["ok"] = out["modes"].get(DEFAULT_MODE, {}).get("available", False) or ok_any
    # legacy fields used by the UI status bar
    d = out["modes"][DEFAULT_MODE]
    out.update({"base_url": d["base_url"], "model": d["model"]})
    return out

#!/usr/bin/env python3
"""Benchmark a generator backend for the NVIDIA Insight app: time-to-first-token, decode tok/s and end-to-end
latency on a realistic briefing prompt, optionally with N concurrent requests.

Usage:
  scripts/12_bench_backend.py --backend ollama --base-url http://127.0.0.1:11434 --model nemotron-3-super:latest
  scripts/12_bench_backend.py --backend openai --base-url http://127.0.0.1:8001/v1 --model nemotron-3-super --concurrency 3
"""
import argparse
import asyncio
import json
import time

import httpx

PROMPT = ("You are an NVIDIA industry analyst. Using what you know about Jensen Huang's 2026 keynotes, write a briefing "
          "on Vera Rubin NVL72 with headings Summary, Technical view, Business view and Insights. Each section has 2-3 "
          "short paragraphs starting with a bold lead-in. Keep it under 500 words.")


async def one(client: httpx.AsyncClient, args, idx: int) -> dict:
    pad = ""
    if args.prompt_pad:  # emulate the app's retrieved passages (~6K tokens of context) to measure realistic prefill
        pad = "\n\nBackground passages (for context only):\n" + " ".join(
            f"Passage {i}: NVIDIA's data-center platform revenue, Vera Rubin NVL72 rack, Blackwell Ultra, "
            f"Spectrum-X networking and Dynamo inference software were discussed at GTC {2026 - i % 2}."
            for i in range(args.prompt_pad))
    msgs = [{"role": "system", "content": "You write concise, well-structured analyst briefings."},
            {"role": "user", "content": PROMPT + pad + ("" if idx == 0 else f" (variant {idx})")}]
    t0 = time.perf_counter()
    first = None
    n_tokens = 0
    text = []
    if args.backend == "ollama":
        payload = {"model": args.model, "messages": msgs, "stream": True, "think": False,
                   "options": {"num_predict": args.max_tokens, "temperature": 0.2, "num_ctx": 32768}}
        async with client.stream("POST", f"{args.base_url}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                piece = d.get("message", {}).get("content", "")
                if piece and first is None:
                    first = time.perf_counter()
                text.append(piece)
                if d.get("done"):
                    n_tokens = d.get("eval_count", 0)
                    prompt_tokens = d.get("prompt_eval_count", 0)
                    break
    else:
        payload = {"model": args.model, "messages": msgs, "stream": True, "max_tokens": args.max_tokens,
                   "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False},
                   "stream_options": {"include_usage": True}}
        prompt_tokens = 0
        async with client.stream("POST", f"{args.base_url}/chat/completions", json=payload,
                                 headers={"Authorization": "Bearer none"}) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("usage"):
                    n_tokens = d["usage"].get("completion_tokens", 0)
                    prompt_tokens = d["usage"].get("prompt_tokens", 0)
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece and first is None:
                        first = time.perf_counter()
                    text.append(piece)
    t1 = time.perf_counter()
    ttft = (first or t1) - t0
    decode_s = t1 - (first or t0)
    return {"idx": idx, "ttft_s": round(ttft, 2), "total_s": round(t1 - t0, 1), "prompt_tokens": prompt_tokens,
            "completion_tokens": n_tokens, "decode_tok_s": round(n_tokens / decode_s, 1) if decode_s > 0 else None,
            "chars": sum(len(p) for p in text)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai"], required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--prompt-pad", type=int, default=0, help="number of ~30-token filler passages to prepend (0 = short prompt)")
    args = ap.parse_args()
    async with httpx.AsyncClient(timeout=900) as client:
        for run in range(args.runs):
            t0 = time.perf_counter()
            res = await asyncio.gather(*[one(client, args, i) for i in range(args.concurrency)])
            wall = time.perf_counter() - t0
            total_tok = sum(r["completion_tokens"] for r in res)
            for r in res:
                print(json.dumps(r))
            print(json.dumps({"run": run, "backend": args.backend, "model": args.model, "concurrency": args.concurrency,
                              "wall_s": round(wall, 1), "aggregate_tok_s": round(total_tok / wall, 1),
                              "mean_ttft_s": round(sum(r["ttft_s"] for r in res) / len(res), 2),
                              "mean_decode_tok_s": round(sum(r["decode_tok_s"] or 0 for r in res) / len(res), 1)}))


if __name__ == "__main__":
    asyncio.run(main())

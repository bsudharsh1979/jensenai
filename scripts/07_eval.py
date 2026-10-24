#!/usr/bin/env python
"""Evaluate base vs fine-tuned Nemotron on the held-out Q&A split (data/sft/val.jsonl).

Both models are queried through OpenAI-compatible endpoints (vLLM / Ollama) with thinking disabled,
so the comparison isolates the effect of the LoRA SFT. Metrics:
  * token-level F1 and ROUGE-L (vs. the teacher's reference answer)
  * optional LLM-as-judge (Nemotron-3-Super via Ollama) scoring factual agreement 1-5

Usage:
  python scripts/07_eval.py --tuned-url http://127.0.0.1:8000/v1 --tuned-model nemotron-nano-4b-jensen \
                            --base-url  http://127.0.0.1:8001/v1 --base-model  nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
                            [--n 60] [--judge]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SFT, read_jsonl  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "eval"
OUT.mkdir(exist_ok=True)


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def f1(pred: str, ref: str) -> float:
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


async def ask(client: httpx.AsyncClient, url: str, model: str, system: str, q: str, think: bool = False) -> str:
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}],
               "max_tokens": 400, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": think}}
    if "11434" in url:
        payload["think"] = think
    r = await client.post(f"{url}/chat/completions", json=payload, headers={"Authorization": "Bearer none"})
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


JUDGE_PROMPT = """You are grading an answer about NVIDIA / Jensen Huang against a reference answer written from a primary source.
Question: {q}
Reference answer: {ref}
Candidate answer: {cand}
Score the candidate's factual agreement with the reference from 1 (contradicts / fabricates) to 5 (fully consistent and specific).
Reply with only the integer."""


async def judge(client: httpx.AsyncClient, q: str, ref: str, cand: str) -> int | None:
    try:
        r = await client.post(os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434") + "/api/chat", json={
            "model": os.environ.get("TEACHER_MODEL", "nemotron-3-super:latest"), "stream": False, "think": False,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, cand=cand)}],
            "options": {"temperature": 0, "num_predict": 5}}, timeout=300)
        m = re.search(r"[1-5]", r.json()["message"]["content"])
        return int(m.group(0)) if m else None
    except Exception:  # noqa: BLE001
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tuned-model", default="nemotron-nano-4b-jensen")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--judge-only", action="store_true",
                    help="skip generation; judge the answers already in eval/results.jsonl (lets the 86 GB judge run after the vLLM servers are stopped)")
    args = ap.parse_args()

    if args.judge_only:
        prev = json.loads((OUT / "summary.json").read_text())
        systems = {k: (None, v) for k, v in prev["systems"].items()}
        results = read_jsonl(OUT / "results.jsonl")
        async with httpx.AsyncClient(timeout=600) as client:
            sem = asyncio.Semaphore(args.concurrency)

            async def judge_row(row: dict) -> None:
                for name in systems:
                    async with sem:
                        row[name]["judge"] = await judge(client, row["question"], row["reference"], row[name]["answer"])

            await asyncio.gather(*[judge_row(r) for r in results])
        write_outputs(systems, results)
        return

    val = read_jsonl(SFT / "val.jsonl")[: args.n]
    system = val[0]["messages"][0]["content"]
    systems = {"tuned": (args.tuned_url.rstrip("/"), args.tuned_model)}
    if args.base_url:
        systems["base"] = (args.base_url.rstrip("/"), args.base_model)

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient(timeout=600) as client:
        async def one(ex: dict) -> dict:
            q, ref = ex["messages"][1]["content"], ex["messages"][2]["content"]
            row = {"question": q, "reference": ref, "meta": ex.get("meta")}
            for name, (url, model) in systems.items():
                async with sem:
                    try:
                        ans = await ask(client, url, model, system, q)
                    except Exception as e:  # noqa: BLE001
                        ans = f"<error: {e}>"
                row[name] = {"answer": ans, "f1": round(f1(ans, ref), 4), "rougeL": round(rouge_l(ans, ref), 4)}
                if args.judge:
                    row[name]["judge"] = await judge(client, q, ref, ans)
            return row

        results = await asyncio.gather(*[one(ex) for ex in val])
    write_outputs(systems, results)


def write_outputs(systems: dict, results: list) -> None:
    summary = {}
    for name in systems:
        rows = [r[name] for r in results]
        summary[name] = {
            "n": len(rows),
            "f1": round(sum(r["f1"] for r in rows) / len(rows), 4),
            "rougeL": round(sum(r["rougeL"] for r in rows) / len(rows), 4),
        }
        js = [r["judge"] for r in rows if r.get("judge") is not None]
        if js:
            summary[name]["judge_mean"] = round(sum(js) / len(js), 3)
            summary[name]["judge_n"] = len(js)
    (OUT / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    (OUT / "summary.json").write_text(json.dumps({"systems": {k: v[1] for k, v in systems.items()}, "metrics": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

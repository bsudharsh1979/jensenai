#!/usr/bin/env python
"""Synthesize an SFT dataset (OpenAI `messages` JSONL) from curated chunks.

Follows the Nemotron post-training recipe for domain data: a larger Nemotron *teacher*
(NVIDIA-Nemotron-3-Super via Ollama) writes context-grounded Q&A pairs from each chunk;
answers must be supported by the chunk and cite the source event/date so the student
(Nemotron-3-Nano-4B) learns both facts and the "attributed answer" style.

Resumable: progress is appended to data/sft/raw_pairs.jsonl; final train/val split is
written to data/sft/train.jsonl and data/sft/val.jsonl.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

from common import CURATED, SFT, env_int, read_jsonl

OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
TEACHER = os.environ.get("TEACHER_MODEL", "nemotron-3-super:latest")
MAX_CHUNKS = env_int("MAX_CHUNKS", 450)
PAIRS_PER_CHUNK = env_int("PAIRS_PER_CHUNK", 3)
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.08"))
SEED = 1111

SYSTEM_PROMPT = (
    "You are NVIDIA Insight, an assistant that answers questions about NVIDIA and its CEO Jensen Huang, "
    "grounded in his speeches, keynotes, earnings calls and interviews from the last six months. "
    "Answer precisely, cite the event and date when relevant, and say so if something is not covered."
)

TEACHER_PROMPT = """You are building a training set for a small assistant that answers questions about NVIDIA and Jensen Huang.

Source: {title} ({source_type}, published {date})
URL: {url}

<passage>
{text}
</passage>

Write {n} diverse question/answer pairs that a user might ask about NVIDIA, Jensen Huang, NVIDIA's products, strategy, numbers, partners or Jensen's opinions, answerable ONLY from the passage above.
Rules:
- Questions must be self-contained (mention the event/topic, not "the passage").
- Mix question types: factual (numbers, names, dates), explanatory ("why does Jensen say..."), and summarizing.
- Answers: 2-5 sentences, factual, in the assistant's own words, and mention the source event and approximate date when it adds context (e.g. "At GTC Berlin in October 2026, Huang said..."). Do not invent facts.
- Transcripts may contain other speakers (hosts, analysts, executives, guests). Attribute a statement to Jensen Huang ONLY if the passage makes clear he said it; otherwise attribute it to the named speaker, to "NVIDIA", or to "a speaker at <event>".
- The published date is the upload/publication date, not necessarily the event date; prefer the event name over an exact day.
- If the passage is mostly noise or has fewer than {n} good facts, return fewer pairs.
Return ONLY a JSON array like: [{{"question": "...", "answer": "..."}}, ...]"""


def call_teacher(prompt: str, timeout: int = 900) -> str:
    # Reasoning is disabled on the teacher ("think": false): we want compact, grounded pairs,
    # not long thinking traces (Nemotron-3 Super reasons by default).
    r = httpx.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": TEACHER,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.4, "top_p": 0.9, "num_predict": 800, "num_ctx": 4096},
            "keep_alive": "30m",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_pairs(raw: str) -> list[dict]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        # Ollama json mode sometimes wraps: {"pairs": [...]} or {"question":..,"answer":..}
        if "question" in data and "answer" in data:
            data = [data]
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if lists else []
    out = []
    for p in data if isinstance(data, list) else []:
        if not isinstance(p, dict):
            continue
        q, a = str(p.get("question", "")).strip(), str(p.get("answer", "")).strip()
        if len(q) > 15 and len(a) > 40 and "passage" not in q.lower():
            out.append({"question": q, "answer": a})
    return out


def priority(chunk: dict) -> tuple:
    """Speech transcripts first (youtube / transcript pages), NVIDIA first-party second, then press."""
    url = chunk.get("url") or ""
    st = chunk.get("source_type")
    if st == "youtube" or "transcript" in url or "on-demand" in url or "q4cdn" in url:
        tier = 0
    elif "nvidia.com" in url:
        tier = 1
    else:
        tier = 2
    return (tier, chunk.get("date") or "", chunk["chunk_id"])


def main() -> None:
    chunks = read_jsonl(CURATED / "chunks.jsonl")
    if not chunks:
        sys.exit("no chunks; run 03_curate.py first")
    chunks.sort(key=priority)
    # Spread coverage: round-robin across documents so that one long keynote doesn't take every slot.
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    ordered: list[dict] = []
    while len(ordered) < MAX_CHUNKS and any(by_doc.values()):
        for doc_id in list(by_doc):
            if by_doc[doc_id]:
                ordered.append(by_doc[doc_id].pop(0))
            else:
                del by_doc[doc_id]
    ordered = ordered[:MAX_CHUNKS]

    raw_path = SFT / "raw_pairs.jsonl"
    done = {r["chunk_id"] for r in read_jsonl(raw_path)}
    todo = [c for c in ordered if c["chunk_id"] not in done]
    print(f"{len(ordered)} chunks selected, {len(done)} done, {len(todo)} to generate with {TEACHER}")

    t0 = time.time()
    with open(raw_path, "a") as f:
        for i, c in enumerate(todo, 1):
            prompt = TEACHER_PROMPT.format(
                title=c.get("title") or "untitled", source_type=c["source_type"], date=c.get("date") or "2026",
                url=c.get("url") or "", text=c["text"], n=PAIRS_PER_CHUNK,
            )
            try:
                raw = call_teacher(prompt)
                pairs = parse_pairs(raw)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] teacher error: {e}", flush=True)
                time.sleep(5)
                continue
            rec = {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c.get("title"), "url": c.get("url"),
                   "date": c.get("date"), "pairs": pairs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {len(pairs)} pairs  ({el/i:.0f}s/chunk, eta {(len(todo)-i)*el/i/60:.0f} min)", flush=True)

    build_splits()


def build_splits() -> None:
    rows = read_jsonl(SFT / "raw_pairs.jsonl")
    # Only keep pairs whose source chunk survived the (possibly re-run) curation step.
    valid_chunks = {c["chunk_id"] for c in read_jsonl(CURATED / "chunks.jsonl")}
    rows = [r for r in rows if r["chunk_id"] in valid_chunks]
    examples = []
    seen_q = set()
    for r in rows:
        for p in r["pairs"]:
            key = re.sub(r"\W+", " ", p["question"].lower()).strip()
            if key in seen_q:
                continue
            seen_q.add(key)
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p["question"]},
                    {"role": "assistant", "content": p["answer"]},
                ],
                "meta": {"chunk_id": r["chunk_id"], "url": r.get("url"), "title": r.get("title"), "date": r.get("date")},
            })
    random.Random(SEED).shuffle(examples)
    n_val = max(32, int(len(examples) * VAL_FRACTION))
    val, train = examples[:n_val], examples[n_val:]
    for name, rows_ in (("train", train), ("val", val)):
        with open(SFT / f"{name}.jsonl", "w") as f:
            for e in rows_:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"SFT examples: train={len(train)} val={len(val)} (from {len(rows)} chunks)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--splits-only":
        build_splits()
    else:
        main()

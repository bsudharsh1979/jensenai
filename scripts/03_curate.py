#!/usr/bin/env python
"""Curate the raw corpus following NeMo Curator's standard pipeline stages:

  load -> clean/normalize -> language ID -> heuristic quality filters
       -> exact dedup -> fuzzy (near-duplicate) dedup -> domain/time relevance -> chunking

Outputs:
  data/curated/docs.jsonl    one row per kept document
  data/curated/chunks.jsonl  retrieval / SFT-generation chunks (~CHUNK_WORDS words, overlap)
  data/curated/stats.json    filter funnel statistics
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

from langdetect import DetectorFactory, detect
from rapidfuzz.distance import Levenshtein  # noqa: F401  (kept for optional pairwise checks)

from common import CURATED, RAW, env_int, normalize_ws, sha1, window, write_jsonl

DetectorFactory.seed = 0

CHUNK_WORDS = env_int("CHUNK_WORDS", 350)
CHUNK_OVERLAP = env_int("CHUNK_OVERLAP", 60)
MIN_WORDS = env_int("MIN_WORDS", 150)
NEAR_DUP_JACCARD = 0.80

BOILERPLATE = re.compile(
    r"^(subscribe|sign up|sign in|log in|share this|read more|advertisement|cookie|privacy policy|terms of (use|service)|"
    r"all rights reserved|follow us|related articles?|©|photo:|image:|getty images|reuters/|\(reuters\)\s*-?\s*$)",
    re.I,
)
TIMESTAMP = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")
RELEVANCE = re.compile(r"\b(jensen|huang|nvidia)\b", re.I)
# YouTube: keep first-party speech material (keynotes, interviews, calls), drop retail stock commentary.
YT_PERSON = re.compile(r"\b(jensen|huang)\b", re.I)
YT_EVENT = re.compile(r"\b(keynote|gtc|computex|siggraph|earnings call|fireside|interview|conversation|commencement|q&a|town hall|summit|address|speech)\b", re.I)
YT_NOISE = re.compile(r"\b(stock|buy|sell|price target|portfolio|invest(ing|or|ment)?|options|bull|bear|crash|rally|dividend|etf)\b|\$\d|should you", re.I)


def load_raw() -> list[dict]:
    docs = []
    for sub in ("youtube", "web", "manual"):
        for f in sorted((RAW / sub).glob("*.json")):
            try:
                docs.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                pass
    return docs


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or TIMESTAMP.match(s) or BOILERPLATE.match(s):
            continue
        if len(s.split()) <= 2 and not s.endswith((".", "?", "!")):
            # drop nav crumbs / stray labels
            continue
        lines.append(s)
    return normalize_ws("\n".join(lines))


def quality_ok(text: str) -> tuple[bool, str]:
    words = text.split()
    n = len(words)
    if n < MIN_WORDS:
        return False, "too_short"
    mean_len = sum(len(w) for w in words) / n
    if not (3 <= mean_len <= 10):
        return False, "word_len"
    symbols = sum(text.count(c) for c in "#…{}[]<>|\\")
    if symbols / n > 0.1:
        return False, "symbol_ratio"
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        dup_lines = 1 - len(set(lines)) / len(lines)
        if dup_lines > 0.3:
            return False, "dup_lines"
    # repeated 5-gram ratio (transcripts repeat a bit; be lenient)
    grams = [" ".join(words[i : i + 5]) for i in range(0, n - 5)]
    if grams:
        c = Counter(grams)
        rep = sum(v for v in c.values() if v > 1) / len(grams)
        if rep > 0.35:
            return False, "repeated_ngrams"
    return True, "ok"


def shingles(text: str, k: int = 5) -> set[int]:
    """Hashed word k-gram shingles (stride 1 so offset differences between transcripts don't matter)."""
    w = re.findall(r"\w+", text.lower())
    return {hash(" ".join(w[i : i + k])) for i in range(0, max(1, len(w) - k))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def chunk(text: str, doc_id: str) -> list[dict]:
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    chunks, cur, cur_n, idx = [], [], 0, 0
    for s in sents:
        wn = len(s.split())
        if cur_n + wn > CHUNK_WORDS and cur:
            chunks.append({"chunk_id": f"{doc_id}#{idx}", "doc_id": doc_id, "text": " ".join(cur)})
            idx += 1
            # overlap: keep trailing sentences up to CHUNK_OVERLAP words
            keep, kn = [], 0
            for t in reversed(cur):
                if kn + len(t.split()) > CHUNK_OVERLAP:
                    break
                keep.insert(0, t)
                kn += len(t.split())
            cur, cur_n = keep, kn
        cur.append(s)
        cur_n += wn
    if cur_n >= 40:
        chunks.append({"chunk_id": f"{doc_id}#{idx}", "doc_id": doc_id, "text": " ".join(cur)})
    return chunks


def main() -> None:
    start, end = window()
    raw = load_raw()
    funnel = Counter(loaded=len(raw))
    kept: list[dict] = []
    seen_hash: set[str] = set()

    for d in raw:
        text = clean(d.get("text") or "")
        d["text"] = text
        ok, why = quality_ok(text)
        if not ok:
            funnel[f"drop_quality_{why}"] += 1
            continue
        try:
            lang = detect(text[:4000])
        except Exception:  # noqa: BLE001
            lang = "unk"
        if lang != "en":
            funnel["drop_lang"] += 1
            continue
        rel = len(RELEVANCE.findall(text))
        if rel < 3:
            funnel["drop_irrelevant"] += 1
            continue
        if d.get("source_type") == "youtube":
            t = d.get("title") or ""
            is_speech = bool(YT_PERSON.search(t)) or ("nvidia" in t.lower() and YT_EVENT.search(t))
            if not is_speech or (YT_NOISE.search(t) and not YT_PERSON.search(t)):
                funnel["drop_yt_title"] += 1
                continue
        if d.get("date") and d.get("discovered_via") not in ("seed", "manual"):
            dd = date.fromisoformat(d["date"])
            if not (start <= dd <= end):
                funnel["drop_out_of_window"] += 1
                continue
        h = sha1(re.sub(r"\W+", " ", text.lower()))
        if h in seen_hash:
            funnel["drop_exact_dup"] += 1
            continue
        seen_hash.add(h)
        d["lang"] = lang
        d["n_words"] = len(text.split())
        d["relevance_hits"] = rel
        kept.append(d)

    # Fuzzy dedup: prefer the longer document of a near-duplicate pair.
    kept.sort(key=lambda x: -x["n_words"])
    sh = [shingles(d["text"]) for d in kept]
    drop = set()
    for i in range(len(kept)):
        if i in drop:
            continue
        for j in range(i + 1, len(kept)):
            if j in drop:
                continue
            jac = jaccard(sh[i], sh[j])
            # Same-day YouTube re-streams of one keynote have different auto-captions; use a looser bar.
            same_event = (kept[i]["source_type"] == kept[j]["source_type"] == "youtube" and kept[i].get("date") and kept[i].get("date") == kept[j].get("date"))
            if jac >= NEAR_DUP_JACCARD or (same_event and jac >= 0.35):
                drop.add(j)
                funnel["drop_near_dup"] += 1
    kept = [d for i, d in enumerate(kept) if i not in drop]
    funnel["kept_docs"] = len(kept)

    chunks: list[dict] = []
    for d in kept:
        for c in chunk(d["text"], d["id"]):
            c.update({"title": d.get("title"), "url": d.get("url"), "date": d.get("date"), "source_type": d["source_type"]})
            chunks.append(c)
    funnel["chunks"] = len(chunks)
    funnel["total_words"] = sum(d["n_words"] for d in kept)

    write_jsonl(CURATED / "docs.jsonl", kept)
    write_jsonl(CURATED / "chunks.jsonl", chunks)
    by_type = Counter(d["source_type"] for d in kept)
    stats = {"funnel": dict(funnel), "by_source_type": dict(by_type), "window": [start.isoformat(), end.isoformat()]}
    (CURATED / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

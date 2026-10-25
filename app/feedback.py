"""Reinforcement layer: human feedback on answers, used online and offline.

Online (every request): rated past answers whose question is semantically close to the new one are looked up.
  * a "good" answer becomes a style/depth exemplar in the prompt;
  * "bad" answers contribute their reviewer comments as explicit things to avoid, and the passages they cited are
    down-weighted in retrieval for that question.
Offline: ``scripts/10_build_preference_data.py`` turns good/bad pairs into DPO preference data
(prompt / chosen / rejected) for NeMo AutoModel, closing the loop into the fine-tuned model.

Storage: append-only JSONL at data/feedback/feedback.jsonl (one record per rating).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = ROOT / "data" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"


class FeedbackStore:
    def __init__(self, embed_fn):
        """embed_fn(list[str]) -> np.ndarray of normalised query embeddings (the retriever's encode_query)."""
        self._embed = embed_fn
        self._lock = threading.Lock()
        self.records: list[dict] = []
        self._emb: np.ndarray | None = None
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        if FEEDBACK_FILE.exists():
            self.records = [json.loads(l) for l in FEEDBACK_FILE.read_text().splitlines() if l.strip()]
        self._reindex()

    def _reindex(self) -> None:
        qs = [r["question"] for r in self.records]
        self._emb = self._embed(qs) if qs else None

    def add(self, rec: dict) -> dict:
        rec = {"id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
        with self._lock:
            with FEEDBACK_FILE.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.records.append(rec)
            self._reindex()
        return rec

    def similar(self, question: str, threshold: float = 0.80, limit: int = 6) -> list[dict]:
        if self._emb is None or not self.records:
            return []
        q = self._embed([question])[0]
        sims = self._emb @ q
        order = np.argsort(-sims)[:limit]
        return [{**self.records[int(i)], "similarity": float(sims[i])} for i in order if sims[i] >= threshold]

    def guidance(self, question: str) -> tuple[str, dict[str, float]]:
        """Prompt guidance text and per-chunk retrieval weights derived from ratings on similar questions."""
        sim = self.similar(question)
        if not sim:
            return "", {}
        good = [r for r in sim if r.get("rating") == "good"]
        bad = [r for r in sim if r.get("rating") == "bad"]
        parts: list[str] = []
        weights: dict[str, float] = {}
        if good:
            g = max(good, key=lambda r: r["similarity"])
            parts.append("A previous answer to a very similar question was rated GOOD by the user. Match its depth, "
                         "structure and specificity (do not copy it; the sources below may differ):\n" + g["answer"][:1500])
        if bad:
            notes = [f"- {r.get('comment').strip()}" for r in bad if (r.get("comment") or "").strip()]
            if notes:
                parts.append("Previous answers to similar questions were rated BAD; the user's complaints were:\n" + "\n".join(notes[:5])
                             + "\nAvoid these problems.")
            else:
                parts.append("A previous answer to a similar question was rated BAD by the user (no comment). Be more precise, "
                             "more specific to the question, and stricter about citing only what the sources say.")
            for r in bad:
                for cid in r.get("cited_chunk_ids") or []:
                    weights[cid] = min(weights.get(cid, 1.0), 0.7)
        return "\n\n".join(parts), weights

    def stats(self) -> dict:
        n = len(self.records)
        good = sum(r.get("rating") == "good" for r in self.records)
        return {"n": n, "good": good, "bad": n - good, "file": str(FEEDBACK_FILE.relative_to(ROOT))}

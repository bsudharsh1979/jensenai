"""Hybrid retriever: Nemotron-3-Embed dense vectors + BM25, fused with reciprocal rank fusion."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "index"
_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class Retriever:
    def __init__(self, index_dir: Path = INDEX, device: str | None = None):
        meta = json.loads((index_dir / "meta.json").read_text())
        self.embed_model_name = meta["embed_model"]
        self.chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text().splitlines() if l.strip()]
        self.emb = np.load(index_dir / "embeddings.npy").astype(np.float32)

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([_tok(c["text"]) for c in self.chunks])

        import torch
        from sentence_transformers import SentenceTransformer

        device = device or os.environ.get("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            self.embed_model_name, device=device,
            model_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32, "attn_implementation": "sdpa"},
        )
        self.model.max_seq_length = 512

    def search(self, query: str, top_k: int = 6, candidates: int = 40) -> list[dict]:
        q = self.model.encode_query([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        dense = self.emb @ q
        dense_rank = np.argsort(-dense)[:candidates]
        bm = self.bm25.get_scores(_tok(query))
        bm_rank = np.argsort(-bm)[:candidates]

        # Reciprocal rank fusion (k=60), then light de-duplication by document.
        fused: dict[int, float] = {}
        for r, i in enumerate(dense_rank):
            fused[int(i)] = fused.get(int(i), 0) + 1.0 / (60 + r)
        for r, i in enumerate(bm_rank):
            fused[int(i)] = fused.get(int(i), 0) + 1.0 / (60 + r)
        order = sorted(fused, key=lambda i: -fused[i])

        out, per_doc = [], {}
        for i in order:
            c = self.chunks[i]
            if per_doc.get(c["doc_id"], 0) >= 2:
                continue
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            out.append({**c, "score": round(float(fused[i]), 5), "dense": round(float(dense[i]), 4)})
            if len(out) >= top_k:
                break
        return out

    def stats(self) -> dict:
        docs = {}
        for c in self.chunks:
            d = docs.setdefault(c["doc_id"], {"title": c.get("title"), "url": c.get("url"), "date": c.get("date"), "source_type": c.get("source_type"), "chunks": 0})
            d["chunks"] += 1
        dates = sorted(d["date"] for d in docs.values() if d.get("date"))
        return {"n_chunks": len(self.chunks), "n_docs": len(docs), "date_range": [dates[0], dates[-1]] if dates else None,
                "embed_model": self.embed_model_name, "docs": sorted(docs.values(), key=lambda d: d.get("date") or "", reverse=True)}

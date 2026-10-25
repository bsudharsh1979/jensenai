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
_NOISY_TITLE = re.compile(
    r"\b(live|watch live|reaction|react|supercut|speechless|shocks|stunned|destroys|exposed|pregame|post.?game|"
    r"stock|\$nvda|price target|should you buy|analysis|breakdown|explained|recap|summary|highlights|"
    r"풀이|解説|分析)\b", re.I)
_FILLER = re.compile(r"\b(um+|uh+|you know|like,|sort of|kind of)\b", re.I)


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


CITE_RE = re.compile(r"\[(\d{1,2})\]")


class Retriever:
    def __init__(self, index_dir: Path = INDEX, device: str | None = None):
        meta = json.loads((index_dir / "meta.json").read_text())
        self.embed_model_name = meta["embed_model"]
        self.chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text().splitlines() if l.strip()]
        self.emb = np.load(index_dir / "embeddings.npy").astype(np.float32)

        from rank_bm25 import BM25Okapi

        # index title + date + text so event names that only appear in titles (Computex, SIGGRAPH, Q2 FY2027 …) match
        self.bm25 = BM25Okapi([_tok(f"{c.get('title') or ''} {c.get('date') or ''} {c['text']}") for c in self.chunks])

        import torch
        from sentence_transformers import SentenceTransformer

        device = device or os.environ.get("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            self.embed_model_name, device=device,
            model_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32, "attn_implementation": "sdpa"},
        )
        self.model.max_seq_length = 512
        self._quality = np.array([self._quality_prior(c) for c in self.chunks], dtype=np.float32)

    def search(self, query: str, top_k: int = 10, candidates: int = 60, weights: dict[str, float] | None = None) -> list[dict]:
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
        # Source-quality prior: primary material (transcripts, earnings calls, NVIDIA's own publications)
        # outranks live-stream commentary, reaction videos and filler-heavy speech-to-text at equal relevance.
        for i in fused:
            fused[i] *= self._quality[i]
            if weights:
                fused[i] *= weights.get(self.chunks[i]["chunk_id"], 1.0)  # feedback layer: passages cited in bad answers
        order = sorted(fused, key=lambda i: -fused[i])

        out, per_doc = [], {}
        for i in order:
            c = self.chunks[i]
            cap = 3 if self._quality[i] >= 1.0 else 1
            if per_doc.get(c["doc_id"], 0) >= cap:
                continue
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            out.append({**c, "idx": int(i), "score": round(float(fused[i]), 5), "dense": round(float(dense[i]), 4), "quality": round(float(self._quality[i]), 2)})
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _quality_prior(c: dict) -> float:
        title = (c.get("title") or "").lower()
        url = (c.get("url") or "").lower()
        text = c.get("text") or ""
        words = max(len(text.split()), 1)
        q = 1.0
        if c.get("source_type") == "web":
            if any(k in url for k in ("nvidianews", "blogs.nvidia", "nvidia.com")):
                q *= 1.35  # NVIDIA's own words
            elif any(k in title for k in ("transcript", "earnings call", "keynote", "full speech", "remarks", "fireside")):
                q *= 1.30  # verbatim transcripts
            elif any(k in url for k in ("note.com", "medium.com", "substack", "reddit", "36kr")):
                q *= 0.80  # blogs / aggregators
        else:  # youtube captions
            if _NOISY_TITLE.search(title):
                q *= 0.55  # live-stream chatter, reaction and commentary channels
            elif any(k in title for k in ("keynote", "earnings call", "fireside", "interview", "commencement", "transcript", "full")):
                q *= 1.0
            else:
                q *= 0.85
        filler = len(_FILLER.findall(text)) / words
        if filler > 0.04:
            q *= 0.7  # heavy "um / uh / you know": raw speech-to-text, often not Jensen
        return q

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode_query(texts, convert_to_numpy=True, normalize_embeddings=True)

    _SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“(])")
    _CITE = re.compile(r"\[\d{1,2}\]")

    def attribute(self, text: str, hits: list[dict], threshold: float = 0.38) -> tuple[str, list[int]]:
        """Guarantee citations without cluttering the text: paragraphs that carry no [n] get ONE citation (the
        best-matching source by embedding similarity) appended at the end. Headings, bold-only lines and fenced
        code / mermaid blocks are left untouched."""
        if not hits:
            return text, []
        lines = text.split("\n")
        targets: list[int] = []
        in_fence = False
        for li, line in enumerate(lines):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            stripped = line.strip()
            if in_fence or not stripped or stripped.startswith("#") or CITE_RE.search(stripped):
                continue
            if len(re.sub(r"\*\*.+?\*\*", "", stripped).split()) >= 8:
                targets.append(li)
        if targets:
            q = self.model.encode_query([lines[i] for i in targets], convert_to_numpy=True, normalize_embeddings=True)
            d = np.stack([self.emb[h["idx"]] for h in hits])
            sims = q @ d.T
            for row, li in enumerate(targets):
                best = int(np.argmax(sims[row]))
                if sims[row, best] >= threshold:
                    lines[li] = lines[li].rstrip() + f" [{best + 1}]"
        text = "\n".join(lines)
        cited = sorted({int(n) for n in CITE_RE.findall(text) if 1 <= int(n) <= len(hits)})
        return text, cited

    def stats(self) -> dict:
        docs = {}
        for c in self.chunks:
            d = docs.setdefault(c["doc_id"], {"title": c.get("title"), "url": c.get("url"), "date": c.get("date"), "source_type": c.get("source_type"), "chunks": 0})
            d["chunks"] += 1
        dates = sorted(d["date"] for d in docs.values() if d.get("date"))
        return {"n_chunks": len(self.chunks), "n_docs": len(docs), "date_range": [dates[0], dates[-1]] if dates else None,
                "embed_model": self.embed_model_name, "docs": sorted(docs.values(), key=lambda d: d.get("date") or "", reverse=True)}

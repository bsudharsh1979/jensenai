#!/usr/bin/env python
"""Build the hybrid retrieval index (dense Nemotron-3-Embed-1B + BM25) over curated chunks.

Outputs in data/index/:
  chunks.jsonl        chunk metadata/text in index order
  embeddings.npy      float16 [n_chunks, 2048] L2-normalised passage embeddings
  meta.json           embedding model + dims
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import CURATED, INDEX, read_jsonl

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/Nemotron-3-Embed-1B-BF16")
BATCH = int(os.environ.get("EMBED_BATCH", "32"))


def main() -> None:
    chunks = read_jsonl(CURATED / "chunks.jsonl")
    if not chunks:
        raise SystemExit("no chunks; run 03_curate.py first")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device, model_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32, "attn_implementation": "sdpa"})
    model.max_seq_length = 2048

    def passage(c: dict) -> str:
        head = f"{c.get('title') or ''} ({c.get('date') or ''})".strip()
        return f"{head}\n{c['text']}"

    t = time.time()
    emb = model.encode_document([passage(c) for c in chunks], batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    print(f"embedded {len(chunks)} chunks in {time.time()-t:.1f}s -> {emb.shape}")

    np.save(INDEX / "embeddings.npy", emb.astype(np.float16))
    with open(INDEX / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (INDEX / "meta.json").write_text(json.dumps({"embed_model": EMBED_MODEL, "dim": int(emb.shape[1]), "n_chunks": len(chunks), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    print("index written to", INDEX)


if __name__ == "__main__":
    main()

"""Render the technical-architecture infographic (SVG + PNG): fine-tuning method, RAG design, serving stack,
answer pipeline and the reinforcement (feedback) loop. Numbers are read from the pipeline artefacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CURATED, SFT, INDEX  # noqa: E402

OUT_DIR = ROOT / "docs" / "img"
GREEN, DARK, PANEL, TXT, MUTED, LINE = "#76B900", "#0f1115", "#1b1f27", "#f3f4f6", "#9aa3af", "#2a3040"
BLUE, AMBER, PINK = "#4fb3ff", "#ffb347", "#ff6e9c"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def count_lines(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0


def load() -> dict:
    stats = json.loads((CURATED / "stats.json").read_text())
    meta = json.loads((INDEX / "meta.json").read_text()) if (INDEX / "meta.json").exists() else {}
    ev = json.loads((ROOT / "eval" / "summary.json").read_text()) if (ROOT / "eval" / "summary.json").exists() else {}
    fb = ROOT / "data" / "feedback" / "feedback.jsonl"
    return dict(chunks=stats["funnel"]["chunks"], docs=stats["funnel"]["kept_docs"], dim=meta.get("dim", 2048),
                train=count_lines(SFT / "train.jsonl"), val=count_lines(SFT / "val.jsonl"),
                eval=ev.get("metrics", {}), feedback=count_lines(fb))


def panel(x, y, w, h, tag, title, accent=GREEN):
    return [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{PANEL}" stroke="{LINE}"/>',
            f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{accent}"/>',
            f'<text x="{x+22}" y="{y+32}" fill="{accent}" font-size="13" font-weight="700" letter-spacing="2">{esc(tag)}</text>',
            f'<text x="{x+22}" y="{y+60}" fill="{TXT}" font-size="21" font-weight="700">{esc(title)}</text>']


def bullets(x, y, lines, color=MUTED, size=13.5, gap=24, dot=GREEN):
    out = []
    for i, ln in enumerate(lines):
        out.append(f'<circle cx="{x}" cy="{y+i*gap-5}" r="3.5" fill="{dot}"/>')
        out.append(f'<text x="{x+14}" y="{y+i*gap}" fill="{color}" font-size="{size}">{esc(ln)}</text>')
    return out


def kv(x, y, k, v, accent=GREEN):
    return [f'<text x="{x}" y="{y}" fill="{accent}" font-size="26" font-weight="800">{esc(v)}</text>',
            f'<text x="{x}" y="{y+20}" fill="{MUTED}" font-size="12.5" letter-spacing="1">{esc(k.upper())}</text>']


def box(x, y, w, h, label, sub="", fill="#232a36", stroke=GREEN):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
           f'<text x="{x+w/2}" y="{y+h/2+(0 if sub else 5)}" text-anchor="middle" fill="{TXT}" font-size="13.5" font-weight="700">{esc(label)}</text>']
    if sub:
        out.append(f'<text x="{x+w/2}" y="{y+h/2+17}" text-anchor="middle" fill="{MUTED}" font-size="11">{esc(sub)}</text>')
    return out


def arrow(x1, y1, x2, y2, color=GREEN, label=""):
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 10 * math.cos(ang), y2 - 10 * math.sin(ang)
    px, py = 6 * math.sin(ang), -6 * math.cos(ang)
    out = [f'<line x1="{x1}" y1="{y1}" x2="{hx}" y2="{hy}" stroke="{color}" stroke-width="2.2"/>',
           f'<polygon points="{x2},{y2} {hx+px},{hy+py} {hx-px},{hy-py}" fill="{color}"/>']
    if label:
        out.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-8}" text-anchor="middle" fill="{color}" font-size="11.5" font-weight="600">{esc(label)}</text>')
    return out


def build(d: dict) -> str:
    W, H = 1720, 1460
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{DARK}"/>',
         f'<text x="60" y="68" fill="{TXT}" font-size="38" font-weight="800">JensenAI <tspan fill="{GREEN}">·</tspan> technical architecture</text>',
         f'<text x="60" y="100" fill="{MUTED}" font-size="17">Fine-tuning method, retrieval design, serving stack, answer pipeline and the human-feedback loop — all running on one NVIDIA DGX Spark (GB10, 128 GB unified memory).</text>']

    # ---------------------------------------------------------------- row 1: fine-tuning | RAG
    y1, h1 = 130, 430
    s += panel(60, y1, 800, h1, "FINE-TUNING METHOD", "Parameter-efficient LoRA SFT with NeMo AutoModel")
    s += kv(84, y1 + 112, "trainable params", "19.2M / 0.48%")
    s += kv(330, y1 + 112, "rank · alpha · dropout", "r16 · α32 · 0.05")
    s += kv(600, y1 + 112, "train time (GB10)", "6.5 min")
    s += bullets(92, y1 + 172, [
        "Student: Nemotron-3-Nano-4B-BF16 — dense hybrid Mamba-2 + Transformer (nemotron_h)",
        f"Data: {d['train'] + d['val']:,} grounded Q&A pairs synthesised by Nemotron-3-Super (3 per passage)",
        "Recipe: NeMo AutoModel 26.08 · LoRA on attention + MLP, *.out_proj excluded (Mamba frozen)",
        "Optimiser: AdamW 1e-4 cosine · bf16 · FSDP2 · 3 epochs / 249 steps · lowest-val-loss checkpoint",
        "Loss: train 2.50 → 1.06 · val 2.01 → 1.72 (held-out set of 57 questions)",
        "Merge: pure-tensor W' = W + (α/r)·B·A — Mamba dt_bias / out_proj never re-initialised",
        "Export: llama.cpp GGUF bf16 (raw-config hparams) → Ollama Modelfile, explicit ChatML template",
    ], gap=27)
    ev = d["eval"]
    if ev:
        t, b = ev.get("tuned", {}), ev.get("base", {})
        s.append(f'<text x="84" y="{y1+h1-60}" fill="{GREEN}" font-size="13" font-weight="700" letter-spacing="2">HELD-OUT RESULT · BASE → FINE-TUNED</text>')
        s.append(f'<text x="84" y="{y1+h1-30}" fill="{TXT}" font-size="16">Token F1 {b.get("f1",0):.3f} → <tspan fill="{GREEN}" font-weight="700">{t.get("f1",0):.3f}</tspan>'
                 f'   ·   ROUGE-L {b.get("rougeL",0):.3f} → <tspan fill="{GREEN}" font-weight="700">{t.get("rougeL",0):.3f}</tspan>'
                 f'   ·   Super-as-judge {b.get("judge_mean",0):.2f} → <tspan fill="{GREEN}" font-weight="700">{t.get("judge_mean",0):.2f}</tspan> / 5   ·   wins 47 / 57</text>')

    s += panel(890, y1, 770, h1, "RAG TYPE", "Hybrid dense + lexical retrieval with quality priors", BLUE)
    s += kv(914, y1 + 112, "indexed chunks", f"{d['chunks']:,}", BLUE)
    s += kv(1130, y1 + 112, "documents", f"{d['docs']}", BLUE)
    s += kv(1320, y1 + 112, "embedding dims", f"{d['dim']}", BLUE)
    s += bullets(922, y1 + 172, [
        f"Dense: Nemotron-3-Embed-1B ({d['dim']}-d cosine) over 350-word passages with title + date",
        "Lexical: BM25 over title + date + text, so event names in video titles are matchable",
        "Fusion: Reciprocal Rank Fusion (k=60) × source-quality prior (transcripts ↑, reaction streams ↓)",
        "Diversity: per-document cap (3 for high-quality docs, 1 otherwise), 60 candidates → top 8",
        "Conversation-aware: follow-ups retrieved with the previous question merged in",
        "Citation guarantee: uncited paragraphs get one embedding-attributed [n] (cos ≥ 0.38)",
        "Feedback-weighted: passages cited in 'bad' answers down-weighted ×0.7 for similar questions",
    ], dot=BLUE, gap=27)

    # ---------------------------------------------------------------- row 2: answer pipeline flow
    y2, h2 = y1 + h1 + 30, 300
    s += panel(60, y2, 1600, h2, "ANSWER PIPELINE", "From question to cited, structured, exportable briefing", AMBER)
    by = y2 + 110
    flow = [("Question", "+ notes of prior turns", 170), ("Retrieve", "hybrid RRF · quality · feedback", 215),
            ("Prompt", "rules · 8 sources · format", 190), ("Generate", "Super 120B (expert) · Nano-4B LoRA (fast)", 290),
            ("Tidy & attribute", "≤2 cites / paragraph · lead-ins", 230), ("Render", "sections · Mermaid · code · PDF", 215)]
    x = 90
    for i, (lab, sub, w) in enumerate(flow):
        s += box(x, by, w, 62, lab, sub, stroke=AMBER)
        if i < len(flow) - 1:
            nx = x + w + 34
            s += arrow(x + w + 2, by + 31, nx - 2, by + 31, AMBER)
            x = nx
    s += bullets(92, by + 110, [
        "Streaming SSE: status → sources → tokens → final (tidied text) → cited ids → done",
        "Architecture & code only for system / stack questions: Mermaid diagram + verified NVIDIA snippet (TensorRT-LLM, Dynamo, NIM, NeMo, cuDF, Isaac, CUDA Python, OpenUSD)",
        "Per-request nonce vs Ollama prefix cache · conversation notes, not verbatim history · repeat_penalty 1.05",
    ], dot=AMBER, gap=26)

    # ---------------------------------------------------------------- row 3: serving stack | reinforcement loop
    y3, h3 = y2 + h2 + 30, 420
    s += panel(60, y3, 800, h3, "SERVING STACK · DGX SPARK", "Two Nemotrons on one GPU, swapped on demand")
    s += box(90, y3 + 95, 340, 70, "Expert · nemotron-3-super", "120B MoE · Q4_K_M · ~91 GB · 32k ctx · ~20 tok/s")
    s += box(460, y3 + 95, 370, 70, "Fast · nemotron-nano-4b-jensen", "LoRA merge · GGUF bf16 · 16k ctx · ~32 tok/s")
    s += box(90, y3 + 190, 340, 56, "Ollama runtime (:11434)", "keep_alive 24h · app-level exclusive eviction")
    s += box(460, y3 + 190, 370, 56, "Nemotron-3-Embed-1B + BM25", "sentence-transformers · rank_bm25 · in-process")
    s += box(90, y3 + 270, 740, 56, "FastAPI :8600 · systemd · SSE · /api/chat /api/answer /api/feedback /api/pdf")
    s += bullets(92, y3 + 362, ["GB10 Grace Blackwell · 128 GB unified · CUDA 13 · aarch64 · no cloud · WeasyPrint PDFs server-side"], gap=24)

    s += panel(890, y3, 770, h3, "REINFORCEMENT LAYER", "Human good / bad ratings close the loop online and offline", PINK)
    # loop diagram
    cx, cy = 1275, y3 + 150
    s += box(cx - 320, cy - 28, 170, 56, "Answer", "+ sources, mode, model", stroke=PINK)
    s += box(cx - 85, cy - 28, 170, 56, "Good / Bad + comment", "POST /api/feedback", stroke=PINK)
    s += box(cx + 150, cy - 28, 200, 56, "feedback.jsonl", "question · answer · rating · cited ids", stroke=PINK)
    s += arrow(cx - 148, cy, cx - 87, cy, PINK)
    s += arrow(cx + 87, cy, cx + 148, cy, PINK)
    s += box(cx - 320, cy + 70, 300, 56, "Online: exemplars & avoid-notes", "similar question (cos ≥ 0.80) → prompt", stroke=PINK)
    s += box(cx + 20, cy + 70, 330, 56, "Offline: DPO preference pairs", "scripts/10 → dpo.jsonl → NeMo AutoModel", stroke=PINK)
    s += arrow(cx + 250, cy + 30, cx - 100, cy + 68, PINK)
    s += arrow(cx + 250, cy + 30, cx + 185, cy + 68, PINK)
    s += bullets(922, cy + 165, [
        "Good answers to similar questions → exemplars; bad-answer comments → explicit 'avoid' rules",
        "Sources cited in bad answers down-weighted in retrieval for that question family",
        "Good/bad pairs per question → chosen/rejected DPO data to re-tune the fast model",
        f"Ratings recorded so far: {d['feedback']}",
    ], dot=PINK, gap=25)

    s.append(f'<text x="60" y="{H-24}" fill="#5b6270" font-size="13">github.com/bsudharsh1979/jensenai · generated by scripts/11_render_tech_infographic.py from the pipeline artefacts</text>')
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build(load())
    (OUT_DIR / "tech_architecture_infographic.svg").write_text(svg)
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT_DIR / "tech_architecture_infographic.png"), output_width=1720)
    print("wrote", OUT_DIR / "tech_architecture_infographic.png")


if __name__ == "__main__":
    main()

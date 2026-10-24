"""Render the "how it was formulated" pipeline infographic (SVG + PNG) from the real pipeline artefacts.

Reads data/curated/stats.json, data/sft/{train,val}.jsonl, data/index/meta.json and eval/summary.json
(if present) so every number on the graphic is the number that was actually measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CURATED, SFT, INDEX  # noqa: E402

OUT_DIR = ROOT / "docs" / "img"
GREEN, DARK, PANEL, TXT, MUTED = "#76B900", "#0f1115", "#1b1f27", "#f3f4f6", "#9aa3af"


def count_lines(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0


def load() -> dict:
    stats = json.loads((CURATED / "stats.json").read_text())
    f = stats["funnel"]
    raw_dir = ROOT / "data" / "raw"
    n_yt = len(list((raw_dir / "youtube").glob("*.json")))
    n_web = len(list((raw_dir / "web").glob("*.json"))) + len(list((raw_dir / "manual").glob("*.json")))
    meta = json.loads((INDEX / "meta.json").read_text()) if (INDEX / "meta.json").exists() else {}
    ev = json.loads((ROOT / "eval" / "summary.json").read_text()) if (ROOT / "eval" / "summary.json").exists() else {}
    return dict(
        n_yt=n_yt, n_web=n_web, loaded=f["loaded"], kept=f["kept_docs"], chunks=f["chunks"],
        words=f["total_words"], dropped=f["loaded"] - f["kept_docs"],
        kept_yt=stats["by_source_type"].get("youtube", 0), kept_web=stats["by_source_type"].get("web", 0),
        window=stats["window"], train=count_lines(SFT / "train.jsonl"), val=count_lines(SFT / "val.jsonl"),
        dim=meta.get("dim", 2048), eval=ev,
    )


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def card(x, y, w, h, step, title, big, lines, accent=GREEN):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{PANEL}" stroke="#2a3040"/>',
           f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{accent}"/>',
           f'<text x="{x+20}" y="{y+30}" fill="{accent}" font-size="13" font-weight="700" letter-spacing="2">STEP {step}</text>',
           f'<text x="{x+20}" y="{y+56}" fill="{TXT}" font-size="20" font-weight="700">{esc(title)}</text>',
           f'<text x="{x+20}" y="{y+100}" fill="{accent}" font-size="30" font-weight="800">{esc(big)}</text>']
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x+20}" y="{y+130+i*21}" fill="{MUTED}" font-size="14">{esc(ln)}</text>')
    return "\n".join(out)


def arrow(x1, y, x2):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2-10}" y2="{y}" stroke="{GREEN}" stroke-width="3"/>'
            f'<polygon points="{x2-12},{y-8} {x2},{y} {x2-12},{y+8}" fill="{GREEN}"/>')


def build(d: dict) -> str:
    W, H = 1720, 900
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{DARK}"/>',
         f'<text x="60" y="70" fill="{TXT}" font-size="40" font-weight="800">JensenAI <tspan fill="{GREEN}">·</tspan> how NVIDIA Insight was formulated</text>',
         f'<text x="60" y="104" fill="{MUTED}" font-size="18">Everything Jensen Huang said in public, {d["window"][0]} → {d["window"][1]}, distilled into a 4B Nemotron model and a cited Q&amp;A app — all on one DGX Spark.</text>']

    cw, ch, gap, y0 = 300, 250, 30, 150
    cards = [
        ("1", "Collect", f"{d['n_yt'] + d['n_web']} docs",
         [f"{d['n_yt']} YouTube caption tracks (yt-dlp)", f"{d['n_web']} web pages / transcripts / PDFs",
          "NVIDIA Newsroom + Blog + news RSS", "16 date-sorted YouTube searches"]),
        ("2", "Curate", f"{d['chunks']:,} chunks",
         [f"{d['loaded']} docs loaded → {d['kept']} kept", "lang-ID · quality · exact + fuzzy dedup",
          "title & relevance & date filters", f"{d['words']:,} words, 350-word chunks"]),
        ("3", "Synthesize", f"{d['train'] + d['val']:,} Q&A pairs",
         ["teacher: Nemotron-3-Super (Ollama)", "3 grounded, attributed pairs / chunk",
          f"{d['train']:,} train · {d['val']} held-out", "OpenAI messages JSONL"]),
        ("4", "Fine-tune", "LoRA r16 · α32",
         ["NeMo AutoModel 26.08 container", "Nemotron-3-Nano-4B (hybrid Mamba-2)",
          "19.2M trainable (0.48%) · bf16 · FSDP2", "AdamW 1e-4 cosine · 3 epochs"]),
        ("5", "Serve", "vLLM + RAG",
         ["merged checkpoint → vLLM 26.02", f"Nemotron-3-Embed-1B ({d['dim']}-d) + BM25",
          "reciprocal-rank fusion, [n] citations", "FastAPI · SSE · systemd on :8600"]),
    ]
    for i, (step, title, big, lines) in enumerate(cards):
        x = 60 + i * (cw + gap)
        s.append(card(x, y0, cw, ch, step, title, big, lines))
        if i < len(cards) - 1:
            s.append(arrow(x + cw + 4, y0 + ch / 2, x + cw + gap - 4))

    # bottom band: platform + results
    by = y0 + ch + 50
    s.append(f'<rect x="60" y="{by}" width="800" height="330" rx="14" fill="{PANEL}" stroke="#2a3040"/>')
    s.append(f'<text x="84" y="{by+40}" fill="{GREEN}" font-size="13" font-weight="700" letter-spacing="2">PLATFORM</text>')
    s.append(f'<text x="84" y="{by+72}" fill="{TXT}" font-size="22" font-weight="700">NVIDIA DGX Spark — single box, no cloud</text>')
    plat = ["GB10 Grace Blackwell · 128 GB unified memory · CUDA 13", "Teacher (120B MoE), student training, embeddings and serving share one GPU",
            "Training: nvcr.io/nvidia/nemo-automodel:26.08 (official Spark playbook)", "Serving: nvcr.io/nvidia/vllm:26.02-py3, mamba cache fp32, reasoning toggle",
            "Open models only: Nemotron-3 Nano 4B · Super 120B-A12B · Embed 1B", "Nemotron Open Model License · reproducible with 8 scripts"]
    for i, ln in enumerate(plat):
        s.append(f'<circle cx="92" cy="{by+104+i*34-5}" r="4" fill="{GREEN}"/>')
        s.append(f'<text x="108" y="{by+104+i*34}" fill="{MUTED}" font-size="16">{esc(ln)}</text>')

    s.append(f'<rect x="890" y="{by}" width="790" height="330" rx="14" fill="{PANEL}" stroke="#2a3040"/>')
    s.append(f'<text x="914" y="{by+40}" fill="{GREEN}" font-size="13" font-weight="700" letter-spacing="2">RESULTS · HELD-OUT Q&amp;A</text>')
    ev = d["eval"]
    if ev:
        m = ev.get("metrics", ev)
        t, b = m.get("tuned", {}), m.get("base", {})
        n = t.get("n", d["val"])
        s.append(f'<text x="914" y="{by+72}" fill="{TXT}" font-size="22" font-weight="700">Base vs fine-tuned Nemotron-3-Nano-4B ({n} questions)</text>')
        rows = [("Token F1", b.get("f1"), t.get("f1"), 1.0), ("ROUGE-L", b.get("rougeL"), t.get("rougeL"), 1.0),
                ("Judge score (1–5, Nemotron-3-Super)", b.get("judge_mean"), t.get("judge_mean"), 5.0)]
        for i, (name, bv, tv, mx) in enumerate(rows):
            yy = by + 110 + i * 70
            s.append(f'<text x="914" y="{yy}" fill="{MUTED}" font-size="15">{esc(name)}</text>')
            for j, (lab, v, col) in enumerate([("base", bv, "#5b6270"), ("fine-tuned", tv, GREEN)]):
                if v is None:
                    continue
                w = max(4, int(560 * float(v) / mx))
                s.append(f'<rect x="914" y="{yy+8+j*20}" width="{w}" height="14" rx="4" fill="{col}"/>')
                s.append(f'<text x="{914+w+8}" y="{yy+20+j*20}" fill="{TXT}" font-size="13">{lab} {float(v):.3f}</text>')
    else:
        s.append(f'<text x="914" y="{by+72}" fill="{TXT}" font-size="22" font-weight="700">Evaluation pending</text>')
        s.append(f'<text x="914" y="{by+104}" fill="{MUTED}" font-size="16">run scripts/07_eval.py and re-render</text>')

    s.append(f'<text x="60" y="{H-28}" fill="#5b6270" font-size="13">github.com/bsudharsh1979/jensenai · generated by scripts/09_render_infographic.py from the pipeline artefacts</text>')
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build(load())
    (OUT_DIR / "pipeline_infographic.svg").write_text(svg)
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT_DIR / "pipeline_infographic.png"), output_width=1720)
    print("wrote", OUT_DIR / "pipeline_infographic.svg", OUT_DIR / "pipeline_infographic.png")


if __name__ == "__main__":
    main()

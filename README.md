# JensenAI — NVIDIA Insight: Jensen Huang Q&A on DGX Spark

![NVIDIA Insight on DGX Spark](docs/img/app_infographic.jpg)

A fully local, end-to-end pipeline that turns everything Jensen Huang said in public during the last six months
(Apr 24 – Oct 24 2026: GTC Taipei/Computex, SIGGRAPH, GTC Berlin, Q1/Q2 FY27 earnings calls, analyst conferences,
commencement speeches, interviews and podcasts) into an analyst-grade, fully cited Q&A web app on a single
**DGX Spark (GB10, 128 GB unified memory)**. Two Nemotron generators sit behind one curated retrieval index:
**Expert** mode uses **Nemotron-3-Super 120B-A12B** for six-section expert briefings; **Fast** mode uses
**Nemotron-3-Nano-4B LoRA-fine-tuned** on the corpus (NeMo AutoModel) for quick answers.

```
YouTube captions + transcripts + NVIDIA newsroom/blog + press  ──►  NeMo-Curator-style curation
        │                                                                  │
        │                      Nemotron-3-Super (teacher, Ollama) ◄────────┤  context-grounded Q&A synthesis
        │                                                                  ▼
        │                      NeMo AutoModel LoRA SFT (nvcr.io/nvidia/nemo-automodel:26.08) on Nemotron-3-Nano-4B
        │                                                                  ▼
        └── Nemotron-3-Embed-1B + BM25 + quality prior ──► FastAPI app ◄── Ollama: Nemotron-3-Super (expert) / fine-tuned Nano-4B GGUF (fast)
```

> **Executive summary with infographics:** [docs/EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)

## Key features, main benefits and added features

![Key features, main benefits, added features](docs/img/features_benefits_infographic.jpg)

**Key features**

* **Expert mode: Nemotron-3-Super 120B-A12B** — the strongest Nemotron that fits on a Spark, running locally in Ollama; reads 8–10 retrieved transcript passages and writes an analyst briefing in which every factual sentence is cited and anything the sources don't cover is said to be not covered.
* **Fast mode: fine-tuned Nemotron-3-Nano-4B** — LoRA SFT on 6 months of Jensen Huang keynotes, earnings calls, interviews and NVIDIA press, trained with NeMo AutoModel on the DGX Spark itself; exported to GGUF for Ollama (~30 s per briefing).
* **Hybrid RAG with guaranteed citations** — Nemotron-3-Embed-1B dense retrieval fused with BM25 and a *source-quality prior* (verbatim transcripts and NVIDIA publications outrank live-stream chatter and reaction videos). After generation, any sentence the model left uncited is attributed to the best-matching passage with the same embedding model, and the sources panel shows which passages were actually cited.
* **Runs 100% on DGX Spark** — teacher, training, embeddings, inference and web app on one GB10 box; nothing leaves the machine.
* **Structured analyst briefings** — every answer is written in paragraphs under six fixed sections: *Summary · Technical view · Business view · Real-world use cases · Who is already using it · Insights*, so one question yields the engineering facts, the commercial meaning, concrete deployment scenarios, named adopters (only those the sources name, each cited) and non-obvious takeaways. `structured: false` in the API request returns a short direct answer instead.
* **Streaming web chat UI** — token-by-token SSE answers rendered with section headings, bold sub-topic lead-ins, sources panel, RAG on/off toggle, example questions.
* **Architecture diagrams and NVIDIA code — only when relevant** — for questions about a system, chip, stack or developer workflow (Vera Rubin NVL72, Dynamo, DGX Spark, Cosmos…) the briefing adds an *Architecture & code* section: a Mermaid architecture/data-flow diagram built from what the sources describe, plus a short example using the fitting NVIDIA library (TensorRT-LLM, Dynamo, NIM, NeMo AutoModel, cuDF/RAPIDS, Isaac Sim, Cosmos, CUDA Python, OpenUSD) taken from a catalog of verified snippets (`app/nvidia_snippets.py`). Financial, strategic and event-summary questions never get it.
* **Export to PDF** — one click under any answer produces an A4 briefing (question, sections, rendered diagrams, code, numbered sources with URLs, model/time footer) via WeasyPrint (`POST /api/pdf`).
* **Reinforcement layer (good / bad feedback)** — 👍/👎 plus an optional comment per answer (`POST /api/feedback` → `data/feedback/feedback.jsonl`). Online, ratings on semantically similar questions (cos ≥ 0.80) shape the next answer: a good answer becomes a depth/structure exemplar, bad-answer comments become explicit "avoid" rules, and passages cited in bad answers are down-weighted in retrieval. Offline, `scripts/10_build_preference_data.py` turns good/bad pairs into DPO data for NeMo AutoModel to re-tune the fast model.
* **Agentic claim verification (precision layer)** — after drafting, the expert model fact-checks every cited sentence against the full text of the passage it cites and returns a verdict per sentence: *supported*, *corrected* (rewritten to say only what the source says) or *remove*. Unsupported claims never reach the reader; the answer shows "✔ Verified 86% of 21 claims · 0 corrected · 3 removed" with a *show changes* link. On by default in Expert mode (`verify: false` to skip). A planted fabrication ("$250 billion revenue, Jensen retiring") is caught and removed in testing.
* **Precise mode = NeMo Agent Toolkit workflow** — `nat/` is a NeMo Agent Toolkit plugin (`nvidia-nat` 1.9) registering the app's capabilities as toolkit functions (`jensen_retrieve`, `jensen_draft`, `jensen_verify`) and a workflow `jensen_precise_briefing` that runs *draft → verify → refine*: if fewer than 90 % of sentences verify, it re-drafts with the fact-checker's notes (up to 2 rounds) and keeps the best-scoring version. Served by `nat serve` (systemd unit `jensen-nat`, port 8601) and exposed in the UI as the third mode; also runnable as `nat run … --input "…"` and evaluable with `nat eval` (toolkit `tunable_rag_evaluator`, LLM-as-judge for coverage/correctness/relevance against reference answers, local Nemotron-3-Super as judge). *Note: NVIDIA deprecated the NeMo Agent Toolkit in October 2026 (EOL end-2026) in favour of NeMo Relay / Helix / Fabric; the integration is isolated in `nat/` and `app/verify.py` so it can be re-hosted on the successor without touching the app.*
* **Expert-quality diagrams, on screen and in print** — architecture diagrams are layered `flowchart TB` graphs (compute / networking / software / applications as sub-graphs, two-line quoted labels, labelled edges, a fixed colour code with legend), plus a `sequenceDiagram` for data flows or a `timeline` for dated roadmaps when the sources give them; every briefing ends with a **Use-case inspiration** section (3-5 use-case *types* this topic inspires, each labelled "(inspiration)" where extrapolated) closed by a Mermaid **use-case map** (mindmap). For PDF/print the browser re-renders each diagram in a light print theme with pure-SVG text and rasterises it at 3× (WeasyPrint cannot draw Mermaid's HTML labels), so printouts are crisp with captions and legends.
* **Readable by design** — short paragraphs under each heading, each sub-topic with a bold lead-in, at most two citations per paragraph; a post-processing pass (`_tidy`) collapses citation stacks and normalises spacing, and uncited paragraphs receive one embedding-attributed citation rather than one per sentence.

**Main benefits**

* **Private** — no cloud API, no data egress, no per-token cost.
* **Accurate** — spot-checked expert answers reproduce every figure from the transcripts (Q2 FY27: $96.2 B revenue, $89 B data center, $108 B ±2 % Q3 guide, 75 % GM, $279 B supply commitments, $26 B returned to shareholders, $32 B inventory, ~70 % FY28 growth) and refuse to name adopters the sources don't name.
* **Two speeds** — Expert briefings in ~60 s (Nemotron-3-Super, ~20 tok/s on GB10); Fast answers in ~30 s from the fine-tuned 4B.
* **Stays current** — re-run the nine numbered scripts to extend the corpus and re-train in well under an hour (collection aside, training takes ~7 minutes).

**Added features**

* **Mode switch in the UI and API** (`"mode": "expert" | "fast"`); the app evicts the other mode's model from Ollama first because Super (~91 GB) and anything else cannot share the Spark's memory with other workloads.
* **Thinking mode on/off** per mode (`EXPERT_THINKING=true` to let Super reason before answering).
* **JSON endpoint** `/api/answer` and SSE endpoint `/api/chat` on `:8600`; optional OpenAI-compatible vLLM serving of the fine-tuned model (`deploy/vllm_serve.sh`, `jensen-vllm.service`) when Super is not in use.
* **Systemd service** — `jensen-insight` starts with the Spark (Ollama must be running).
* **Built-in evaluation** — base vs fine-tuned token-F1, ROUGE-L and Nemotron-3-Super LLM-as-judge on a held-out split.
* **Alternative Ollama/GGUF path** and **auto-generated infographics** (pipeline and technical architecture) rendered from the measured artefacts.

## Technical architecture (infographic)

![Technical architecture: fine-tuning method, RAG type, answer pipeline, serving stack, reinforcement layer](docs/img/tech_architecture_infographic.png)

**Fine-tuning method.** Parameter-efficient LoRA SFT (r=16, α=32, dropout 0.05; 19.2M trainable parameters = 0.48 %) on `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, a dense hybrid Mamba-2/Transformer, using the NeMo AutoModel 26.08 container and NVIDIA's DGX Spark recipe (AdamW 1e-4 cosine, bf16, FSDP2, 3 epochs / 249 steps, 6.5 min). Training data is 719 grounded Q&A pairs synthesised by Nemotron-3-Super from the curated corpus (3 per passage). The adapter is merged with a pure-tensor update (W' = W + (α/r)·B·A) so the Mamba `dt_bias`/`out_proj` tensors are never re-initialised, then exported to GGUF with raw-config hparams and served by Ollama with an explicit ChatML template. Held-out result: token-F1 0.313 → 0.380, ROUGE-L 0.207 → 0.273, Super-as-judge 1.42 → 1.91 / 5, wins 47 / 57.

**RAG type.** Hybrid retrieval: dense `Nemotron-3-Embed-1B` (2048-d cosine) over 2,389 passages plus BM25 over title + date + text, fused with reciprocal-rank fusion (k=60), multiplied by a source-quality prior (transcripts, earnings calls and NVIDIA publications up; live-stream reactions and commentary down), with a per-document cap for diversity. Follow-up questions are detected and retrieved with the previous question merged in; a citation guarantee attributes any uncited paragraph to its best-matching passage; and passages cited in answers the user rated bad are down-weighted (×0.7) for similar questions.

**Answer pipeline.** Question + compact notes of prior turns → retrieval → prompt (analyst rules, 8 numbered sources, six-section format + use-case inspiration, optional architecture/code section with the snippet catalog) → generation by Nemotron-3-Super (expert) or the fine-tuned Nano-4B (fast) → tidy & attribute (≤ 2 citations per paragraph, lead-ins, fences preserved, Mermaid labels sanitised, guaranteed code example) → **verify** (sentence-level fact-check against cited passages; correct/remove) → render (headings, Mermaid, code, PDF). Precise mode wraps draft+verify in the NeMo Agent Toolkit refine loop. A per-request nonce defeats Ollama's prefix cache and conversation *notes* instead of verbatim history stop answer recycling.

**Serving stack.** One GB10 GPU hosts both Nemotrons through Ollama (Super Q4_K_M ~91 GB resident, Nano GGUF swapped in on demand with app-level eviction), the embedding model and BM25 in-process, and a FastAPI/SSE app on `:8600` run as a systemd user service; WeasyPrint renders PDFs server-side.

## Architecture and methods

| Area | What it does | Why it matters |
|---|---|---|
| **Model choice** | Fine-tune target: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` — 4B hybrid Mamba-2/Transformer SLM, 262K context, *explicitly targeted at DGX Spark*. Expert generator: `nemotron-3-super` (120B-A12B MoE, Q4_K_M, Ollama) — the largest Nemotron that fits in 128 GB | The 4B trains in minutes and answers in seconds; testing showed a 4B cannot sustain expert-level, hallucination-free synthesis over 8–10 long passages, while Super reproduces transcript figures exactly, so the app offers both |
| **Data collection** | `yt-dlp` pulls captions for every keynote/interview found by 16 date-sorted YouTube searches; `trafilatura` extracts transcripts (StockAnalysis, Investing.com, ScribeHawk, FactSet PDF), NVIDIA Newsroom/Blog RSS and Bing News RSS for press coverage | 300+ raw documents, ~900K words, all from the 6-month window |
| **Curation (NeMo Curator stages)** | Unicode/boilerplate cleaning → language ID → heuristic quality filters (length, symbol ratio, repeated n-grams) → exact dedup → near-dup dedup with hashed 5-gram shingles (catches re-streams of the same keynote) → relevance & time-window filters → sentence-aware chunking | 173 docs / 2,389 passages / 677K words of clean, de-duplicated, on-topic text |
| **Synthetic SFT data** | `nvidia/nemotron-3-super` (Ollama, thinking off) writes 3 self-contained, source-attributed Q&A pairs per passage; teacher is told to attribute statements only when the speaker is clear | 719 Q&A pairs (662 train / 57 held-out) from 260 prioritised passages; Nemotron post-training recipe for domain data: a larger Nemotron teacher distils knowledge *and* an "attributed answer" style into the small student |
| **Fine-tuning (NeMo best practices)** | NeMo AutoModel recipe `configs/nemotron_nano_4b_jensen_lora_rtx_spark.yaml` derived from NVIDIA's official `nemotron_nano_4b_squad_peft.yaml` and the `*_rtx_spark.yaml` memory settings: LoRA r=16 / α=32 / dropout 0.05 on all linear layers except Mamba `out_proj`, AdamW 1e-4 cosine with warm-up, bf16, FSDP2 + activation checkpointing, answer-only loss on the model's own chat template (`ChatDataset`) | Runs in the DGX Spark-validated `nemo-automodel:26.08` container with the pinned Spark fix commit; 19.2M trainable params (0.48%), ~8.7 GB peak memory, 249 steps in 6.5 min |
| **Merge & export** | LoRA merged into a standalone HF checkpoint (`scripts/06_merge_lora.py`), optional GGUF export for Ollama | One artefact deployable by vLLM, TensorRT-LLM, Ollama |
| **Serving** | Ollama hosts both generators (Super resident, fine-tuned Nano GGUF on demand); vLLM (`nvcr.io/nvidia/vllm:26.02-py3`, `--mamba_ssm_cache_dtype float32`) remains available for the fine-tuned model | One runtime, native thinking toggle and context control; app-level eviction keeps memory safe |
| **RAG** | `nvidia/Nemotron-3-Embed-1B-BF16` dense retrieval + BM25, reciprocal-rank fusion, source-quality prior, per-document caps, embedding-based citation attribution for uncited sentences | Numbers, dates and attributions come from verbatim transcripts and are verifiable in one click |
| **App** | FastAPI + streaming SSE chat UI, structured six-section analyst briefings (technical / business / use cases / adopters / insights), `/api/answer` JSON endpoint, corpus browser, systemd user units for boot-time start | Runs as a service on the DGX Spark at `http://<spark>:8600` |
| **Evaluation** | Base vs fine-tuned on a held-out split: token-F1, ROUGE-L and Nemotron-3-Super LLM-as-judge | +21% F1, +32% ROUGE-L, +35% judge score; see [Results](#results) |

## Hardware / software

* NVIDIA DGX Spark — GB10 (Blackwell, SM 12.1), 20-core Arm, 128 GB LPDDR5X unified memory, CUDA 13.0, driver 580
* Containers: `nvcr.io/nvidia/nemo-automodel:26.08` (training/merge), `nvcr.io/nvidia/vllm:26.02-py3` (serving)
* Host venv: Python 3.12, PyTorch 2.14 cu130 (aarch64), transformers 5.x, sentence-transformers 6.x
* Teacher / judge: Ollama `nemotron-3-super:latest` (Nemotron-3-Super-120B-A12B)

## Repository layout

```
configs/
  sources.yaml                                   # seed URLs, YouTube queries, RSS feeds, 6-month window
  nemotron_nano_4b_jensen_lora_rtx_spark.yaml    # NeMo AutoModel LoRA recipe (DGX Spark)
scripts/
  01_collect_youtube.py   # yt-dlp search + captions -> data/raw/youtube
  02_collect_web.py       # seeds + index crawl + RSS -> trafilatura/pypdf -> data/raw/web
  03_curate.py            # clean / langid / quality / dedup / chunk -> data/curated
  04_generate_sft.py      # Nemotron-3-Super teacher -> data/sft/{train,val}.jsonl (OpenAI messages)
  05_train_nemo_automodel.sh  # docker run nemo-automodel:26.08 + recipe
  06_merge_lora.sh / .py  # LoRA -> merged HF checkpoint (models/nemotron-nano-4b-jensen-merged)
  07_eval.py              # base vs tuned: F1 / ROUGE-L / LLM judge
  08_build_index.py       # Nemotron-3-Embed-1B + BM25 index -> data/index
  09_render_infographic.py# pipeline infographic (SVG/PNG) from the measured artefacts
  10_build_preference_data.py # good/bad feedback -> DPO pairs (data/sft/dpo.jsonl) + NeMo AutoModel DPO config
  11_render_tech_infographic.py # technical-architecture infographic
app/
  server.py  retriever.py  llm.py  feedback.py  verify.py  nvidia_snippets.py  static/index.html  static/vendor/mermaid.min.js
nat/                      # NeMo Agent Toolkit plugin (pip install -e nat/ in .venv-nat): tools + jensen_precise_briefing workflow
configs/nat/              # jensen_precise.yml (nat run / nat serve), jensen_eval.yml (nat eval)
eval/nat_dataset.json     # held-out questions for nat eval
deploy/
  vllm_serve.sh  run_app.sh  Modelfile  systemd/jensen-vllm.service  systemd/jensen-insight.service  systemd/jensen-nat.service
data/   curated/stats.json, sft/train.jsonl, sft/val.jsonl, index/meta.json (large artefacts are git-ignored)
eval/   summary.json
data/feedback/feedback.jsonl  # human ratings (reinforcement layer)
docs/   EXECUTIVE_SUMMARY.md, img/app_infographic.jpg, img/pipeline_infographic.{svg,png}, img/tech_architecture_infographic.{svg,png}
```

## Quick start (DGX Spark)

```bash
git clone https://github.com/bsudharsh1979/jensenai.git ~/jensenai && cd ~/jensenai   # systemd units assume ~/jensenai
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r requirements.txt
export HF_TOKEN=hf_...            # Hugging Face token (model downloads)

# 1-3  corpus
.venv/bin/python scripts/01_collect_youtube.py
.venv/bin/python scripts/02_collect_web.py
.venv/bin/python scripts/03_curate.py

# 4    synthetic SFT data (needs `ollama pull nemotron-3-super`)
MAX_CHUNKS=260 .venv/bin/python scripts/04_generate_sft.py

# 5-6  fine-tune with NeMo AutoModel (official DGX Spark playbook container + pinned source) and merge
docker pull nvcr.io/nvidia/nemo-automodel:26.08
scripts/05_train_nemo_automodel.sh
scripts/06_merge_lora.sh

# 7    retrieval index
.venv/bin/python scripts/08_build_index.py

# 8    serve + app  (Ollama must be running; `ollama pull nemotron-3-super` for expert mode)
scripts/06b_export_gguf.py && (cd deploy && ollama create nemotron-nano-4b-jensen -f Modelfile)
deploy/run_app.sh               # web UI on :8600  (expert = Nemotron-3-Super, fast = fine-tuned Nano)
```

Make it a permanent service on the Spark:

```bash
mkdir -p ~/.config/systemd/user && cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now jensen-insight
loginctl enable-linger $USER
```

### Precise mode (NeMo Agent Toolkit)

```bash
python3 -m venv .venv-nat && .venv-nat/bin/pip install "nvidia-nat[langchain]" greenlet && .venv-nat/bin/pip install -e nat/
.venv-nat/bin/nat run  --config_file configs/nat/jensen_precise.yml --input "What is Vera Rubin and when is it in production?"
.venv-nat/bin/nat serve --config_file configs/nat/jensen_precise.yml --port 8601     # or: systemctl --user enable --now jensen-nat
.venv-nat/bin/pip install "nvidia-nat[eval]" && .venv-nat/bin/nat eval --config_file configs/nat/jensen_eval.yml
```

The app proxies the workflow at `POST /api/precise` (same SSE events as `/api/chat`) and exposes the tools it uses at `POST /api/retrieve` and `POST /api/verify`.

**`nat eval` result (6 held-out questions, `eval/nat_dataset.json`, Oct 25 2026, judge = Nemotron-3-Super):**

| item | sentences verified (round 1) | removed / corrected | judge score (coverage · correctness · relevance) |
|---|---|---|---|
| agentic AI at GTC Taipei | 94 % | 0 / 1 | 0.78 |
| industries with NVIDIA deals (Oct 2026) | 100 % | 0 / 0 | 0.96 |
| RoboWIZ platform (GTC Berlin) | 94 % | 2 / 0 | 1.00 |
| first programmable GPU (SIGGRAPH) | 100 % | 0 / 0 | 1.00 |
| data-center growth driver (Q2 FY27) | 94 % | 1 / 0 | 0.54 |
| Egypt / Grand Egyptian Museum partners | 97 % | 1 / 0 | 0.91 |
| **average** | **96.5 %** | 4 / 1 | **0.87** |

Every answer reached the 90 % verification target in a single round (the refine round is only triggered below 90 %, as seen in earlier runs: 0.77 → 0.90, 0.85 → 0.97). The judge's two sub-1 scores are **coverage** gaps (the briefing omitted a detail present in the reference, e.g. the CFO's "Blackwell Ultra" wording), not factual errors — the verifier's job is precision, and no judged answer contained a statement contradicting its reference. Raw outputs: `eval/nat_results/` (judge reasoning per item and the full generated briefings).

### GGUF export for Ollama (fast mode)

```bash
scripts/06b_export_gguf.py                       # -> models/gguf/nemotron-nano-4b-jensen-bf16.gguf
cd deploy && ollama create nemotron-nano-4b-jensen -f Modelfile && cd ..
```

`06b_export_gguf.py` wraps llama.cpp's converter with one fix: transformers ≥ 5 fills Nemotron-H configs with MoE
defaults, so the dense 4B is otherwise exported as `nemotron_h_moe` and crashes the Ollama runner. The Modelfile
carries the exact Nemotron-3 ChatML template with the non-thinking `<think></think>` prefix.

## NeMo fine-tuning details

The recipe follows the DGX Spark playbook ("Fine-Tune with NVIDIA NeMo") and the NeMo AutoModel SFT/PEFT guide:

* **Container + pinned source**: `nemo-automodel:26.08` with the Automodel checkout at `3b5663a6` (PR 3766, Spark unified-memory fixes), mounted over `/opt/Automodel`.
* **Model path**: `NeMoAutoModelForCausalLM` routes dense Nemotron-H to NeMo's custom `nemotron_v3` implementation (FSDP2-only, TP unsupported); `disable_mmap: true` avoids page-migration stalls on UMA.
* **LoRA**: `dim: 16`, `alpha: 32` (α = 2r, the documented sweet spot), `dropout: 0.05`, `exclude_modules: ["*.out_proj"]` because Mamba-2 kernels read `out_proj.weight` directly; Triton fused LoRA kernels on.
* **Memory**: `local_batch_size: 2`, `global_batch_size: 8` (grad-accum 4), `activation_checkpointing: true` (the `_rtx_spark` recipes' setting), bf16.
* **Data**: OpenAI-`messages` JSONL via `ChatDataset` — the tokenizer's own `chat_template.jinja` is used, loss is masked to assistant tokens, `seq_length: 2048`.
* **Optimiser**: AdamW, lr 1e-4 (10× the full-SFT default, standard for LoRA), cosine to 1e-5 with 20 warm-up steps, wd 0.01, 3 epochs, validation every 25 steps; `LOWEST_VAL` checkpoint is merged.
* **Merge**: pure-tensor merge `W' = W + (α/r)·B·A` on CPU (`scripts/06_merge_lora.py`) — the same maths as NeMo's `LinearLoRA.merge` / PEFT `merge_and_unload`, but without instantiating the model: loading Nemotron-H through transformers' remote code silently re-initialised Mamba `dt_bias`/`out_proj` and produced a garbage checkpoint, so every non-LoRA tensor is copied byte-for-byte from the base and verified (72 changed / 191 identical). The result is a drop-in replacement for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.

## How it was formulated

![Pipeline](docs/img/pipeline_infographic.png)

## Results

### Expert mode in production (Nemotron-3-Super + RAG)

Spot checks on the live app, each answer verified line-by-line against the curated transcripts:

| Question | Verified facts in the answer | Adopters section | Time |
|---|---|---|---|
| Q2 FY2027 results and Q3 guidance | $96.2 B revenue (+106 % y/y), $89 B data center (92.7 %), hyperscale $49 B / ACIE $40 B, adj. EPS $2.22, 75 % GM, Q3 guide $108 B ±2 %, $279 B supply commitments, $26 B returned ($20 B buybacks), $32 B inventory, ~70 % FY28 growth, 15–100× agentic compute | "The sources do not name specific adopters" (correct) | 78 s incl. model load |
| GTC Berlin 2026 announcements | DSX Pod, OpenShell, Sentry, Hyperion, RTX Spark, DGX Station with Windows, ~$10 B Europe commitment, 14 startup investments, 12,000+ partnerships, $10 B cost of a 3-month 1 GW delay | L'Oréal, Revolut, Spotify, Merck, Vodafone, Zara, Porsche, Nestlé, HSBC — all named in the keynote | 60 s |
| AI as manufacturing / jobs follow-up | May 29 interview, "exactly backward", radiology/law/software examples, PC-revolution analogy | — | 63 s |

### Fine-tune evaluation (fast mode model)

Held-out split: 57 Q&A pairs never seen in training (`data/sft/val.jsonl`), both models served by vLLM with
thinking off, temperature 0.2, no retrieval (pure model knowledge). Judge: Nemotron-3-Super scoring factual agreement
with the reference 1–5. Full per-question outputs in `eval/results.jsonl`.

| Model | Token F1 | ROUGE-L | Judge (1–5) | Wins (F1) |
|---|---|---|---|---|
| Nemotron-3-Nano-4B base | 0.313 | 0.207 | 1.42 | 10 / 57 |
| **Nemotron-3-Nano-4B + Jensen LoRA** | **0.380** (+21%) | **0.273** (+32%) | **1.91** (+35%) | **47 / 57** |

What changed qualitatively: the base model's knowledge stops before the window, so it typically answers
*"SIGGRAPH 2026 has not occurred yet"* or substitutes H100-era facts; the fine-tuned model knows the events happened,
who was there and what was said, but still misremembers exact figures (e.g. Q1 FY27 revenue). That is exactly the gap
the RAG layer closes — in the app the same question returns the correct figures with citations. Training curve: train
loss 2.5 → 1.06, validation loss 2.01 → 1.72 (best at step 165 of 249; later epochs start to overfit, so the
`LOWEST_VAL` checkpoint is the one merged). Wall-clock on GB10: 6.5 min training, 1 min merge.

Live check of the fine-tuned 4B alone (fast mode, `POST /api/answer`, ~30 s):

> *What did Jensen Huang announce at GTC Berlin 2026?* → **Summary:** DSX Pod — seven Vera Rubin racks in a
> road-transportable shipping container — plus the containerised AI-factory platform, keynote 21 Oct 2026 [1][2];
> **Technical view:** integrated memory/networking/power/liquid cooling, extreme co-design; **Business view:** ~$10 B
> committed to Europe, thousands of partners, neocloud supply-chain implications; **Real-world use cases:** on-site
> factory AI, telecom network intelligence, sovereign AI; **Who is already using it:** Mercedes-Benz, Volkswagen
> (Hyperion), European developers on OpenShell [1]–[10]; **Insights:** AI-as-manufacturing thesis, five-layer cake,
> neocloud credit risk to watch.

The 4B model occasionally over-generalises in the adopters section (e.g. attaching a company to the wrong product);
every claim carries a citation so it can be checked in one click.


## Data sources (6-month window)

Primary: GTC Taipei/Computex 2026 keynote (Jun 1), GTC Taipei financial analyst Q&A (Jun 3), BofA / TD Cowen /
Goldman Sachs conferences, 2026 annual stockholder meeting (Jun 24), SIGGRAPH 2026 keynote (Jul 20), Q1 FY27
(May 20) and Q2 FY27 (Aug 26) earnings calls, CMU 2026 commencement, GTC Paris supercut, Invest in Germany Summit
fireside (Oct 19), GTC Berlin 2026 keynote (Oct 21) and interviews (Bloomberg, CNBC, Lex Fridman, Open Interest,
Dreamforce, Korea Society), plus NVIDIA Newsroom press releases and NVIDIA Blog posts from the period.

All text is used for research/educational fine-tuning; original content belongs to its publishers. The model is
released under the NVIDIA Nemotron Open Model License (same as the base).

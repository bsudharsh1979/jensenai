# JensenAI — NVIDIA Insight: Jensen Huang Q&A on DGX Spark

![NVIDIA Insight on DGX Spark](docs/img/app_infographic.jpg)

A fully local, end-to-end pipeline that fine-tunes **NVIDIA Nemotron-3-Nano-4B** on everything Jensen Huang said in
public during the last six months (Apr 24 – Oct 24 2026: GTC Taipei/Computex, SIGGRAPH, GTC Berlin, Q1/Q2 FY27
earnings calls, analyst conferences, commencement speeches, interviews and podcasts) and serves it as a
retrieval-augmented Q&A web app on a single **DGX Spark (GB10, 128 GB unified memory)**.

```
YouTube captions + transcripts + NVIDIA newsroom/blog + press  ──►  NeMo-Curator-style curation
        │                                                                  │
        │                      Nemotron-3-Super (teacher, Ollama) ◄────────┤  context-grounded Q&A synthesis
        │                                                                  ▼
        │                      NeMo AutoModel LoRA SFT (nvcr.io/nvidia/nemo-automodel:26.08) on Nemotron-3-Nano-4B
        │                                                                  ▼
        └── Nemotron-3-Embed-1B + BM25 hybrid index ──► FastAPI app ◄── vLLM (nvcr.io/nvidia/vllm:26.02-py3)
```

> **Executive summary with infographics:** [docs/EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md)

## Key features, main benefits and added features

![Key features, main benefits, added features](docs/img/features_benefits_infographic.jpg)

**Key features**

* **Fine-tuned Nemotron-3-Nano-4B** — LoRA SFT on 6 months of Jensen Huang keynotes, earnings calls, interviews and NVIDIA press, trained with NeMo AutoModel on the DGX Spark itself.
* **Hybrid RAG with citations** — Nemotron-3-Embed-1B dense retrieval fused with BM25; every answer carries `[n]` citations that link to the source speech or article.
* **Runs 100% on DGX Spark** — teacher, training, embeddings, inference and web app on one GB10 box; nothing leaves the machine.
* **Streaming web chat UI** — token-by-token SSE answers, sources panel, RAG on/off toggle, example questions.

**Main benefits**

* **Private** — no cloud API, no data egress, no per-token cost.
* **Accurate** — retrieval keeps dates, dollar figures and attributions exact; the fine-tune supplies the vocabulary and recall a stock 4B model lacks (the base model answers GTC Berlin 2026 questions with H100-era facts).
* **Fast** — a 4B model answers in seconds on the Spark while leaving >100 GB free for other workloads.
* **Stays current** — re-run the nine numbered scripts to extend the corpus and re-train in well under an hour (collection aside, training takes ~7 minutes).

**Added features**

* **Thinking mode on/off** per request (`chat_template_kwargs.enable_thinking`) through the Nano-v3 reasoning parser.
* **OpenAI-compatible API** on `:8000` (vLLM) and a simple `/api/answer` JSON endpoint on `:8600`, so any client or agent framework can use it.
* **Systemd services** — `jensen-vllm` and `jensen-insight` start with the Spark.
* **Built-in evaluation** — base vs fine-tuned token-F1, ROUGE-L and Nemotron-3-Super LLM-as-judge on a held-out split.
* **Alternative Ollama/GGUF path** and an **auto-generated pipeline infographic** rendered from the measured artefacts.

## Architecture and methods

| Area | What it does | Why it matters |
|---|---|---|
| **Model choice** | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` — 4B hybrid Mamba-2/Transformer SLM, 262K context, Nemotron Open Model License, *explicitly targeted at DGX Spark / Jetson Thor / RTX* | Fits comfortably next to other workloads on the Spark's 128 GB; dense (no MoE routing to manage); the Nemotron 3 family's best accuracy-per-watt for edge agents |
| **Data collection** | `yt-dlp` pulls captions for every keynote/interview found by 16 date-sorted YouTube searches; `trafilatura` extracts transcripts (StockAnalysis, Investing.com, ScribeHawk, FactSet PDF), NVIDIA Newsroom/Blog RSS and Bing News RSS for press coverage | 300+ raw documents, ~900K words, all from the 6-month window |
| **Curation (NeMo Curator stages)** | Unicode/boilerplate cleaning → language ID → heuristic quality filters (length, symbol ratio, repeated n-grams) → exact dedup → near-dup dedup with hashed 5-gram shingles (catches re-streams of the same keynote) → relevance & time-window filters → sentence-aware chunking | 173 docs / 2,389 passages / 677K words of clean, de-duplicated, on-topic text |
| **Synthetic SFT data** | `nvidia/nemotron-3-super` (Ollama, thinking off) writes 3 self-contained, source-attributed Q&A pairs per passage; teacher is told to attribute statements only when the speaker is clear | 719 Q&A pairs (662 train / 57 held-out) from 260 prioritised passages; Nemotron post-training recipe for domain data: a larger Nemotron teacher distils knowledge *and* an "attributed answer" style into the small student |
| **Fine-tuning (NeMo best practices)** | NeMo AutoModel recipe `configs/nemotron_nano_4b_jensen_lora_rtx_spark.yaml` derived from NVIDIA's official `nemotron_nano_4b_squad_peft.yaml` and the `*_rtx_spark.yaml` memory settings: LoRA r=16 / α=32 / dropout 0.05 on all linear layers except Mamba `out_proj`, AdamW 1e-4 cosine with warm-up, bf16, FSDP2 + activation checkpointing, answer-only loss on the model's own chat template (`ChatDataset`) | Runs in the DGX Spark-validated `nemo-automodel:26.08` container with the pinned Spark fix commit; 19.2M trainable params (0.48%), ~8.7 GB peak memory, 249 steps in 6.5 min |
| **Merge & export** | LoRA merged into a standalone HF checkpoint (`scripts/06_merge_lora.py`), optional GGUF export for Ollama | One artefact deployable by vLLM, TensorRT-LLM, Ollama |
| **Serving** | vLLM container recommended by the model card for Spark/Thor, with `--mamba_ssm_cache_dtype float32` and the Nano-v3 reasoning parser; OpenAI-compatible API | Drop-in for any OpenAI client; thinking mode can be toggled per request |
| **RAG** | `nvidia/Nemotron-3-Embed-1B-BF16` dense retrieval + BM25, reciprocal-rank fusion, per-document cap, inline `[n]` citations with links back to the speech/article | Fine-tune gives the model Jensen's recent worldview and vocabulary; retrieval keeps numbers and dates exact and verifiable |
| **App** | FastAPI + streaming SSE chat UI, `/api/answer` JSON endpoint, corpus browser, systemd user units for boot-time start | Runs as a service on the DGX Spark at `http://<spark>:8600` |
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
app/
  server.py  retriever.py  llm.py  static/index.html
deploy/
  vllm_serve.sh  run_app.sh  Modelfile  systemd/jensen-vllm.service  systemd/jensen-insight.service
data/   curated/stats.json, sft/train.jsonl, sft/val.jsonl, index/meta.json (large artefacts are git-ignored)
eval/   summary.json
docs/   EXECUTIVE_SUMMARY.md, img/app_infographic.jpg, img/pipeline_infographic.{svg,png}
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

# 8    serve + app
docker pull nvcr.io/nvidia/vllm:26.02-py3
deploy/vllm_serve.sh &          # OpenAI API on :8000  (served name: nemotron-nano-4b-jensen)
deploy/run_app.sh               # web UI on :8600
```

Make it a permanent service on the Spark:

```bash
mkdir -p ~/.config/systemd/user && cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now jensen-vllm jensen-insight
loginctl enable-linger $USER
```

### Alternative backend: Ollama

`scripts/06_merge_lora.sh` produces a HF checkpoint; convert it with llama.cpp and register it in Ollama
(the Open WebUI stack already on the Spark can then use it):

```bash
python ~/llama.cpp/convert_hf_to_gguf.py models/nemotron-nano-4b-jensen-merged --outtype bf16 \
       --outfile models/gguf/nemotron-nano-4b-jensen-bf16.gguf
ollama create nemotron-nano-4b-jensen -f deploy/Modelfile
LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=nemotron-nano-4b-jensen deploy/run_app.sh
```

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

Live check on the Spark (fine-tuned model + RAG, `POST /api/answer`, 13 s):

> *What did Jensen Huang announce at GTC Berlin 2026?* → "…NVIDIA has committed nearly $10 billion to the European AI
> ecosystem and is working with 12,000+ partners across Europe… full-stack AI factory spans hardware, system software,
> CUDA-X libraries, and models…" with six cited sources (NVIDIA blog, keynote transcript, live-blog, YouTube).


## Data sources (6-month window)

Primary: GTC Taipei/Computex 2026 keynote (Jun 1), GTC Taipei financial analyst Q&A (Jun 3), BofA / TD Cowen /
Goldman Sachs conferences, 2026 annual stockholder meeting (Jun 24), SIGGRAPH 2026 keynote (Jul 20), Q1 FY27
(May 20) and Q2 FY27 (Aug 26) earnings calls, CMU 2026 commencement, GTC Paris supercut, Invest in Germany Summit
fireside (Oct 19), GTC Berlin 2026 keynote (Oct 21) and interviews (Bloomberg, CNBC, Lex Fridman, Open Interest,
Dreamforce, Korea Society), plus NVIDIA Newsroom press releases and NVIDIA Blog posts from the period.

All text is used for research/educational fine-tuning; original content belongs to its publishers. The model is
released under the NVIDIA Nemotron Open Model License (same as the base).

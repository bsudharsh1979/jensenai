"""Turn human good/bad feedback into DPO preference data for NeMo AutoModel (offline half of the reinforcement layer).

Pairs are formed per question family (embedding cosine >= 0.85 between questions): every GOOD answer is a "chosen"
and every BAD answer to the same family is a "rejected". When a family has only BAD answers, the chosen side is
regenerated on demand with the expert model via the running app (--regenerate), so the fast model learns from Super.

Output: data/sft/dpo.jsonl with {"prompt": [...messages...], "chosen": str, "rejected": str} and a NeMo AutoModel DPO
config at configs/nemotron_nano_4b_jensen_dpo_rtx_spark.yaml (preference fine-tune of the merged SFT checkpoint).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SFT  # noqa: E402

FEEDBACK = ROOT / "data" / "feedback" / "feedback.jsonl"
OUT = SFT / "dpo.jsonl"
CFG = ROOT / "configs" / "nemotron_nano_4b_jensen_dpo_rtx_spark.yaml"

DPO_CFG = """# NeMo AutoModel DPO recipe for the fast model (derived from the LoRA SFT recipe in this repo).
# Run inside nvcr.io/nvidia/nemo-automodel:26.08:  automodel finetune llm -c configs/nemotron_nano_4b_jensen_dpo_rtx_spark.yaml
step_scheduler:
  num_epochs: 2
  ckpt_every_steps: 50
  val_every_steps: 25
  global_batch_size: 8
  local_batch_size: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /work/models/nemotron-nano-4b-jensen-merged   # SFT merge = policy init
  torch_dtype: bfloat16
  trust_remote_code: true

peft:
  peft_fn: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true
  exclude_modules: ["*.out_proj"]
  dim: 16
  alpha: 32
  dropout: 0.05

loss_fn:
  _target_: nemo_automodel.components.loss.dpo.DPOLoss   # sigmoid DPO, beta 0.1
  beta: 0.1

dataset:
  _target_: nemo_automodel.components.datasets.llm.preference_dataset.PreferenceDataset
  path_or_dataset: /work/data/sft/dpo.jsonl
  prompt_key: prompt
  chosen_key: chosen
  rejected_key: rejected
  seq_length: 4096

optimizer:
  _target_: torch.optim.AdamW
  lr: 5.0e-6
  weight_decay: 0.0

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: 1
  tp_size: 1

checkpoint:
  enabled: true
  checkpoint_dir: /work/runs/dpo
  save_consolidated: true
"""


def embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("nvidia/Nemotron-3-Embed-1B-BF16", trust_remote_code=True)
    return m.encode_query(texts, convert_to_numpy=True, normalize_embeddings=True)


def regenerate(question: str, base: str) -> str:
    body = json.dumps({"messages": [{"role": "user", "content": question}], "mode": "expert"}).encode()
    req = urllib.request.Request(f"{base}/api/answer", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["answer"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="ask the running app (expert mode) for a chosen answer when a family has only bad ones")
    ap.add_argument("--app", default="http://127.0.0.1:8600")
    ap.add_argument("--sim", type=float, default=0.85)
    args = ap.parse_args()

    recs = [json.loads(l) for l in FEEDBACK.read_text().splitlines() if l.strip()] if FEEDBACK.exists() else []
    if not recs:
        print("no feedback yet:", FEEDBACK); return
    import numpy as np
    E = embed([r["question"] for r in recs])
    sims = E @ E.T
    assigned = [-1] * len(recs); families: list[list[int]] = []
    for i in range(len(recs)):
        if assigned[i] >= 0:
            continue
        fam = [j for j in range(len(recs)) if assigned[j] < 0 and sims[i, j] >= args.sim]
        for j in fam:
            assigned[j] = len(families)
        families.append(fam)

    pairs = []
    for fam in families:
        good = [recs[j] for j in fam if recs[j]["rating"] == "good"]
        bad = [recs[j] for j in fam if recs[j]["rating"] == "bad"]
        if not bad:
            continue
        q = recs[fam[0]]["question"]
        chosen = [g["answer"] for g in good]
        if not chosen and args.regenerate:
            print("regenerating chosen for:", q[:80]); chosen = [regenerate(q, args.app)]
        for c in chosen:
            for b in bad:
                pairs.append({"prompt": [{"role": "user", "content": q}], "chosen": c, "rejected": b["answer"],
                              "meta": {"bad_comment": b.get("comment"), "mode": b.get("mode")}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    CFG.write_text(DPO_CFG)
    print(f"{len(recs)} ratings → {len(families)} question families → {len(pairs)} DPO pairs → {OUT}\nconfig: {CFG}")


if __name__ == "__main__":
    main()

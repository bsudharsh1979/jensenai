#!/usr/bin/env python
"""Merge a NeMo AutoModel LoRA adapter into the Nemotron base and save a standalone HF checkpoint.

Pure-tensor merge (W' = W + alpha/r * B @ A), identical to NeMo AutoModel's LinearLoRA.merge and
PEFT's merge_and_unload — but it never instantiates the model. Loading Nemotron-H through
transformers' remote code re-initialises Mamba `dt_bias`/`out_proj` on some versions, which silently
corrupts a merged checkpoint; copying every non-LoRA tensor byte-for-byte from the base avoids that.
Runs on CPU, needs no GPU and no container.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

EXTRA_FILES = [
    "config.json", "configuration_nemotron_h.py", "modeling_nemotron_h.py", "chat_template.jinja",
    "nano_v3_reasoning_parser.py", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    adapter = Path(args.adapter_path)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    print(f"adapter r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale}")

    lora: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(adapter / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            # base_model.model.backbone.layers.0.mixer.in_proj.lora_A.weight -> backbone.layers.0.mixer.in_proj
            mod, ab = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
            lora.setdefault(mod, {})[ab.split(".")[0]] = f.get_tensor(k)
    print(f"{len(lora)} LoRA-adapted modules")

    snap = Path(snapshot_download(args.base_model))
    merged: dict[str, torch.Tensor] = {}
    touched = 0
    for shard in sorted(snap.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                w = f.get_tensor(k)
                mod = k.removesuffix(".weight")
                if k.endswith(".weight") and mod in lora:
                    a, b = lora[mod]["A"].float(), lora[mod]["B"].float()
                    w = (w.float() + scale * (b @ a)).to(w.dtype)
                    touched += 1
                merged[k] = w.contiguous()
    missing = set(lora) - {k.removesuffix(".weight") for k in merged}
    if missing:
        raise SystemExit(f"adapter modules not found in base: {sorted(missing)[:5]}")
    print(f"merged {touched}/{len(merged)} tensors")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, out / "model.safetensors", metadata={"format": "pt"})
    for name in EXTRA_FILES:
        if (snap / name).exists():
            shutil.copy(snap / name, out / name)
    # the base generation_config sets top_p without do_sample; make it explicit for strict loaders
    gc = json.loads((snap / "generation_config.json").read_text())
    gc["do_sample"] = True
    (out / "generation_config.json").write_text(json.dumps(gc, indent=2))
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Export the merged Nemotron-3-Nano-4B checkpoint to GGUF for Ollama / llama.cpp.

Thin wrapper around llama.cpp's convert_hf_to_gguf.py with one fix: the converter loads hyper-parameters through
transformers' AutoConfig, and transformers >= 5 fills Nemotron-H configs with MoE defaults (num_experts_per_tok=2,
n_routed_experts, ...). The dense 4B then gets mislabelled as `nemotron_h_moe` and the Ollama runner crashes on load.
Here hparams are read from the raw config.json instead, so the dense model exports as `nemotron_h`.

Usage: scripts/06b_export_gguf.py [--model models/nemotron-nano-4b-jensen-merged] [--outfile models/gguf/<name>.gguf]
Requires a llama.cpp checkout (LLAMA_CPP, default ~/llama.cpp).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA = Path(os.environ.get("LLAMA_CPP", Path.home() / "llama.cpp"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "nemotron-nano-4b-jensen-merged"))
    ap.add_argument("--outfile", default=str(ROOT / "models" / "gguf" / "nemotron-nano-4b-jensen-bf16.gguf"))
    ap.add_argument("--outtype", default="bf16")
    args = ap.parse_args()

    sys.path.insert(0, str(LLAMA))
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    import convert_hf_to_gguf as conv  # noqa: E402

    def raw_hparams(dir_model: Path, is_mistral_format: bool = False) -> dict:
        h = json.loads((Path(dir_model) / "config.json").read_text())
        if "text_config" in h:
            h = {**h, **h["text_config"]}
        return h

    conv.ModelBase.load_hparams = staticmethod(raw_hparams)
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    sys.argv = ["convert_hf_to_gguf.py", args.model, "--outtype", args.outtype, "--outfile", args.outfile]
    conv.main()


if __name__ == "__main__":
    main()

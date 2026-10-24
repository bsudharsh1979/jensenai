#!/usr/bin/env bash
# Merge the trained LoRA adapter into the base model -> standalone HF checkpoint.
# Pure-tensor merge on the host (CPU, ~1 min); no GPU or container required.
# Usage: scripts/06_merge_lora.sh [adapter_dir_in_repo] [output_dir_in_repo]
#   default adapter: checkpoints/nemotron_nano_4b_jensen_lora/LOWEST_VAL/model
#   default output : models/nemotron-nano-4b-jensen-merged
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER="${1:-checkpoints/nemotron_nano_4b_jensen_lora/LOWEST_VAL/model}"
OUT="${2:-models/nemotron-nano-4b-jensen-merged}"
BASE="${BASE_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$ROOT/models"

"$PY" "$ROOT/scripts/06_merge_lora.py" --base-model "$BASE" --adapter-path "$ROOT/$ADAPTER" --output-dir "$ROOT/$OUT"
ls -la "$ROOT/$OUT"
echo "merged model: $ROOT/$OUT"

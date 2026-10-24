#!/usr/bin/env bash
# Serve the merged fine-tuned Nemotron-3-Nano-4B with vLLM on DGX Spark (OpenAI-compatible API on :8000).
# Uses the NVIDIA vLLM container recommended for Spark/Thor in the Nemotron-3-Nano model card, plus the
# model card's serving flags (mamba_ssm_cache_dtype float32, Nano-v3 reasoning parser).
#
# Usage: deploy/vllm_serve.sh [model_dir_or_hf_id] [served_name] [port]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/models/nemotron-nano-4b-jensen-merged}"
NAME="${2:-nemotron-nano-4b-jensen}"
PORT="${3:-8000}"
IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.02-py3}"
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"   # 4B model: ~25% of 128 GB leaves room for Ollama/other apps
CONTAINER="${VLLM_CONTAINER_NAME:-vllm-$NAME}"

if [ -d "$MODEL" ]; then
  MOUNT=(-v "$MODEL:/model:ro"); MODEL_ARG=/model
else
  MOUNT=(); MODEL_ARG="$MODEL"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
exec docker run --name "$CONTAINER" --rm --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME_HOST/token" 2>/dev/null || true)}" \
  -v "$HF_HOME_HOST:/root/.cache/huggingface" "${MOUNT[@]}" \
  "$IMAGE" bash -lc "
    cd /tmp && ( [ -f nano_v3_reasoning_parser.py ] || cp ${MODEL_ARG}/nano_v3_reasoning_parser.py . 2>/dev/null || \
      wget -q https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/main/nano_v3_reasoning_parser.py ) ;
    vllm serve '$MODEL_ARG' --served-model-name '$NAME' --host 0.0.0.0 --port $PORT \
      --max-num-seqs 8 --max-model-len $MAX_LEN --gpu-memory-utilization $GPU_UTIL \
      --trust-remote-code --mamba_ssm_cache_dtype float32 \
      --reasoning-parser-plugin /tmp/nano_v3_reasoning_parser.py --reasoning-parser nano_v3"

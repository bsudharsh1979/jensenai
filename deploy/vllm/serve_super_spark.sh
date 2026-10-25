#!/usr/bin/env bash
# Serve Nemotron-3-Super-120B-A12B (NVFP4) with vLLM on a single DGX Spark — OpenAI-compatible API on :8001.
#
# This is NVIDIA's documented Super-on-Spark recipe (NVIDIA-NeMo/Nemotron usage-cookbook, SparkDeploymentGuide):
# vLLM nightly cu130 image, NVFP4 weights on the Marlin FP4 backend, FP8 KV cache, FP32 Mamba SSM cache, chunked
# prefill, async scheduling and MTP speculative decoding (3 draft tokens from the checkpoint's built-in MTP head).
# Deviations from the guide, all for coexisting with the other workloads on this 128 GB box:
#   * --max-model-len 65536 instead of 1M (the app uses <= 32K) and --gpu-memory-utilization 0.70 (~85 GB) so the
#     fine-tuned Nano-4B in Ollama, the embedder and other GPU users keep ~35 GB;
#   * prefix caching stays OFF by default (VLLM_PREFIX_CACHING=1 to opt in): Super's Mamba-2 recurrent state is not
#     safely prefix-cacheable (the app saw stale-state replays with llama.cpp and adds a per-request nonce for that).
# Ollama's copy of Super must NOT be resident at the same time (91 GB) — deploy/vllm/unload_ollama_super.sh.
#
# Usage: deploy/vllm/serve_super_spark.sh            (env overrides below)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${SUPER_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
NAME="${SUPER_SERVED_NAME:-nemotron-3-super}"
PORT="${SUPER_PORT:-8001}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:cu130-nightly}"
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
SPEC_TOKENS="${VLLM_MTP_TOKENS:-3}"
CONTAINER="${VLLM_CONTAINER_NAME:-vllm-super}"
PARSER="$ROOT/deploy/vllm/super_v3_reasoning_parser.py"
PREFIX_FLAG="--no-enable-prefix-caching"
[ "${VLLM_PREFIX_CACHING:-0}" = "1" ] && PREFIX_FLAG="--enable-prefix-caching"

if [ ! -f "$PARSER" ]; then
  curl -fsSL -o "$PARSER" https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
exec docker run --rm --name "$CONTAINER" --gpus all --ipc=host --network host \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$HF_HOME_HOST:/root/.cache/huggingface" \
  -v "$PARSER:/app/super_v3_reasoning_parser.py:ro" \
  "$IMAGE" \
    --model "$MODEL" \
    --served-model-name "$NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-chunked-prefill \
    $PREFIX_FLAG \
    --max-num-seqs "$MAX_SEQS" \
    --max-model-len "$MAX_LEN" \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative_config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS,\"moe_backend\":\"triton\"}" \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

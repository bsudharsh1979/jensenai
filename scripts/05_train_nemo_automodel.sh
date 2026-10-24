#!/usr/bin/env bash
# LoRA SFT with NeMo AutoModel inside NVIDIA's DGX Spark-validated container.
# Follows https://build.nvidia.com/spark/nemo-fine-tune (container 26.08 + pinned Automodel source).
#
# Usage: scripts/05_train_nemo_automodel.sh [extra automodel overrides...]
#   e.g. scripts/05_train_nemo_automodel.sh --step_scheduler.max_steps 2    # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${NEMO_AUTOMODEL_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.08}"
AUTOMODEL_SRC="${AUTOMODEL_SRC:-$HOME/Automodel}"          # pinned to 3b5663a6 (PR 3766, Spark memory fixes)
RECIPE="${RECIPE:-/workspace/configs/nemotron_nano_4b_jensen_lora_rtx_spark.yaml}"
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
MEM="${CONTAINER_MEM:-110g}"   # 128 GB Spark; leave headroom for the host. Use 56g on a 64 GB Spark.

if [ ! -d "$AUTOMODEL_SRC" ]; then
  echo "Cloning NeMo AutoModel (pinned) into $AUTOMODEL_SRC"
  git clone https://github.com/NVIDIA-NeMo/Automodel.git "$AUTOMODEL_SRC"
  git -C "$AUTOMODEL_SRC" checkout --detach 3b5663a6fdf0fee03abbe82e431ca3a7ba6dd8e2
fi

HF_TOKEN_VALUE="${HF_TOKEN:-$(cat "$HF_HOME_HOST/token" 2>/dev/null || true)}"
mkdir -p "$ROOT/checkpoints" "$ROOT/logs"

exec docker run --rm \
  --gpus all --ipc=host \
  --memory="$MEM" --memory-swap="$MEM" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_TOKEN="$HF_TOKEN_VALUE" -e HF_HOME=/root/.cache/huggingface \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
  -v "$HF_HOME_HOST:/root/.cache/huggingface" \
  -v "$AUTOMODEL_SRC:/opt/Automodel" \
  -v "$ROOT:/workspace" \
  -w /opt/Automodel \
  --entrypoint /usr/bin/bash \
  "$IMAGE" -lc "automodel $RECIPE --nproc-per-node 1 $*" 2>&1 | tee "$ROOT/logs/05_train_$(date +%Y%m%d_%H%M%S).log"

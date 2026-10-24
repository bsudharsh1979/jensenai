#!/usr/bin/env bash
# Start the NVIDIA Insight web app (FastAPI) on DGX Spark.
# Env: LLM_BASE_URL (default vLLM http://127.0.0.1:8000/v1), LLM_MODEL, APP_PORT (default 8600)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export LLM_MODEL="${LLM_MODEL:-nemotron-nano-4b-jensen}"
export APP_PORT="${APP_PORT:-8600}"
exec "$ROOT/.venv/bin/uvicorn" app.server:app --host 0.0.0.0 --port "$APP_PORT" --log-level info

#!/usr/bin/env bash
# Evict Ollama's copy of Nemotron-3-Super (~91 GB resident) so the vLLM NVFP4 server can start on the same
# 128 GB DGX Spark. Does not stop Ollama (the fine-tuned Nano-4B fast model keeps being served from it).
set -euo pipefail
OLLAMA="${OLLAMA_URL:-http://127.0.0.1:11434}"
for m in $(curl -fsS "$OLLAMA/api/ps" | python3 -c 'import sys,json;[print(x["name"]) for x in json.load(sys.stdin).get("models",[]) if "super" in x["name"]]'); do
  echo "unloading $m from Ollama"
  curl -fsS "$OLLAMA/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
done
sleep 3
curl -fsS "$OLLAMA/api/ps" | python3 -c 'import sys,json;print("still loaded:", [x["name"] for x in json.load(sys.stdin).get("models",[])])'

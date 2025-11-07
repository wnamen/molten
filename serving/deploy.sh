#!/bin/bash
# Deploy K2-Instruct with vLLM on cloud GPU
# Optimized for tight budget: single A100, spot instance

set -e

MODEL_PATH="${MODEL_PATH:-MoonshotAI/Kimi-K2-Instruct}"
MAX_LEN="${MAX_MODEL_LEN:-65536}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "🚀 Deploying K2-Instruct MLE-bench API"
echo "Model: $MODEL_PATH"
echo "Max context: $MAX_LEN"
echo "Tensor parallel: $TENSOR_PARALLEL"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Ensure GPU is available."
    exit 1
fi

nvidia-smi

# Start server
python3 serve.py \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --max-model-len "$MAX_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL"


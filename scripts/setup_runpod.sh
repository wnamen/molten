#!/bin/bash
# Complete setup script for RunPod deployment
# Run this after SSH'ing into your pod

set -e

echo "🚀 Setting up Molten MLE-bench POC on RunPod"
echo "=============================================="

# Update system
echo "📦 Updating system packages..."
apt-get update -qq

# Install git if not present
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    apt-get install -y git
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install torch transformers accelerate openai requests numpy pandas fastapi uvicorn pydantic

# Install vLLM (this may take a few minutes)
echo "📦 Installing vLLM (this may take a few minutes)..."
pip install vllm

# Install MLE-bench (will need Kaggle creds later)
echo "📦 Installing MLE-bench..."
pip install -e git+https://github.com/openai/mle-bench.git#egg=mlebench || echo "⚠️  MLE-bench install failed - you may need to set up Kaggle creds first"

# Install Kaggle API
pip install kaggle

# Check GPU
echo "🔍 Checking GPU..."
nvidia-smi

# Check PyTorch/CUDA
echo "🔍 Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set up Kaggle credentials: mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json"
echo "2. Clone your repo or upload code"
echo "3. Start serving: cd serving && python3 serve.py"
echo "4. In another terminal, test: curl http://localhost:8000/health"


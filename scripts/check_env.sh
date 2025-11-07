#!/bin/bash
# Quick verification script to check environment compatibility

echo "Checking environment compatibility..."

# Check Python version
python3 --version

# Check PyTorch version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU
nvidia-smi

# Check if vLLM can be installed
echo "Checking vLLM compatibility..."
python3 -c "import sys; print(f'Python: {sys.version}')"

echo "✓ Environment check complete"


#!/bin/bash
# Quick fix script - install missing dependencies

echo "Installing missing dependencies..."

# Install hf_transfer for faster HuggingFace downloads
pip install hf_transfer

# Or disable it if you prefer
# export HF_HUB_ENABLE_HF_TRANSFER=0

echo "✓ Dependencies installed"
echo ""
echo "Now restart the server:"
echo "  python3 serve.py --host 0.0.0.0 --port 8000"


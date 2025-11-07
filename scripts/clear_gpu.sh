#!/bin/bash
# Clear GPU memory by killing processes using CUDA

echo "Checking for processes using GPU..."

# Find processes using GPU
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    if [ ! -z "$pid" ]; then
        echo "Killing process $pid using GPU..."
        kill -9 $pid 2>/dev/null || true
    fi
done

# Clear PyTorch cache
python3 << EOF
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ Cleared PyTorch CUDA cache")
else:
    print("No CUDA available")
EOF

# Wait a moment
sleep 2

# Check GPU memory
echo ""
echo "Current GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo ""
echo "✓ GPU memory cleared. You can now start the server."


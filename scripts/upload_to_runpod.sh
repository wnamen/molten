#!/bin/bash
# Transfer molten codebase to RunPod pod
# Usage: ./upload_to_runpod.sh

POD_USER="n28b6uuyy3fhiy-64411d8a"
POD_HOST="ssh.runpod.io"
SSH_KEY="/Users/williamnamen/projects/molten/id_ed25519"
REMOTE_DIR="/workspace/molten"

echo "📤 Uploading molten codebase to RunPod..."
echo "Target: ${POD_USER}@${POD_HOST}:${REMOTE_DIR}"

# Create remote directory
ssh -i "$SSH_KEY" "${POD_USER}@${POD_HOST}" "mkdir -p ${REMOTE_DIR}"

# Upload files (excluding large/unnecessary files)
rsync -avz --progress \
  -e "ssh -i $SSH_KEY" \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'workspace/' \
  --exclude '*.log' \
  --exclude '*.csv' \
  --exclude 'lora_adapters/' \
  --exclude '.cache/' \
  ./ "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/"

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next: SSH into pod and run setup:"
echo "  ssh -i $SSH_KEY ${POD_USER}@${POD_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  bash scripts/setup_runpod.sh"


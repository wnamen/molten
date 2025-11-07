#!/bin/bash
# Set up SSH keys in a container for git operations
# Usage: Run this script inside the container after copying your SSH key

set -e

echo "🔑 Setting up SSH keys for git..."

# Create .ssh directory if it doesn't exist
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Check if private key was provided as argument or needs to be pasted
if [ -z "$1" ]; then
    echo "⚠️  No SSH key path provided."
    echo ""
    echo "Option 1: Copy your private key into the container first:"
    echo "  From your local machine:"
    echo "  scp -P <port> ~/.ssh/id_ed25519 root@<container-ip>:~/.ssh/"
    echo ""
    echo "Option 2: Paste your private key (will prompt for input):"
    echo "  Press Ctrl+D after pasting the key"
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    cat > ~/.ssh/id_ed25519
else
    # Copy key from provided path
    cp "$1" ~/.ssh/id_ed25519
fi

# Set correct permissions
chmod 600 ~/.ssh/id_ed25519

# Add GitHub to known_hosts
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true

# Test SSH connection
echo ""
echo "🧪 Testing SSH connection to GitHub..."
if ssh -T git@github.com -o StrictHostKeyChecking=no 2>&1 | grep -q "successfully authenticated"; then
    echo "✅ SSH key is working!"
else
    echo "⚠️  SSH key may not be added to your GitHub account."
    echo "   Add your public key at: https://github.com/settings/keys"
    echo ""
    echo "   Your public key is:"
    ssh-keygen -y -f ~/.ssh/id_ed25519
fi

echo ""
echo "✅ SSH setup complete!"
echo "   You can now clone repositories using: git clone git@github.com:user/repo.git"


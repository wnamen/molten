# RunPod Deployment Guide

## Step 1: SSH into Pod

From RunPod dashboard, click "Connect" and copy the SSH command, then run it locally:

```bash
ssh root@<pod-ip> -p <port>
```

## Step 2: Initial Setup

Run the setup script (or manually install):

```bash
# Option A: If you've uploaded the repo
cd molten
bash scripts/setup_runpod.sh

# Option B: Manual setup
pip install --upgrade pip
pip install torch transformers accelerate vllm openai requests numpy pandas fastapi uvicorn pydantic
pip install kaggle
pip install -e git+https://github.com/openai/mle-bench.git#egg=mlebench
```

## Step 3: Set Up Kaggle Credentials

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# Paste your Kaggle API credentials (download from kaggle.com/settings)
chmod 600 ~/.kaggle/kaggle.json
```

## Step 4: Get Your Code on the Pod

**Option A: Git clone (if repo is pushed)**
```bash
git clone <your-repo-url>
cd molten
```

**Option B: Upload via RunPod file manager**
- Use RunPod's file manager to upload the `molten` directory
- Or use `scp` from your local machine:
  ```bash
  scp -r -P <port> /path/to/molten root@<pod-ip>:/workspace/
  ```

## Step 5: Deploy K2-Instruct Server

```bash
cd molten/serving
python3 serve.py --host 0.0.0.0 --port 8000
```

**Or run in background:**
```bash
nohup python3 serve.py --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

## Step 6: Test the Server

In another terminal (or new SSH session):

```bash
# Health check
curl http://localhost:8000/health

# Test chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.6
  }'
```

## Step 7: Set Up MLE-bench Environment

```bash
cd ../bench
python3 setup_env.py --prepare-lite
```

## Step 8: Run Baseline Evaluation

```bash
python3 run_lite.py \
  --api-base http://localhost:8000/v1 \
  --workspace /workspace/mlebench_workspace \
  --output /workspace/baseline_results.json
```

## Troubleshooting

- **Port forwarding**: RunPod may require port forwarding. Check RunPod dashboard for exposed ports.
- **Model download**: First run will download K2-Instruct (~60GB). Ensure you have enough disk space.
- **GPU memory**: If OOM, reduce `max_model_len` in `serve.py` (try 32768 instead of 65536).


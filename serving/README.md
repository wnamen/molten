# K2-Instruct Serving

Deploy Kimi-K2-Instruct with vLLM for MLE-bench agent.

## Cloud GPU Setup

### RunPod (Recommended for budget)

1. Create account at https://runpod.io
2. Create pod: A100 80GB, PyTorch 2.1, CUDA 12.1
3. SSH into pod
4. Clone repo and run:
   ```bash
   cd serving
   pip install -r requirements.txt
   ./deploy.sh
   ```

### Vast.ai

1. Search for A100 80GB spot instances
2. SSH into instance
3. Same setup as RunPod

### Lambda Labs

1. Request A100 access
2. SSH into instance
3. Same setup as RunPod

## Environment Variables

- `MODEL_PATH`: HuggingFace model path (default: `MoonshotAI/Kimi-K2-Instruct`)
- `MAX_MODEL_LEN`: Max context length (default: 65536)
- `TENSOR_PARALLEL_SIZE`: GPU parallelism (default: 1)
- `HOST`: API host (default: 0.0.0.0)
- `PORT`: API port (default: 8000)

## Usage

```bash
# Start server
./deploy.sh

# Test health
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

## Cost Optimization

- Use spot/preemptible instances
- Cap context at 64K initially
- Shut down when not running experiments
- Cache model weights on persistent volume


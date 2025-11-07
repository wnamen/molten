# Molten: K2-Instruct MLE-bench POC

Proof-of-concept to beat MLE-bench using Kimi-K2-Instruct with minimal budget.

## Quick Start

### 1. Setup Cloud GPU

- Rent spot A100 80GB from RunPod/Vast.ai/Lambda Labs (~$0.50-1.50/hr)
- SSH into instance

### 2. Deploy K2-Instruct

```bash
cd serving
pip install -r requirements.txt
./deploy.sh
```

### 3. Setup MLE-bench Environment

```bash
cd bench
pip install -r requirements.txt
python setup_env.py --prepare-lite --build-docker
```

### 4. Run Baseline

```bash
cd bench
python run_lite.py \
  --api-base http://localhost:8000/v1 \
  --workspace ./workspace \
  --output ./baseline_results.json
```

### 5. Fine-tune (Optional)

```bash
cd finetune
pip install -r requirements.txt

# Curate data
python curate_data.py --include-synthetic --output ./finetune_data.jsonl

# Train LoRA
python train_lora.py \
  --data-path ./finetune_data.jsonl \
  --output-dir ./lora_adapters
```

### 6. Re-evaluate

```bash
cd bench
python run_lite.py \
  --api-base http://localhost:8000/v1 \
  --workspace ./workspace_improved \
  --output ./improved_results.json
```

### 7. Compare Results

```bash
cd eval
python report.py \
  --compare-baseline ../bench/baseline_results.json \
  --compare-improved ../bench/improved_results.json \
  --output comparison_report.md
```

## Structure

- `serving/` - vLLM deployment for K2-Instruct
- `agent/` - MLE-bench compliant agent with tools
- `bench/` - MLE-bench orchestration and evaluation
- `finetune/` - PEFT training recipes
- `eval/` - Evaluation reports and comparisons

## Cost Optimization

- Use spot/preemptible instances
- Cap context at 64K initially
- Shut down when not running experiments
- Cache model weights on persistent volume

## References

- [Kimi-K2 Deployment Guide](https://github.com/MoonshotAI/Kimi-K2?tab=readme-ov-file#4-deployment)
- [MLE-bench README](https://github.com/openai/mle-bench/?tab=readme-ov-file)

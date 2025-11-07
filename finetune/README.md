# Fine-tuning K2-Instruct for MLE-bench

Fine-tune Kimi-K2-Instruct with LoRA adapters specialized for ML engineering tasks.

## Data Curation

```bash
# Create training dataset
python curate_data.py \
  --include-synthetic \
  --kaggle-notebooks ./kaggle_notebooks \
  --output ./finetune_data.jsonl
```

## Training

```bash
# Train LoRA adapters
python train_lora.py \
  --model-path MoonshotAI/Kimi-K2-Instruct \
  --data-path ./finetune_data.jsonl \
  --output-dir ./lora_adapters \
  --lora-r 16 \
  --lora-alpha 32 \
  --num-epochs 2 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4
```

## MoE-Aware LoRA

The training script targets attention and MLP layers while avoiding router modifications:
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

This allows efficient fine-tuning without full expert finetuning.

## Loading Adapters

After training, load adapters in vLLM or inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("MoonshotAI/Kimi-K2-Instruct")
model = PeftModel.from_pretrained(base_model, "./lora_adapters")
```


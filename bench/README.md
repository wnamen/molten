# MLE-bench Environment Setup

Setup and prepare MLE-bench datasets and Docker environment.

## Prerequisites

1. **Kaggle API credentials**: Download `kaggle.json` from https://www.kaggle.com/settings
2. **Docker**: For building the benchmark environment
3. **Git LFS**: For downloading large files

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment (lite for quick start)
python setup_env.py --prepare-lite

# Or full dataset (takes ~2 days)
python setup_env.py --prepare-all

# Build Docker environment
python setup_env.py --build-docker
```

## Environment Variables

- `MLEBENCH_CACHE`: Cache directory for datasets (default: `~/.cache/mlebench`)
- `KAGGLE_CREDS`: Path to kaggle.json (default: `~/.kaggle/kaggle.json`)

## Dataset Preparation

The lite subset is recommended for initial testing. Full dataset preparation can take up to 2 days.

After preparation, datasets are cached and can be reused across runs.


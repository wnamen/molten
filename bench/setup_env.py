#!/usr/bin/env python3
"""
MLE-bench environment setup and dataset preparation.
Handles Kaggle API, Docker environment, and dataset caching.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional

class MLEBenchEnv:
    """Manage MLE-bench environment and datasets."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        kaggle_creds_path: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir or os.getenv("MLEBENCH_CACHE", "~/.cache/mlebench")).expanduser()
        self.kaggle_creds_path = Path(
            kaggle_creds_path or os.getenv("KAGGLE_CREDS", "~/.kaggle/kaggle.json")
        ).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_kaggle_creds(self):
        """Ensure Kaggle credentials are in place."""
        kaggle_dir = self.kaggle_creds_path.parent
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.kaggle_creds_path.exists():
            print(f"⚠️  Kaggle credentials not found at {self.kaggle_creds_path}")
            print("Please download kaggle.json from https://www.kaggle.com/settings")
            print(f"and place it at {self.kaggle_creds_path}")
            return False
        
        # Ensure correct permissions
        os.chmod(self.kaggle_creds_path, 0o600)
        print(f"✓ Kaggle credentials found at {self.kaggle_creds_path}")
        return True
    
    def install_mlebench(self):
        """Install mlebench package."""
        try:
            import mlebench
            print("✓ mlebench already installed")
            return True
        except ImportError:
            print("Installing mlebench...")
            result = subprocess.run(
                ["pip", "install", "-e", "git+https://github.com/openai/mle-bench.git#egg=mlebench"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("✓ mlebench installed")
                return True
            else:
                print(f"❌ Failed to install mlebench: {result.stderr}")
                return False
    
    def prepare_lite(self):
        """Prepare lite dataset subset."""
        print("Preparing MLE-bench lite dataset...")
        result = subprocess.run(
            ["mlebench", "prepare", "--lite"],
            env={**os.environ, "MLEBENCH_CACHE": str(self.cache_dir)},
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Lite dataset prepared")
            return True
        else:
            print(f"❌ Failed to prepare lite dataset: {result.stderr}")
            return False
    
    def prepare_all(self):
        """Prepare full dataset (takes ~2 days)."""
        print("Preparing full MLE-bench dataset (this will take ~2 days)...")
        result = subprocess.run(
            ["mlebench", "prepare", "--all"],
            env={**os.environ, "MLEBENCH_CACHE": str(self.cache_dir)},
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Full dataset prepared")
            return True
        else:
            print(f"❌ Failed to prepare full dataset: {result.stderr}")
            return False
    
    def setup_docker_env(self):
        """Build MLE-bench Docker environment."""
        dockerfile_path = Path(__file__).parent.parent / "environment" / "Dockerfile"
        if not dockerfile_path.exists():
            print("⚠️  MLE-bench Dockerfile not found. Cloning repo...")
            result = subprocess.run(
                ["git", "clone", "https://github.com/openai/mle-bench.git", "mle-bench-repo"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"❌ Failed to clone mle-bench: {result.stderr}")
                return False
        
        dockerfile_path = Path("mle-bench-repo/environment/Dockerfile")
        if dockerfile_path.exists():
            print("Building MLE-bench Docker image...")
            result = subprocess.run(
                ["docker", "build", "--platform", "linux/amd64", "-t", "mlebench-env", "-f", str(dockerfile_path), "."],
                cwd=dockerfile_path.parent.parent,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("✓ Docker image built")
                return True
            else:
                print(f"❌ Failed to build Docker image: {result.stderr}")
                return False
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup MLE-bench environment")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for datasets")
    parser.add_argument("--kaggle-creds", type=str, help="Path to kaggle.json")
    parser.add_argument("--prepare-lite", action="store_true", help="Prepare lite dataset")
    parser.add_argument("--prepare-all", action="store_true", help="Prepare full dataset")
    parser.add_argument("--build-docker", action="store_true", help="Build Docker environment")
    
    args = parser.parse_args()
    
    env = MLEBenchEnv(
        cache_dir=args.cache_dir,
        kaggle_creds_path=args.kaggle_creds,
    )
    
    success = True
    success &= env.setup_kaggle_creds()
    success &= env.install_mlebench()
    
    if args.build_docker:
        success &= env.setup_docker_env()
    
    if args.prepare_lite:
        success &= env.prepare_lite()
    elif args.prepare_all:
        success &= env.prepare_all()
    
    if success:
        print("\n✓ MLE-bench environment setup complete")
    else:
        print("\n❌ Setup incomplete. Check errors above.")
        exit(1)

if __name__ == "__main__":
    main()


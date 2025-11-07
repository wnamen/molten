#!/usr/bin/env python3
"""
Run baseline evaluation on MLE-bench lite subset.
Collects metrics, logs, and scores for analysis.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.controller import MLEBenchAgent, TaskBudget

def get_lite_competitions() -> List[str]:
    """Get list of competitions in lite subset."""
    # These are typically smaller/faster competitions
    # Actual list should come from mlebench CLI or config
    return [
        "spaceship-titanic",
        "tabular-playground-series-dec-2021",
        "tabular-playground-series-may-2022",
        # Add more as needed
    ]

def run_baseline(
    api_base: str = "http://localhost:8000/v1",
    workspace_dir: Path = Path("./workspace"),
    competitions: Optional[List[str]] = None,
    max_time_per_task: int = 3600,
    max_tool_calls_per_task: int = 100,
) -> Dict[str, Any]:
    """Run baseline evaluation on competitions."""
    competitions = competitions or get_lite_competitions()
    
    agent = MLEBenchAgent(
        api_base=api_base,
        workspace_dir=workspace_dir,
    )
    
    results = []
    summary = {
        "total_tasks": len(competitions),
        "successful": 0,
        "failed": 0,
        "scores": [],
        "start_time": datetime.now().isoformat(),
    }
    
    for comp_id in competitions:
        print(f"\n{'='*60}")
        print(f"Running: {comp_id}")
        print(f"{'='*60}")
        
        task_description = f"""Solve the {comp_id} Kaggle competition.

Steps:
1. Download and explore the dataset
2. Understand the competition format and evaluation metric
3. Build and train a model
4. Generate predictions in the correct submission format
5. Grade your submission to verify correctness

Work systematically and validate your approach."""
        
        budget = TaskBudget(
            max_time_seconds=max_time_per_task,
            max_tool_calls=max_tool_calls_per_task,
        )
        
        result = agent.run_task(comp_id, task_description, budget)
        
        task_result = {
            "competition_id": comp_id,
            "success": result.success,
            "score": result.score,
            "submission_path": result.submission_path,
            "error": result.error,
            "tool_calls": result.tool_calls,
            "time_elapsed": result.time_elapsed,
        }
        
        results.append(task_result)
        
        if result.success:
            summary["successful"] += 1
            if result.score is not None:
                summary["scores"].append(result.score)
            print(f"✓ Success: Score = {result.score}")
        else:
            summary["failed"] += 1
            print(f"✗ Failed: {result.error}")
        
        # Save individual logs
        log_dir = workspace_dir / comp_id
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "result.json").write_text(json.dumps(task_result, indent=2))
        if result.logs:
            (log_dir / "logs.json").write_text(json.dumps(result.logs, indent=2))
    
    summary["end_time"] = datetime.now().isoformat()
    summary["results"] = results
    
    # Calculate statistics
    if summary["scores"]:
        summary["mean_score"] = sum(summary["scores"]) / len(summary["scores"])
        summary["min_score"] = min(summary["scores"])
        summary["max_score"] = max(summary["scores"])
    
    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline evaluation on MLE-bench lite")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--competitions", type=str, nargs="+", help="Specific competitions to run")
    parser.add_argument("--max-time", type=int, default=3600, help="Max time per task (seconds)")
    parser.add_argument("--max-tool-calls", type=int, default=100, help="Max tool calls per task")
    parser.add_argument("--output", type=str, default="./baseline_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    print("🚀 Starting baseline evaluation on MLE-bench lite")
    print(f"API: {args.api_base}")
    print(f"Workspace: {args.workspace}")
    print(f"Max time per task: {args.max_time}s")
    print(f"Max tool calls per task: {args.max_tool_calls}")
    
    summary = run_baseline(
        api_base=args.api_base,
        workspace_dir=Path(args.workspace),
        competitions=args.competitions,
        max_time_per_task=args.max_time,
        max_tool_calls_per_task=args.max_tool_calls,
    )
    
    # Save summary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    
    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    if summary.get("scores"):
        print(f"Mean score: {summary.get('mean_score', 'N/A')}")
        print(f"Score range: {summary.get('min_score', 'N/A')} - {summary.get('max_score', 'N/A')}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


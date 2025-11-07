#!/usr/bin/env python3
"""
Evaluation utilities and report generation for MLE-bench results.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def load_results(results_path: Path) -> Dict[str, Any]:
    """Load evaluation results."""
    return json.loads(results_path.read_text())

def generate_report(results: Dict[str, Any], output_path: Path):
    """Generate markdown report from results."""
    lines = [
        "# MLE-bench Evaluation Report",
        "",
        f"**Start Time:** {results.get('start_time', 'N/A')}",
        f"**End Time:** {results.get('end_time', 'N/A')}",
        "",
        "## Summary",
        "",
        f"- Total Tasks: {results['total_tasks']}",
        f"- Successful: {results['successful']}",
        f"- Failed: {results['failed']}",
    ]
    
    if results.get("scores"):
        lines.extend([
            f"- Mean Score: {results.get('mean_score', 'N/A')}",
            f"- Score Range: {results.get('min_score', 'N/A')} - {results.get('max_score', 'N/A')}",
        ])
    
    lines.extend([
        "",
        "## Per-Task Results",
        "",
        "| Competition | Success | Score | Tool Calls | Time (s) | Error |",
        "|-------------|---------|-------|------------|----------|------|",
    ])
    
    for result in results.get("results", []):
        success = "✓" if result["success"] else "✗"
        score = result.get("score", "N/A")
        tool_calls = result.get("tool_calls", 0)
        time_elapsed = f"{result.get('time_elapsed', 0):.1f}"
        error = result.get("error", "")[:50] if result.get("error") else ""
        
        lines.append(
            f"| {result['competition_id']} | {success} | {score} | {tool_calls} | {time_elapsed} | {error} |"
        )
    
    output_path.write_text("\n".join(lines))

def compare_results(baseline_path: Path, improved_path: Path, output_path: Path):
    """Compare baseline vs improved results."""
    baseline = load_results(baseline_path)
    improved = load_results(improved_path)
    
    lines = [
        "# MLE-bench Comparison Report",
        "",
        "## Summary Comparison",
        "",
        "| Metric | Baseline | Improved | Change |",
        "|--------|----------|----------|--------|",
    ]
    
    baseline_success = baseline.get("successful", 0)
    improved_success = improved.get("successful", 0)
    success_change = improved_success - baseline_success
    
    lines.append(f"| Successful Tasks | {baseline_success} | {improved_success} | {success_change:+d} |")
    
    if baseline.get("scores") and improved.get("scores"):
        baseline_mean = baseline.get("mean_score", 0)
        improved_mean = improved.get("mean_score", 0)
        mean_change = improved_mean - baseline_mean
        
        lines.append(f"| Mean Score | {baseline_mean:.4f} | {improved_mean:.4f} | {mean_change:+.4f} |")
    
    lines.extend([
        "",
        "## Per-Task Comparison",
        "",
        "| Competition | Baseline Score | Improved Score | Change |",
        "|-------------|----------------|----------------|--------|",
    ])
    
    # Match competitions
    baseline_results = {r["competition_id"]: r for r in baseline.get("results", [])}
    improved_results = {r["competition_id"]: r for r in improved.get("results", [])}
    
    for comp_id in set(baseline_results.keys()) | set(improved_results.keys()):
        baseline_score = baseline_results.get(comp_id, {}).get("score", "N/A")
        improved_score = improved_results.get(comp_id, {}).get("score", "N/A")
        
        if baseline_score != "N/A" and improved_score != "N/A":
            change = improved_score - baseline_score
            change_str = f"{change:+.4f}"
        else:
            change_str = "N/A"
        
        lines.append(f"| {comp_id} | {baseline_score} | {improved_score} | {change_str} |")
    
    output_path.write_text("\n".join(lines))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("--results", type=str, required=True, help="Results JSON file")
    parser.add_argument("--output", type=str, help="Output markdown file")
    parser.add_argument("--compare-baseline", type=str, help="Baseline results to compare")
    parser.add_argument("--compare-improved", type=str, help="Improved results to compare")
    
    args = parser.parse_args()
    
    if args.compare_baseline and args.compare_improved:
        compare_results(
            Path(args.compare_baseline),
            Path(args.compare_improved),
            Path(args.output or "comparison_report.md"),
        )
    else:
        results = load_results(Path(args.results))
        generate_report(results, Path(args.output or "report.md"))


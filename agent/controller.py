#!/usr/bin/env python3
"""
MLE-bench agent controller with planning, retries, and budget limits.
Uses K2-Instruct with tool calling for autonomous ML engineering.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from openai import OpenAI

from agent.tools import ToolRegistry

@dataclass
class TaskBudget:
    """Budget constraints for a task."""
    max_time_seconds: int = 3600  # 1 hour default
    max_tool_calls: int = 100
    max_retries: int = 3

@dataclass
class TaskResult:
    """Result of running a task."""
    success: bool
    score: Optional[float] = None
    submission_path: Optional[str] = None
    error: Optional[str] = None
    tool_calls: int = 0
    time_elapsed: float = 0.0
    logs: List[Dict[str, Any]] = None

class MLEBenchAgent:
    """Agent controller for MLE-bench tasks."""
    
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "dummy",  # Not needed for local
        workspace_dir: Path = Path("./workspace"),
        kaggle_creds_path: Optional[str] = None,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.tools = ToolRegistry(self.workspace_dir, kaggle_creds_path)
        self.tool_schemas = self.tools.get_tool_schemas()
    
    def run_task(
        self,
        competition_id: str,
        task_description: str,
        budget: Optional[TaskBudget] = None,
    ) -> TaskResult:
        """Run a single MLE-bench task."""
        budget = budget or TaskBudget()
        start_time = time.time()
        tool_call_count = 0
        logs = []
        
        # Create task-specific workspace
        task_workspace = self.workspace_dir / competition_id
        task_workspace.mkdir(parents=True, exist_ok=True)
        task_tools = ToolRegistry(task_workspace)
        
        # Initial system message
        system_message = """You are an expert ML engineer solving Kaggle competitions.
Your goal is to:
1. Understand the competition and dataset
2. Build a model that performs well
3. Generate a submission CSV in the correct format
4. Grade your submission to verify correctness

You have access to tools for:
- Python execution (for data processing, modeling, etc.)
- Filesystem operations (read/write files)
- Kaggle dataset downloads
- Submission grading

Work systematically: explore data, build models iteratively, validate, and submit.
"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": task_description},
        ]
        
        retries = 0
        while retries < budget.max_retries:
            try:
                # Check time budget
                elapsed = time.time() - start_time
                if elapsed > budget.max_time_seconds:
                    return TaskResult(
                        success=False,
                        error=f"Time budget exceeded ({elapsed:.1f}s > {budget.max_time_seconds}s)",
                        tool_calls=tool_call_count,
                        time_elapsed=elapsed,
                        logs=logs,
                    )
                
                # Check tool call budget
                if tool_call_count >= budget.max_tool_calls:
                    return TaskResult(
                        success=False,
                        error=f"Tool call budget exceeded ({tool_call_count} >= {budget.max_tool_calls})",
                        tool_calls=tool_call_count,
                        time_elapsed=elapsed,
                        logs=logs,
                    )
                
                # Call model with tools
                response = self.client.chat.completions.create(
                    model="kimi-k2-instruct",
                    messages=messages,
                    temperature=0.6,
                    tools=task_tools.get_tool_schemas(),
                    tool_choice="auto",
                    max_tokens=4096,
                )
                
                choice = response.choices[0]
                message = choice.message
                
                # Log response
                logs.append({
                    "type": "model_response",
                    "content": message.content,
                    "tool_calls": len(message.tool_calls) if message.tool_calls else 0,
                })
                
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (message.tool_calls or [])
                    ],
                })
                
                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_call_count += 1
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        # Execute tool
                        tool_result = task_tools.call_tool(tool_name, tool_args)
                        
                        logs.append({
                            "type": "tool_call",
                            "tool": tool_name,
                            "args": tool_args,
                            "result": tool_result,
                        })
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(tool_result),
                        })
                        
                        # Check if submission was graded successfully
                        if tool_name == "grade_submission" and tool_result.get("success"):
                            score = tool_result.get("score")
                            submission_path = tool_args.get("submission_path")
                            return TaskResult(
                                success=True,
                                score=score,
                                submission_path=submission_path,
                                tool_calls=tool_call_count,
                                time_elapsed=time.time() - start_time,
                                logs=logs,
                            )
                
                # Check if finished (no tool calls and finish_reason is stop)
                if choice.finish_reason == "stop" and not message.tool_calls:
                    # Check if submission file exists
                    submission_path = task_workspace / "submission.csv"
                    if submission_path.exists():
                        # Try to grade it
                        grade_result = task_tools.grading.grade_sample(
                            str(submission_path),
                            competition_id,
                        )
                        if grade_result.get("success"):
                            return TaskResult(
                                success=True,
                                score=grade_result.get("score"),
                                submission_path=str(submission_path),
                                tool_calls=tool_call_count,
                                time_elapsed=time.time() - start_time,
                                logs=logs,
                            )
                    
                    return TaskResult(
                        success=False,
                        error="Task finished without successful submission",
                        tool_calls=tool_call_count,
                        time_elapsed=time.time() - start_time,
                        logs=logs,
                    )
                
            except Exception as e:
                logs.append({
                    "type": "error",
                    "error": str(e),
                })
                retries += 1
                if retries >= budget.max_retries:
                    return TaskResult(
                        success=False,
                        error=f"Failed after {retries} retries: {str(e)}",
                        tool_calls=tool_call_count,
                        time_elapsed=time.time() - start_time,
                        logs=logs,
                    )
                time.sleep(1)  # Brief pause before retry
        
        return TaskResult(
            success=False,
            error="Max retries exceeded",
            tool_calls=tool_call_count,
            time_elapsed=time.time() - start_time,
            logs=logs,
        )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run MLE-bench agent on a task")
    parser.add_argument("--competition-id", type=str, required=True)
    parser.add_argument("--task-description", type=str, help="Task description")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--max-time", type=int, default=3600)
    parser.add_argument("--max-tool-calls", type=int, default=100)
    
    args = parser.parse_args()
    
    agent = MLEBenchAgent(
        api_base=args.api_base,
        workspace_dir=Path(args.workspace),
    )
    
    task_description = args.task_description or f"Solve the {args.competition_id} competition."
    
    budget = TaskBudget(
        max_time_seconds=args.max_time,
        max_tool_calls=args.max_tool_calls,
    )
    
    result = agent.run_task(args.competition_id, task_description, budget)
    
    print(f"\n{'='*60}")
    print(f"Task: {args.competition_id}")
    print(f"Success: {result.success}")
    if result.score is not None:
        print(f"Score: {result.score}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Time elapsed: {result.time_elapsed:.1f}s")
    print(f"{'='*60}")
    
    # Save logs
    log_path = Path(args.workspace) / args.competition_id / "logs.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result.logs, indent=2))

if __name__ == "__main__":
    main()


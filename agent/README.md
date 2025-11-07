# MLE-bench Agent

Autonomous agent for solving MLE-bench tasks using K2-Instruct with tool calling.

## Tools

- `python_exec`: Sandboxed Python execution
- `fs_read/fs_write/fs_list`: Filesystem operations (workspace-scoped)
- `kaggle_download`: Download competition datasets
- `grade_submission`: Grade submissions using MLE-bench grader

## Usage

```bash
# Run agent on a competition
python controller.py \
  --competition-id spaceship-titanic \
  --api-base http://localhost:8000/v1 \
  --workspace ./workspace \
  --max-time 3600
```

## Architecture

The agent uses a plan-act-observe loop:
1. Receives task description
2. Plans approach using K2-Instruct
3. Executes tools (Python, filesystem, Kaggle, grading)
4. Observes results and iterates
5. Generates submission CSV
6. Grades submission

All operations are sandboxed to the task workspace and respect budget limits.


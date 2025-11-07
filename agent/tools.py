#!/usr/bin/env python3
"""
Tools for MLE-bench agent: Python execution, filesystem, Kaggle, grading.
All tools are sandboxed and rules-compliant.
"""

import os
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

class PythonExecTool:
    """Sandboxed Python execution tool."""
    
    def __init__(self, workspace_dir: Path, timeout: int = 300):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
    
    def execute(self, code: str, timeout_s: Optional[int] = None) -> Dict[str, Any]:
        """Execute Python code in sandboxed workspace."""
        timeout = timeout_s or self.timeout
        
        # Write code to temp file
        script_path = self.workspace_dir / "exec_script.py"
        script_path.write_text(code)
        
        # Execute in subprocess with timeout
        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(self.workspace_dir)},
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}\n{traceback.format_exc()}",
                "returncode": -1,
            }

class FilesystemTool:
    """Filesystem read/write tool (limited to workspace)."""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    def read(self, path: str) -> Dict[str, Any]:
        """Read file content."""
        file_path = (self.workspace_dir / path).resolve()
        
        # Security: ensure path is within workspace
        if not str(file_path).startswith(str(self.workspace_dir)):
            return {
                "success": False,
                "error": "Path outside workspace",
            }
        
        try:
            if file_path.exists():
                content = file_path.read_text()
                return {
                    "success": True,
                    "content": content,
                    "path": str(file_path),
                }
            else:
                return {
                    "success": False,
                    "error": "File not found",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def write(self, path: str, content: str) -> Dict[str, Any]:
        """Write file content."""
        file_path = (self.workspace_dir / path).resolve()
        
        # Security: ensure path is within workspace
        if not str(file_path).startswith(str(self.workspace_dir)):
            return {
                "success": False,
                "error": "Path outside workspace",
            }
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return {
                "success": True,
                "path": str(file_path),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        dir_path = (self.workspace_dir / path).resolve()
        
        if not str(dir_path).startswith(str(self.workspace_dir)):
            return {
                "success": False,
                "error": "Path outside workspace",
            }
        
        try:
            if dir_path.exists() and dir_path.is_dir():
                items = [item.name for item in dir_path.iterdir()]
                return {
                    "success": True,
                    "items": items,
                    "path": str(dir_path),
                }
            else:
                return {
                    "success": False,
                    "error": "Directory not found",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

class KaggleTool:
    """Kaggle dataset download tool."""
    
    def __init__(self, kaggle_creds_path: Optional[str] = None):
        self.kaggle_creds_path = Path(
            kaggle_creds_path or os.getenv("KAGGLE_CREDS", "~/.kaggle/kaggle.json")
        ).expanduser()
    
    def download(self, competition_id: str, dest_dir: Path) -> Dict[str, Any]:
        """Download competition dataset."""
        if not self.kaggle_creds_path.exists():
            return {
                "success": False,
                "error": "Kaggle credentials not found",
            }
        
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", competition_id, "-p", str(dest_dir)],
                capture_output=True,
                text=True,
                env={**os.environ, "KAGGLE_CONFIG_DIR": str(self.kaggle_creds_path.parent)},
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "dest_dir": str(dest_dir),
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

class GradingTool:
    """MLE-bench grading tool wrapper."""
    
    def __init__(self, mlebench_cache: Optional[str] = None):
        self.mlebench_cache = mlebench_cache or os.getenv("MLEBENCH_CACHE", "~/.cache/mlebench")
    
    def grade_sample(self, submission_path: str, competition_id: str) -> Dict[str, Any]:
        """Grade a submission using mlebench grade-sample."""
        try:
            result = subprocess.run(
                ["mlebench", "grade-sample", submission_path, competition_id],
                capture_output=True,
                text=True,
                env={**os.environ, "MLEBENCH_CACHE": self.mlebench_cache},
            )
            
            if result.returncode == 0:
                # Parse score from output (format may vary)
                output = result.stdout
                # Try to extract score
                score = None
                for line in output.split("\n"):
                    if "score" in line.lower() or "metric" in line.lower():
                        # Simple extraction - may need refinement
                        try:
                            import re
                            numbers = re.findall(r"[-+]?\d*\.?\d+", line)
                            if numbers:
                                score = float(numbers[0])
                                break
                        except:
                            pass
                
                return {
                    "success": True,
                    "score": score,
                    "output": output,
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "output": result.stdout,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

class ToolRegistry:
    """Registry of all available tools for agent."""
    
    def __init__(self, workspace_dir: Path, kaggle_creds_path: Optional[str] = None):
        self.workspace_dir = workspace_dir
        self.python_exec = PythonExecTool(workspace_dir)
        self.filesystem = FilesystemTool(workspace_dir)
        self.kaggle = KaggleTool(kaggle_creds_path)
        self.grading = GradingTool()
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "python_exec",
                    "description": "Execute Python code in sandboxed workspace. Returns stdout, stderr, and return code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            },
                            "timeout_s": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 300)",
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs_read",
                    "description": "Read file content from workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to file",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs_write",
                    "description": "Write file content to workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to file",
                            },
                            "content": {
                                "type": "string",
                                "description": "File content",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs_list",
                    "description": "List directory contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path (default: current directory)",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "kaggle_download",
                    "description": "Download Kaggle competition dataset.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "competition_id": {
                                "type": "string",
                                "description": "Kaggle competition ID",
                            },
                        },
                        "required": ["competition_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grade_submission",
                    "description": "Grade submission using MLE-bench grader.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "submission_path": {
                                "type": "string",
                                "description": "Path to submission CSV",
                            },
                            "competition_id": {
                                "type": "string",
                                "description": "Competition ID",
                            },
                        },
                        "required": ["submission_path", "competition_id"],
                    },
                },
            },
        ]
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with arguments."""
        if name == "python_exec":
            return self.python_exec.execute(
                arguments["code"],
                timeout_s=arguments.get("timeout_s"),
            )
        elif name == "fs_read":
            return self.filesystem.read(arguments["path"])
        elif name == "fs_write":
            return self.filesystem.write(arguments["path"], arguments["content"])
        elif name == "fs_list":
            return self.filesystem.list_dir(arguments.get("path", "."))
        elif name == "kaggle_download":
            return self.kaggle.download(
                arguments["competition_id"],
                self.workspace_dir / "data",
            )
        elif name == "grade_submission":
            return self.grading.grade_sample(
                arguments["submission_path"],
                arguments["competition_id"],
            )
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {name}",
            }


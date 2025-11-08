#!/usr/bin/env python3
"""
Serve Qwen2.5-Coder-32B-Instruct (or any HF model) with vLLM for the
MLE-bench agent. Optimized defaults for tight budgets; configurable via
environment variables.
"""

import os
import argparse
from contextlib import asynccontextmanager
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Disable hf_transfer if not available (fallback to regular download)
if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        __import__("hf_transfer")
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Global engine
engine: Optional[AsyncLLMEngine] = None


class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be str or list of dicts


class ChatRequest(BaseModel):
    model: str = "qwen2.5-coder-32b-instruct"
    messages: List[ChatMessage]
    temperature: float = 0.6
    max_tokens: int = 4096
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: str = "auto"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def initialize_engine(
    model_path: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,  # conservative default for 32B dense
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
):
    """Initialize vLLM engine."""
    global engine

    # Optional knobs via env
    quantization = os.getenv("QUANTIZATION")
    kv_cache_dtype = os.getenv("KV_CACHE_DTYPE")

    engine_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enable_lora=False,  # Will enable later for PEFT
        block_size=16,
        gpu_memory_utilization=0.85,
    )
    if quantization:
        engine_kwargs["quantization"] = quantization
    if kv_cache_dtype:
        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype

    engine_args = AsyncEngineArgs(**engine_kwargs)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"✓ Engine initialized: {model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-Coder-32B-Instruct")
    max_len = int(os.getenv("MAX_MODEL_LEN", "16384"))
    tensor_parallel = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

    print(f"Initializing model from {model_path}")
    initialize_engine(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel,
        max_model_len=max_len,
    )

    yield

    # Shutdown (cleanup if needed)
    pass


app = FastAPI(title="Qwen2.5-Coder MLE-bench API", lifespan=lifespan)


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Convert messages to prompt format
    # K2-Instruct uses standard chat format
    prompt_parts = []

    def _to_text(item: Any) -> str:
        if isinstance(item, dict):
            return item.get("text", "")
        return str(item)

    for msg in request.messages:
        role = msg.role
        content = msg.content
        if isinstance(content, list):
            # Handle multimodal (text only for now)
            text_content = " ".join([_to_text(item) for item in content])
            prompt_parts.append(f"{role}: {text_content}")
        else:
            prompt_parts.append(f"{role}: {content}")

    prompt = "\n".join(prompt_parts) + "\nassistant:"

    # Prepare sampling params
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    # Generate
    request_id = f"req_{hash(prompt) % 1000000}"
    generator = engine.generate(prompt, sampling_params, request_id)

    async for request_output in generator:
        if request_output.finished:
            generated_text = request_output.outputs[0].text

            # Compute token counts to avoid long expressions on one line
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
            total_tokens = prompt_tokens + completion_tokens

            return ChatResponse(
                id=f"chatcmpl-{request_id}",
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )

    raise HTTPException(status_code=500, detail="Generation failed")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_PATH", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)

    uvicorn.run(app, host=args.host, port=args.port)

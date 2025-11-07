#!/usr/bin/env python3
"""
Serve Kimi-K2-Instruct with vLLM for MLE-bench agent.
Optimized for tight budget: single GPU, block-FP8 weights, capped context.
"""

import os
import argparse
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json

app = FastAPI(title="K2-Instruct MLE-bench API")

# Global engine
engine: Optional[AsyncLLMEngine] = None

class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be str or list of dicts

class ChatRequest(BaseModel):
    model: str = "kimi-k2-instruct"
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
    max_model_len: int = 65536,  # Cap at 64K for budget
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
):
    """Initialize vLLM engine with K2-Instruct."""
    global engine
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enable_lora=False,  # Will enable later for PEFT
        block_size=16,
        gpu_memory_utilization=0.9,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"✓ Engine initialized: {model_path}")

@app.on_event("startup")
async def startup():
    """Initialize engine on startup."""
    model_path = os.getenv("MODEL_PATH", "MoonshotAI/Kimi-K2-Instruct")
    max_len = int(os.getenv("MAX_MODEL_LEN", "65536"))
    tensor_parallel = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    
    print(f"Initializing K2-Instruct from {model_path}")
    initialize_engine(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel,
        max_model_len=max_len,
    )

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Convert messages to prompt format
    # K2-Instruct uses standard chat format
    prompt_parts = []
    for msg in request.messages:
        role = msg.role
        content = msg.content
        if isinstance(content, list):
            # Handle multimodal (text only for now)
            text_content = " ".join([
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ])
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
            
            return ChatResponse(
                id=f"chatcmpl-{request_id}",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }],
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                }
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
    parser.add_argument("--model-path", type=str, default=os.getenv("MODEL_PATH", "MoonshotAI/Kimi-K2-Instruct"))
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    
    args = parser.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)
    
    uvicorn.run(app, host=args.host, port=args.port)


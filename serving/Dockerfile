FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Install vLLM and dependencies
RUN pip3 install vllm==0.6.0 \
    transformers>=4.40.0 \
    torch>=2.1.0 \
    accelerate \
    huggingface-hub

# Install MLE-bench dependencies
RUN pip3 install openai \
    anthropic \
    requests \
    numpy \
    pandas

WORKDIR /app

# Copy serving scripts
COPY serving/*.py /app/
COPY serving/*.sh /app/

RUN chmod +x /app/*.sh

EXPOSE 8000

CMD ["python3", "serve.py"]


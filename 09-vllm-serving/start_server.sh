#!/usr/bin/env bash

vllm serve \
    Qwen/Qwen2.5-7B-Instruct-AWQ \
    --served-model-name qwen2.5-7b-awq \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85

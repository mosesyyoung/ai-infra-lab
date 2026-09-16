#!/usr/bin/env bash

curl -s \
  http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-awq",
    "messages": [
      {
        "role": "system",
        "content": "You are an AI infrastructure assistant."
      },
      {
        "role": "user",
        "content": "从 Infra 角度解释什么是 Model Serving。"
      }
    ],
    "temperature": 0,
    "max_tokens": 128
  }' | jq

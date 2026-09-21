# LLM Serving Benchmark Results

Raw benchmark outputs for AI Infra Lab 10.

## Environment

- Ubuntu 24.04 Server
- Intel Core i5-10400F
- 32GB RAM
- NVIDIA GeForce RTX 3060 12GB
- Qwen2.5-7B-Instruct

## Runtimes

Ollama:

```text
Qwen2.5-7B-Instruct
GGUF Q4_K_M
```

vLLM:
```text
Qwen2.5-7B-Instruct-AWQ
AWQ INT4
```

## Benchmark

All tests use vllm bench serve with:

```text
Input Length:  512 tokens
Output Length: 128 tokens
Requests:      100
Concurrency:   1 / 4 / 8 / 16
```

Files:

```text
ollama-c1.log
ollama-c4.log
ollama-c8.log
ollama-c16.log

vllm-c1.log
vllm-c4.log
vllm-c8.log
vllm-c16.log
```

## Note

The vLLM concurrency tests were collected from separate successful
benchmark runs because the test host experienced intermittent system
freezes/reboots during testing.

The results are intended for studying serving metrics and scaling
behavior rather than as a strict runtime or hardware benchmark.

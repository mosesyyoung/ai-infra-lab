import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def mib(value):
    return value / 1024**2


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.to("cuda")
model.eval()


short_prompt = "什么是 GPU？"


long_prompt = """
我正在学习 AI Infrastructure，希望从 GPU、CUDA、PyTorch、
模型推理、推理引擎、Kubernetes 和 Model Serving 的角度，
逐步理解整个 AI Infra 技术栈。

此前我已经学习了 CPU 与 GPU 的区别、CUDA 软件栈、
PyTorch Tensor、FP32、FP16、BF16，以及模型参数、
Transformer、Tokenizer、Token 和 Context。

现在请结合这些背景，从 Infra 的角度解释什么是 GPU，
并说明为什么 GPU 特别适合大语言模型推理。
"""


def run_prompt(name, user_prompt):

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    input_tokens = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()

    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
        )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()

    peak_allocated = torch.cuda.max_memory_allocated()

    total_tokens = outputs.shape[1]

    output_tokens = (
        total_tokens - input_tokens
    )

    generated_ids = outputs[0][input_tokens:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Input Tokens :",
        input_tokens
    )

    print(
        "Output Tokens:",
        output_tokens
    )

    print(
        "Generate Time:",
        f"{elapsed:.3f} s"
    )

    print()
    print("GPU Memory")

    print(
        "Before Allocated:",
        f"{mib(before_allocated):.1f} MiB"
    )

    print(
        "Before Reserved :",
        f"{mib(before_reserved):.1f} MiB"
    )

    print(
        "After Allocated :",
        f"{mib(after_allocated):.1f} MiB"
    )

    print(
        "After Reserved  :",
        f"{mib(after_reserved):.1f} MiB"
    )

    print(
        "Peak Allocated  :",
        f"{mib(peak_allocated):.1f} MiB"
    )

    print()
    print("Answer:")
    print(answer)


run_prompt(
    "Short Prompt",
    short_prompt
)

run_prompt(
    "Long Prompt",
    long_prompt
)

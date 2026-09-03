import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def mib(value):
    return value / 1024**2


def show_gpu_memory(title):
    print()
    print(f"=== {title} ===")

    print(
        "Allocated:",
        f"{mib(torch.cuda.memory_allocated()):.1f} MiB"
    )

    print(
        "Reserved :",
        f"{mib(torch.cuda.memory_reserved()):.1f} MiB"
    )


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

show_gpu_memory("Before Model Loading")


print()
print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


print()
print("Loading Model to CPU RAM...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.eval()


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


parameter_bytes = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)


print()
print(
    "Parameter Count:",
    f"{parameter_count:,}"
)

print(
    "Parameter Memory:",
    f"{parameter_bytes / 1024**3:.3f} GiB"
)

print(
    "Model dtype:",
    next(model.parameters()).dtype
)

print(
    "Model device:",
    next(model.parameters()).device
)


show_gpu_memory("Model on CPU")


print()
print("Moving Model to GPU...")

model.to("cuda")

torch.cuda.synchronize()


print(
    "Model device:",
    next(model.parameters()).device
)


show_gpu_memory("Model on GPU")


input(
    "\nPress Enter to exit..."
)

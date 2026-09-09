import sys
import time

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_IDS = {
    "bf16":
        "Qwen/Qwen2.5-7B-Instruct",

    "int8":
        "Qwen/Qwen2.5-7B-Instruct",

    "int4":
        "Qwen/Qwen2.5-7B-Instruct",

    "awq":
        "Qwen/Qwen2.5-7B-Instruct-AWQ",

    "gptq":
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
}


if len(sys.argv) != 2:
    print(
        "Usage: python quant_vram_test.py "
        "bf16|int8|int4|awq|gptq"
    )
    sys.exit(1)


MODE = sys.argv[1]


if MODE not in MODEL_IDS:
    raise ValueError(
        f"Unknown mode: {MODE}"
    )


MODEL_ID = MODEL_IDS[MODE]


print("=" * 60)
print("Mode :", MODE)
print("Model:", MODEL_ID)
print("GPU  :", torch.cuda.get_device_name(0))
print("=" * 60)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)


load_args = {
    # 故意要求整个模型进入 GPU 0
    # 避免自动 CPU Offload 干扰实验
    "device_map": {"": 0},

    "low_cpu_mem_usage": True,

    # 只读本地 Cache
    "local_files_only": True,
}


if MODE == "bf16":

    load_args["dtype"] = torch.bfloat16


elif MODE == "int8":

    load_args["quantization_config"] = (
        BitsAndBytesConfig(
            load_in_8bit=True
        )
    )


elif MODE == "int4":

    load_args["quantization_config"] = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    )


elif MODE in ("awq", "gptq"):

    # Model 自身的 config.json
    # 已经包含 quantization_config
    load_args["dtype"] = "auto"


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


print()
print("Loading model from local cache...")


start = time.perf_counter()


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    **load_args
)


model.eval()

torch.cuda.synchronize()


load_time = (
    time.perf_counter()
    - start
)


print(
    "Load Time:",
    f"{load_time:.2f} s"
)


if hasattr(
    model,
    "get_memory_footprint"
):

    footprint = (
        model.get_memory_footprint()
        / 1024**3
    )

    print(
        "Model Footprint:",
        f"{footprint:.3f} GiB"
    )


print(
    "Allocated:",
    f"{torch.cuda.memory_allocated() / 1024**3:.3f} GiB"
)

print(
    "Reserved :",
    f"{torch.cuda.memory_reserved() / 1024**3:.3f} GiB"
)


messages = [
    {
        "role": "user",
        "content": (
            "用三点解释模型量化为什么"
            "能够降低 GPU 显存占用。"
        )
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


device = next(
    model.parameters()
).device


inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()


start = time.perf_counter()


with torch.inference_mode():

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )


torch.cuda.synchronize()


generate_time = (
    time.perf_counter()
    - start
)


print(
    "Generate Time:",
    f"{generate_time:.3f} s"
)


print(
    "Peak VRAM:",
    f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GiB"
)


input_tokens = (
    inputs["input_ids"].shape[1]
)

output_tokens = (
    outputs.shape[1]
    - input_tokens
)


print(
    "Input Tokens :",
    input_tokens
)

print(
    "Output Tokens:",
    output_tokens
)


input(
    "\nPress Enter to exit..."
)

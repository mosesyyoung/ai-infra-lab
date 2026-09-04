import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.to("cuda")
model.eval()

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI infrastructure assistant."
    },
    {
        "role": "user",
        "content": "从 Infra 角度解释一下什么是 LLM。"
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
    k: v.to("cuda")
    for k, v in inputs.items()
}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

input_tokens = inputs["input_ids"].shape[1]
total_tokens = outputs.shape[1]
output_tokens = total_tokens - input_tokens

generated_ids = outputs[0][input_tokens:]

answer = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

print("Input Tokens :", input_tokens)
print("Output Tokens:", output_tokens)

print("\nAnswer:")
print(answer)

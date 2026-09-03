from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

text = "从 Infra 角度看，大语言模型到底是什么？"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("Text:")
print(text)

print()

print("Input IDs:")
print(inputs["input_ids"])

print()

print(
    "Token Count:",
    inputs["input_ids"].shape[-1]
)

print()

print("Tokens:")

print(
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )
)

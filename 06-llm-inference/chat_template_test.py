from transformers import AutoTokenizer
 
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
 
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
 
print(prompt)

from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)


response = client.chat.completions.create(
    model="qwen2.5-7b-awq",
    messages=[
        {
            "role": "system",
            "content": "You are an AI infrastructure assistant.",
        },
        {
            "role": "user",
            "content": "从 Infra 角度解释什么是 Model Serving。",
        },
    ],
    temperature=0,
    max_tokens=128,
)


print(response.choices[0].message.content)

print()
print("Usage:")
print(response.usage)

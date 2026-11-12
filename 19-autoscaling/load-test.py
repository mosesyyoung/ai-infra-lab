import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8080/openai/v1/chat/completions"

DURATION = 180
WORKERS = 4

payload = json.dumps({
    "model": "qwen",
    "messages": [{
        "role": "user",
        "content": "请简要解释 Kubernetes 自动扩容的作用。"
    }],
    "max_tokens": 64,
    "temperature": 0
}).encode("utf-8")

def worker(end):
    success = 0
    errors = 0

    while time.monotonic() < end:
        req = urllib.request.Request(
            URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                r.read()
                success += 1
        except Exception:
            errors += 1

    return success, errors

end = time.monotonic() + DURATION

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    results = list(pool.map(
        worker, [end] * WORKERS
    ))

print("Success:", sum(x[0] for x in results))
print("Errors:", sum(x[1] for x in results))

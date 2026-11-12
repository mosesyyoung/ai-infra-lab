from dataclasses import dataclass


@dataclass
class Endpoint:
    name: str
    waiting: int
    kv_usage: float
    cached_tokens: int


def estimate_cost(ep, prompt_tokens):
    uncached = max(
        0, prompt_tokens - ep.cached_tokens
    )

    # Toy cost model, unit: arbitrary score
    prefill_cost = uncached * 0.7
    queue_cost = ep.waiting * 200
    kv_pressure = ep.kv_usage * 500

    return prefill_cost + queue_cost + kv_pressure


def schedule(endpoints, prompt_tokens):
    results = []

    for ep in endpoints:
        cost = estimate_cost(ep, prompt_tokens)
        results.append((ep.name, cost))

    results.sort(key=lambda x: x[1])

    for name, cost in results:
        print(f"{name}: {cost:.0f}")

    print("Selected:", results[0][0])
    print()


endpoints = [
    Endpoint("A", 1, 0.40, 3500),
    Endpoint("B", 0, 0.10, 0),
    Endpoint("C", 3, 0.25, 1500),
]

print("Scenario 1: Cache-aware")
schedule(endpoints, 4000)

endpoints[0].waiting = 15

print("Scenario 2: A is overloaded")
schedule(endpoints, 4000)

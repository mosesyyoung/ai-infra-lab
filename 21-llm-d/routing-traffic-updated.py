#!/usr/bin/env python3
"""llm-d Simulator routing experiment: two traffic modes + per-Pod metrics deltas.

Requirements: Python 3.9+, kubectl configured to access the cluster, an existing
Gateway port-forward at localhost:8080. No third-party Python packages.

Run: python3 routing-traffic-updated.py
"""

import argparse
import collections
import concurrent.futures
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


METRIC_NAMES = (
    "vllm:request_success_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)
METRIC_RE = re.compile(r"^(vllm:[a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")


def run_kubectl(*args):
    result = subprocess.run(
        ["kubectl", *args], capture_output=True, text=True,
        timeout=15, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "kubectl 执行失败")
    return result.stdout


def find_pods(namespace, selector):
    raw = run_kubectl("get", "pods", "-n", namespace, "-l", selector, "-o", "json")
    objs = json.loads(raw)["items"]
    pods = []
    for item in objs:
        name = item["metadata"]["name"]
        ready = (
            item.get("status", {}).get("phase") == "Running"
            and all(x.get("ready", False) for x in item.get("status", {}).get("containerStatuses", []))
            and bool(item.get("status", {}).get("containerStatuses", []))
        )
        if ready:
            pods.append(name)
        else:
            print(f"[警告] Pod 尚未 Ready: {name}", file=sys.stderr)
    return sorted(pods)


def parse_metrics(content):
    found = collections.defaultdict(float)
    seen = set()
    for line in content.splitlines():
        m = METRIC_RE.match(line.strip())
        if not m:
            continue
        name, value = m.groups()
        if name in METRIC_NAMES:
            found[name] += float(value)
            seen.add(name)
    return {name: found[name] if name in seen else None for name in METRIC_NAMES}


def snapshot(pods, namespace, port):
    readings = {}
    for pod in pods:
        url = f"/api/v1/namespaces/{namespace}/pods/{pod}:{port}/proxy/metrics"
        try:
            readings[pod] = parse_metrics(run_kubectl("get", "--raw", url))
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            print(f"[提示] 无法读取 {pod} 的 /metrics：{exc}", file=sys.stderr)
            readings[pod] = {name: None for name in METRIC_NAMES}
    return readings


def delta(before, after, pod, metric):
    old = before.get(pod, {}).get(metric)
    new = after.get(pod, {}).get(metric)
    if old is None or new is None or new < old:
        return None
    return new - old


def pct(latencies, p):
    if not latencies:
        return None
    return sorted(latencies)[max(0, math.ceil(len(latencies) * p) - 1)] * 1000


def request(args, mode, index):
    shared = (
        "你是一位 Linux、Kubernetes 和 AI Infra 专家。"
        "请从系统架构、资源管理、性能优化角度回答问题。"
    ) * 60
    if mode == "shared":
        prompt = shared + f"\n问题编号：{index}"
    else:
        # 与旧 routing-traffic.py 保持相同的 unique prompt 构造。
        prompt = f"这是独立上下文 {index}。" + f"技术文档片段 {index}。" * 200

    payload = json.dumps({
        "model": args.model,
        "prompt": prompt,
        "max_tokens": 32,
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        args.url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    pod, error = None, None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            response.read()
            status = response.status
            pod = response.headers.get("x-inference-pod")
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = str(exc)
        exc.close()
    except Exception as exc:
        status = -1
        error = str(exc)
    return status, time.perf_counter() - start, pod, error


def measure(args, mode, pods):
    print(f"\n[{mode}] 采集三个 Pod 的实验前计数...")
    before = snapshot(pods, args.namespace, args.backend_port)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: request(args, mode, i), range(args.count)))
    after = snapshot(pods, args.namespace, args.backend_port)

    statuses = collections.Counter(result[0] for result in results)
    successful = [r for r in results if r[0] == 200]
    latencies = [r[1] for r in successful]

    pod_counts = {}
    for pod in pods:
        count = delta(before, after, pod, "vllm:request_success_total")
        pod_counts[pod] = int(round(count)) if count is not None else None

    # Fallback for setups where the apiserver Pod proxy is unavailable.
    header_counts = collections.Counter(r[2] for r in successful if r[2])
    if any(v is None for v in pod_counts.values()):
        if sum(header_counts.values()) == len(successful) and successful:
            pod_counts = {pod: header_counts[pod] for pod in pods}
            source = "HTTP x-inference-pod Header"
        else:
            source = "无可靠数据（Pod metrics 与 Header 均不完整）"
    else:
        source = "Pod /metrics 的 request_success_total 前后差值"

    hits = [delta(before, after, p, "vllm:prefix_cache_hits_total") for p in pods]
    queries = [delta(before, after, p, "vllm:prefix_cache_queries_total") for p in pods]
    if all(v is not None for v in hits + queries) and sum(queries) > 0:
        hit_rate = f"{sum(hits) / sum(queries) * 100:.2f}%（模拟 Token 命中）"
    else:
        hit_rate = "未测得（未启用模拟或无指标）"

    print(f"[{mode}] HTTP 状态：{dict(sorted(statuses.items()))}")
    print(f"[{mode}] 成功 {len(successful)}/{args.count}，P50={pct(latencies, .50)!s} ms，P95={pct(latencies, .95)!s} ms")
    print(f"[{mode}] Pod 计数来源：{source}")
    print(f"[{mode}] Pod 分布：{pod_counts}")
    total = sum(v for v in pod_counts.values() if v is not None)
    if all(v is not None for v in pod_counts.values()) and total != len(successful):
        print(f"[警告] Pod 计数增量合计 {total} != HTTP 200 数量 {len(successful)}；"
              "可能有其他并发流量、Pod 重启、指标更新差异，请勿直接当作精确分布。", file=sys.stderr)
    if len(successful) != args.count:
        err_examples = [r[3] for r in results if r[3]]
        print(f"[警告] 测试未全部成功。错误示例：{err_examples[:2]}", file=sys.stderr)
    return {
        "success": len(successful), "count": args.count,
        "p50": pct(latencies, .50), "p95": pct(latencies, .95),
        "pod_counts": pod_counts, "hit_rate": hit_rate,
        "source": source, "status": dict(statuses),
        "consistent": all(v is not None for v in pod_counts.values()) and total == len(successful),
    }


def format_latency(value):
    return f"{value:.2f} ms" if value is not None else "未测得"


def make_markdown(pods, shared, unique):
    labels = {pod: chr(ord("A") + i) for i, pod in enumerate(pods)}
    def count(row, pod):
        value = row["pod_counts"][pod]
        return str(value) if value is not None and row["consistent"] else "待核实"
    rows = [
        ("成功请求数", f"{shared['success']}/{shared['count']}", f"{unique['success']}/{unique['count']}"),
        ("请求完成时间 P50", format_latency(shared["p50"]), format_latency(unique["p50"])),
        ("请求完成时间 P95", format_latency(shared["p95"]), format_latency(unique["p95"])),
    ]
    rows += [(f"Pod {labels[pod]} 请求数增量", count(shared, pod), count(unique, pod)) for pod in pods]
    rows += [("Prefix Cache 实际命中率", shared["hit_rate"], unique["hit_rate"])]
    lines = [
        "| 观察项目 | Shared Prefix | Unique Prefix |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {left} | {right} |" for name, left, right in rows)
    lines += ["", "Pod 名称对应："]
    lines.extend(f"- Pod {labels[pod]}：`{pod}`" for pod in pods)
    lines += [
        "", "注：P50/P95 为客户端观察到的 Simulator 端到端请求完成时间，不代表真实 GPU 推理性能。",
        "Pod 请求数来自每轮测试前后的成功计数器差值；若测试期间存在其他请求，增量可能包含额外流量。",
        "Prefix Cache 指标不存在时写‘未测得’，不解释为 0% 命中。",
        "Shared/Unique 两组 Prompt 长度不严格一致，不能将两组延迟差直接解释为缓存优化效果。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="llm-d Qwen Simulator 调度测试与统计")
    parser.add_argument("--url", default="http://localhost:8080/v1/completions")
    parser.add_argument("--namespace", default="ai-gateway")
    parser.add_argument("--selector", default="app=qwen-sim")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default="routing-traffic-results.md")
    args = parser.parse_args()

    if args.count < 1 or args.concurrency < 1:
        parser.error("--count 和 --concurrency 必须大于 0")
    if not shutil_which_kubectl():
        parser.error("找不到 kubectl，请先安装并配置 kubeconfig")

    host = urllib.parse.urlparse(args.url)
    if host.hostname in ("localhost", "127.0.0.1"):
        try:
            with socket.create_connection((host.hostname, host.port or 80), timeout=2):
                pass
        except OSError:
            parser.error("连接不到 Gateway，请先运行：kubectl port-forward -n ai-gateway svc/inference-gateway 8080:80")

    try:
        pods = find_pods(args.namespace, args.selector)
    except (RuntimeError, subprocess.TimeoutExpired, KeyError) as exc:
        parser.error(f"无法获取 Simulator Pod：{exc}")
    if len(pods) != 3:
        parser.error(f"预期 3 个 Ready Simulator Pod，实际得到 {len(pods)} 个：{pods}")

    print("实际 Pod 映射：")
    for i, pod in enumerate(pods):
        print(f"  {chr(ord('A') + i)} = {pod}")
    print("请不要同时运行其他 Qwen 请求；如有 Rate Limit，请先临时停用该策略。")

    shared = measure(args, "shared", pods)
    unique = measure(args, "unique", pods)
    md = make_markdown(pods, shared, unique)
    print("\n=== 可直接粘贴到博客的 Markdown 数据表 ===\n")
    print(md)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"结果已写入：{args.output}")


def shutil_which_kubectl():
    from shutil import which
    return which("kubectl") is not None


if __name__ == "__main__":
    main()

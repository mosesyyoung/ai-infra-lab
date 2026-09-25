# vLLM Internals Benchmark

对应博客：

《从零搭建一个 AI Infra 实验室⑪：vLLM 为什么快——KV Cache、PagedAttention 与 Continuous Batching》

## Benchmark Results

单位：

- TTFT / TPOT / ITL: ms
- Output Throughput: tok/s

| Experiment | Parameter | P50 TTFT | P99 TTFT | Mean TPOT | P99 ITL | Output Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Context | 256 | 184.96 | 203.17 | 14.81 | 16.79 | 61.93 |
| Context | 512 | 187.34 | 220.53 | 14.88 | 16.67 | 61.60 |
| Context | 1024 | 633.45 | 696.61 | 15.13 | 17.80 | 50.16 |
| Context | 2048 | 1244.36 | 1360.49 | 15.40 | 17.49 | 39.99 |
| Context | 3072 | 1917.76 | 2052.38 | 15.79 | 17.57 | 32.75 |
| Concurrency | 1 | 342.88 | 391.16 | 14.94 | 17.22 | 57.17 |
| Concurrency | 4 | 1166.12 | 1223.67 | 17.23 | 18.42 | 162.09 |
| Concurrency | 8 | 1522.07 | 2442.65 | 20.94 | 21.12 | 227.95 |
| Concurrency | 16 | 2888.05 | 4699.50 | 31.21 | 954.35 | 286.98 |
| max-num-seqs | 4 | 10742.97 | 10959.36 | 17.51 | 19.60 | 160.08 |
| max-num-seqs | 8 | 5682.86 | 6566.85 | 24.25 | 307.89 | 227.20 |
| max-num-seqs | 16 | 3450.45 | 4815.47 | 31.22 | 958.76 | 281.30 |
| Prefix Cache | OFF | 1285.91 | 1366.13 | 15.27 | 16.95 | 34.44 |
| Prefix Cache | ON | 133.67 | 1052.46 | 15.17 | 16.77 | 59.16 |

## Raw Logs

```text
context/       Context Length experiments
concurrency/   Client concurrency experiments
scheduler/     max-num-seqs experiments
prefix-cache/  Prefix Cache OFF / ON experiments
```

## Note

The benchmark server experienced occasional full-system hangs/reboots during testing.

The results were collected from multiple individually completed runs rather than one uninterrupted benchmark session. They are intended to show performance trends, not to serve as a strict hardware benchmark.


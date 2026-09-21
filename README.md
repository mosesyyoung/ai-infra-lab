# AI Infra Lab

从零搭建一个个人 AI Infra 实验室。

这个仓库用于保存我的 AI Infra 学习、实验和博客配套代码，主要环境为：

- Ubuntu 24.04 Server
- NVIDIA GeForce RTX 3060 12GB
- CUDA
- PyTorch
- vLLM
- Docker
- Kubernetes

整个实验路线从 GPU 和 CUDA 开始，逐步进入模型推理、推理引擎、容器化、GPU 调度和 Model Serving。

## Learning Path

```text
GPU
 ↓
CUDA
 ↓
PyTorch
 ↓
Model
 ↓
LLM Inference
 ↓
Quantization
 ↓
llama.cpp / Ollama
 ↓
vLLM
 ↓
Docker
 ↓
Kubernetes
 ↓
Model Serving
 ↓
Observability
```

## Project Structure

```text
ai-infra-lab/
├── README.md
├── .gitignore
│
├── 03-cuda-stack/
│   └── cuda_check.cu
│
├── 04-pytorch-gpu/
│   ├── check_torch.py
│   ├── tensor_device.py
│   ├── matmul_benchmark.py
│   └── precision_benchmark.py
│
├── 05-model-gpu/
│   ├── tokenizer_test.py
│   └── load_model.py
│
├── 06-llm-inference/
│   ├── chat_template_test.py
│   ├── generate_test.py
│   └── prompt_length_test.py
│
├── 07-quantization/
│   └── quant_vram_test.py
│
├── 09-vllm-serving/
│   ├── start_server.sh
│   ├── curl_chat.sh
│   └── vllm_client.py
│
├── 10-serving-benchmark/
│   ├── README.md
│   ├── ollama-c1.log
│   ├── ollama-c4.log
│   ├── ollama-c8.log
│   ├── ollama-c16.log
│   ├── vllm-c1.log
│   ├── vllm-c4.log
│   ├── vllm-c8.log
│   └── vllm-c16.log
│
```

### 03 - CUDA Stack

对应博客：

《从零搭建一个 AI Infra 实验室③：CUDA 到底是什么——拆开 NVIDIA 的软件栈》

这一阶段主要理解：

NVIDIA Driver
CUDA Driver API
CUDA Runtime
CUDA Toolkit
cuDNN
NCCL
Driver / Toolkit 版本兼容关系
CUDA Environment Check

cuda_check.cu 用于验证下面这条链是否真正打通：

```text
CUDA Program
     ↓
CUDA Runtime
     ↓
CUDA Driver
     ↓
NVIDIA GPU
```

编译：

```bash
nvcc 03-cuda-stack/cuda_check.cu -o cuda_check
```

运行：
```bash
./cuda_check
```

程序会输出：

- CUDA Device 数量
- CUDA Driver API Version
- CUDA Runtime Version
- GPU Name
- Compute Capability

例如：

```text
CUDA devices: 1
Driver API version: 13.0
Runtime version: 13.3
GPU: NVIDIA GeForce RTX 3060
Compute Capability: 8.6
```

具体版本以实际环境为准。

### 04 - PyTorch on GPU

对应博客：

《从零搭建一个 AI Infra 实验室④：第一次让 PyTorch 跑在 GPU 上》

这一阶段第一次从 CUDA 软件栈进入 PyTorch Framework，主要实验：

- 检查 PyTorch 是否能够使用 CUDA GPU
- 理解 Tensor 的 `shape`、`dtype` 和 `device`
- 验证 CPU Tensor → GPU Tensor 的数据搬运
- 对比 CPU / GPU 矩阵乘法性能
- 观察 GPU 显存占用和利用率
- 对比 FP32 / FP16 / BF16 的显存与计算表现

核心数据路径：

```text
CPU Tensor
System RAM
     │
     │ .to("cuda")
     ▼
   PCIe
     │
     ▼
GPU Tensor
GPU VRAM
     │
     ▼
PyTorch / CUDA / cuBLAS
     │
     ▼
GPU Compute
```

实验代码：

#### check_torch.py

检查 PyTorch、CUDA 和 GPU 环境：

```bash
python 04-pytorch-gpu/check_torch.py
```

主要输出：

- PyTorch Version
- PyTorch CUDA Version
- CUDA Available
- GPU Name
- Compute Capability
- BF16 Support

#### tensor_device.py

观察 Tensor 从 CPU 内存进入 GPU 显存：

```bash
python 04-pytorch-gpu/tensor_device.py
```

重点观察：

```text
device=cpu
    ↓
.to("cuda")
    ↓
device=cuda:0
```

运行时可以在另一个终端观察 GPU：

```bash
watch -n 0.5 nvidia-smi
```

#### matmul_benchmark.py

对比 CPU 和 GPU 的矩阵乘法性能：

```bash
python 04-pytorch-gpu/matmul_benchmark.py
```

实验重点：

- CPU vs GPU Matrix Multiplication
- GPU 对大规模并行计算的优势
- 小矩阵下 GPU Launch Overhead
- CUDA 异步执行与 torch.cuda.synchronize()

#### precision_benchmark.py

对比不同浮点精度：

```bash
python 04-pytorch-gpu/precision_benchmark.py
```

主要观察：

```text
FP32 = 4 Bytes / element
FP16 = 2 Bytes / element
BF16 = 2 Bytes / element
```

并比较：

- Tensor Size
- GPU Memory
- Matrix Multiplication Time
- FP32 / FP16 / BF16 的实际表现

具体性能数据与 CUDA / PyTorch 版本以实际实验环境为准。

### 05 - LLM Model on GPU

对应博客：

《从零搭建一个 AI Infra 实验室⑤：从 Infra 角度看懂一个 LLM 到底是什么》

这一阶段从 PyTorch Tensor 进一步进入真正的 LLM Model，主要理解：

- Parameter、Weight 与 Model 的关系
- 0.5B / 7B 等模型规模代表什么
- Parameter Count、dtype 与模型内存占用的关系
- Transformer 在 LLM 中的位置
- Tokenizer、Token、Prompt 和 Context
- 模型文件如何加载到 System RAM
- PyTorch Model 如何进一步进入 GPU VRAM

实验模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

核心模型加载路径：

```text
Model Repository
      ↓
Model Files
      ↓
System RAM
      ↓
PyTorch Model
      ↓
model.to("cuda")
      ↓
GPU VRAM 
```

一个模型的参数显存可以粗略理解为：

```text
Parameter Memory
≈
Parameter Count × dtype size
```

例如：

```text
FP32 = 4 Bytes / parameter
FP16 = 2 Bytes / parameter
BF16 = 2 Bytes / parameter
```

因此：

```text
0.5B × FP16 ≈ 1 GB
7B   × FP16 ≈ 14 GB
```

这里计算的是模型参数本身的近似占用，实际推理还会产生其他显存开销。

#### tokenizer_test.py

观察文本如何被 Tokenizer 转换为 Token 和 Token ID：

```bash
python 05-model-gpu/tokenizer_test.py
```

核心数据路径：

```text
Text
 ↓
Tokenizer
 ↓
Token
 ↓
Token ID
 ↓
Model Input
```

#### load_model.py

使用 Transformers 加载模型，并观察模型从 CPU RAM 进入 GPU VRAM：

```bash
python 05-model-gpu/load_model.py
```

重点观察：

- Parameter Count
- Parameter Memory
- Model dtype
- Model device
- torch.cuda.memory_allocated()
- torch.cuda.memory_reserved()
- nvidia-smi

模型首先加载到：

```text
System RAM
```

此时：

```text
Model device: cpu
```

执行：

```text
model.to("cuda")
```

之后，大量 Parameter Tensor 会被搬入 GPU VRAM：

```text
Model device: cuda:0
```

可以在另一个终端同时观察：

```bash
watch -n 0.5 nvidia-smi
```

#### Wi-Fi Power Save

如果实验机通过 Wi-Fi 联网，持续下载较大的模型文件时遇到 SSH 连接异常，可以检查：

```bash
iw dev wlp2s0 get power_save
```

临时关闭：

```bash
sudo iw dev wlp2s0 set power_save off
```

如果关闭后恢复稳定，可以再根据自己的网络环境配置为永久关闭。

这一问题与模型或 PyTorch 本身无直接关系，但持续下载模型文件可能会暴露原本不明显的网络稳定性问题。

### 06 - LLM Inference

对应博客：

《从零搭建一个 AI Infra 实验室⑥：一次 LLM 推理到底发生了什么——从 Prompt 到 Prefill、Decode》

这一阶段从“模型如何进入 GPU”进一步进入真正的 LLM Inference，主要理解：

- Prompt 如何通过 Chat Template 变成模型熟悉的对话格式
- Tokenizer 如何产生 Input Tokens
- Context 到底包含什么
- 真正调用 `model.generate()`
- 为什么 LLM 会一个 Token 一个 Token 地生成
- Prefill 与 Decode 为什么是两种不同的 workload
- KV Cache 为什么能够避免重复计算历史 Token
- Input / Output Token 数如何影响推理
- Short Prompt 与 Long Prompt 的推理差异
- 推理过程中 GPU 显存如何变化

实验模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

一次 LLM 请求的核心路径：

```text
User Prompt
     ↓
Chat Template
     ↓
Tokenizer
     ↓
Input Tokens
     ↓
Context
     ↓
Prefill
     │
     ├── Build KV Cache
     └── Predict Output Token 1
                  ↓
               Decode
                  ↓
          Update KV Cache
                  ↓
         Predict Next Token
                  ↓
                 ...
                  ↓
            Output Tokens
                  ↓
               Answer
```

Prefill and Decode

Prefill 首先处理整个已有 Context：

```text
Prompt Tokens
      ↓
   Prefill
      ↓
Build KV Cache
      ↓
Output Token 1
```

随后进入自回归 Decode。

每一步只处理一个新的 Token：

```text
New Token
   ↓
Q / K / V
   │
   ├── Q → 当前 Attention 使用
   │
   ├── K → append to K Cache
   │
   └── V → append to V Cache
   ↓
Predict Next Token
```

因此正常使用 KV Cache 时，并不会为每个 Output Token 重新执行整个 Prompt 的 Prefill。

KV Cache 保存的是 Transformer 各层历史 Token 的 Key / Value Tensor，而不是文本或 Token ID。

可以粗略理解为：

```text
Q = 当前 Token 想寻找什么
K = 历史 Token 可以如何被匹配
V = 被关注后真正提供的信息
```

历史 K / V 会被后续 Decode Step 反复使用，因此需要缓存；Q 只服务于当前 Attention Step，通常不会长期保存。

#### chat_template_test.py

观察聊天消息如何被 Chat Template 转换成模型训练时熟悉的 Prompt 格式：

```bash
python 06-llm-inference/chat_template_test.py
```

核心路径：

```text
Messages
   ↓
Chat Template
   ↓
Formatted Prompt
```

重点理解：

```text
system
user
assistant
```

等角色信息并不是模型天然理解的，而是通过模型自己的 Chat Template 编码到输入序列中。

#### generate_test.py

真正执行一次 LLM 推理：

```bash
python 06-llm-inference/generate_test.py
```

核心调用：

```text
model.generate(...)
```

重点观察：

- Input Tokens
- Output Tokens
- Generated Answer
- 自回归生成过程

从 Infra 角度，可以把 model.generate() 背后的过程理解为：

```text
Input Tokens
     ↓
Prefill
     ↓
KV Cache
     ↓
Autoregressive Decode
     ↓
Output Tokens
```

#### prompt_length_test.py

比较 Short Prompt 与 Long Prompt：

```bash
python 06-llm-inference/prompt_length_test.py
```

重点记录：

- Input Tokens
- Output Tokens
- Generate Time
- GPU Memory Allocated
- GPU Memory Reserved
- Peak GPU Memory

可以同时在另一个终端观察 GPU：

```bash
watch -n 0.5 nvidia-smi
```

实验主要观察下面这条关系：

```text
Longer Prompt
      ↓
More Input Tokens
      ↓
Larger Context
      ↓
More Prefill Work
      ↓
Larger KV Cache
      ↓
Higher Latency / Memory Pressure
```

这里的 Generate Time 是整个 model.generate() 的总耗时，包括 Prefill、Decode 和部分 Framework 开销，并不等于严格意义上的 TTFT。

后续进入推理引擎时，再进一步观察：

```text
TTFT
TPOT
Throughput
Batching
KV Cache Management
```

### 07 - LLM Quantization

对应博客：

《从零搭建一个 AI Infra 实验室⑦：12GB 显存到底能跑多大模型——模型精度与量化》

这一阶段从 LLM Inference 进一步进入模型显存容量与 Quantization，主要理解：

- Parameter Count 与 Model Weight Size 的关系
- FP32 / FP16 / BF16 / INT8 / INT4 的存储差异
- Quantization 为什么能够降低模型显存占用
- Scale、Zero Point 和 Group Quantization
- BitsAndBytes INT8 / INT4
- AWQ 与 GPTQ 预量化模型
- Model Weight Size 与实际 GPU VRAM Usage 的区别
- Model Fits 与 Serving Fits 的区别

实验模型：

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-7B-Instruct-AWQ
Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
```

模型 Weight 可以粗略估算为：

```text
Weight Size
≈
Parameter Count
×
Bytes Per Parameter
```

例如对于 7B Model：

```text
BF16 / FP16
≈ 7B × 2 Bytes
≈ 14GB

INT8
≈ 7B × 1 Byte
≈ 7GB

INT4
≈ 7B × 0.5 Byte
≈ 3.5GB
```

这里计算的只是理论 Model Weight Size。

真正运行模型时，GPU VRAM 还包括：

```text
GPU VRAM
   │
   ├── Model Weights
   ├── KV Cache
   ├── Activations
   ├── Runtime Buffers
   └── CUDA / Framework Overhead
```

因此：

```text
Model Weight Size
≠
GPU VRAM Usage
```

#### quant_vram_test.py

使用同一个测试程序比较五种模型加载方式：

```text
BF16
INT8
BNB INT4
AWQ INT4
GPTQ INT4
```

运行：

```bash
HF_HUB_OFFLINE=1 python 07-quantization/quant_vram_test.py bf16
HF_HUB_OFFLINE=1 python 07-quantization/quant_vram_test.py int8
HF_HUB_OFFLINE=1 python 07-quantization/quant_vram_test.py int4
HF_HUB_OFFLINE=1 python 07-quantization/quant_vram_test.py awq
HF_HUB_OFFLINE=1 python 07-quantization/quant_vram_test.py gptq
```

实验故意使用：

```python
device_map={"": 0}
```

要求模型完整进入 GPU 0，避免 device_map="auto" 自动把部分 Weight Offload 到 CPU，从而影响显存容量实验。

程序主要记录：

```text
Load Time
Model Footprint
CUDA Memory Allocated
CUDA Memory Reserved
Peak VRAM
Input Tokens
Output Tokens
Generate Time
```

可以同时使用：

```bash
watch -n 0.5 nvidia-smi
```

观察实际 GPU Process Memory。

#### Download Models First

为了避免模型下载过程干扰显存实验，先使用 hf download 把模型下载到 Hugging Face Cache：

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1

hf download Qwen/Qwen2.5-7B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct-AWQ
hf download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
```

然后在实验阶段使用：

```bash
HF_HUB_OFFLINE=1
local_files_only=True
```

完全离线加载模型。

本机实验中使用默认 Xet 下载路径时曾出现系统异常，因此关闭：

```bash
export HF_HUB_DISABLE_XET=1
```

后续下载可以正常 Resume 并完成。

这一现象仅代表当前实验环境，不意味着 Xet 在其他环境中存在相同问题。

#### AWQ / GPTQ Runtime

AWQ / GPTQ 是预量化模型，与 BitsAndBytes 加载时量化不同。

当前实验环境额外使用：

```text
gptqmodel
torchvision
optimum
```

实际依赖与版本可能随 Transformers Quantization Backend 更新而变化，因此应以实验时的官方文档和当前 Python Environment 为准。

这一阶段最终建立下面这条关系：

```text
Parameter Count
       ↓
Precision / Quantization
       ↓
Model Weight Size
       ↓
GPU VRAM
```

对于 RTX 3060 12GB：

```text
7B BF16
→ Weight 本身已经接近或超过显存容量

7B INT8
→ 开始进入合理范围

7B INT4
→ 留出明显更多显存空间
```

但：

```text
Model Fits
≠
Serving Fits
```

长 Context 和更高并发仍然会继续增加 KV Cache 等显存开销。

### 08 - Local LLM Runtime

对应博客：

《从零搭建一个 AI Infra 实验室⑧：Ollama 与 llama.cpp——本地 LLM 推理为什么还有另一条路线》

这一阶段从 Transformers / PyTorch 推理继续进入 Model Runtime，主要理解：

- Model 与 Runtime 的区别
- GGUF 与 Hugging Face Model Format 的定位差异
- llama.cpp 的 CPU / GPU Hybrid Inference
- GPU Offload
- Runtime Context Size
- Ollama 的 Model Management、Lifecycle 和 Local API

核心路线：

```text
GGUF
 ↓
llama.cpp
 ↓
CPU / GPU Offload
 ↓
Ollama
 ↓
Local Model Service
```

这一阶段主要使用现成 Runtime 和 CLI，没有额外的自定义实验代码。

### 09 - vLLM Serving

对应博客：

《从零搭建一个 AI Infra 实验室⑨：从“跑模型”到“模型服务器”——为什么需要 vLLM》

这一阶段从 Local Model Runtime 继续进入 LLM Serving，主要理解：

- Model Runtime 与 Model Server 的区别
- Transformers、llama.cpp、Ollama 与 vLLM 的定位差异
- Ollama 与 vLLM 都可以提供 API，但优化目标不同
- OpenAI Compatible API
- Application 与 Model Runtime 的解耦
- vLLM Server 的基本启动方式
- 使用 curl 调用 Chat Completions API
- 使用 OpenAI Python SDK 调用本地 vLLM

实验模型：

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

核心架构：

```text
Application
     │
     │ HTTP
     ▼
OpenAI-Compatible API
     │
     ▼
vLLM Server
     │
     ▼
Inference Engine
     │
     ▼
Model
     │
     ▼
GPU
```

#### start_server.sh

启动本地 vLLM Server：

```bash
bash 09-vllm-serving/start_server.sh
```

实验配置：

```text
Model: Qwen/Qwen2.5-7B-Instruct-AWQ
Served Model Name: qwen2.5-7b-awq
Host: 127.0.0.1
Port: 8000
Max Model Length: 4096
GPU Memory Utilization: 0.85
```

#### curl_chat.sh

使用 OpenAI Compatible API 调用本地模型：

```bash
bash 09-vllm-serving/curl_chat.sh
```

请求路径：

```text
Client
 ↓
POST /v1/chat/completions
 ↓
vLLM Server
 ↓
Model
 ↓
GPU
 ↓
JSON Response
```

#### vllm_client.py

使用 OpenAI Python SDK 调用同一个本地 Server：

```bash
python 09-vllm-serving/vllm_client.py
```

核心变化是从：

```python
model.generate(...)
```

进入：

```python
client.chat.completions.create(...)
```

Application 不再直接拥有 Model，而是通过标准 API 调用长期运行的 Model Service。


### 10 - LLM Serving Benchmark

对应博客：

《从零搭建一个 AI Infra 实验室⑩：LLM 性能怎么看——TTFT、TPOT、ITL 和 Throughput》

这一阶段从“如何把模型变成 Server”进一步进入 LLM Serving Performance，主要理解：

- TTFT（Time To First Token）
- TPOT（Time Per Output Token）
- ITL（Inter-Token Latency）
- End-to-End Latency
- Request Throughput
- Output Tokens/s
- P50 / P95 / P99 与 Tail Latency
- Concurrency 与 Latency / Throughput 的关系

实验继续使用：

```text
Ubuntu 24.04 Server
Intel Core i5-10400F
32GB RAM
NVIDIA GeForce RTX 3060 12GB
```

使用同一个：

```text
vllm bench serve
```

作为 Benchmark Client，对：

```text
Ollama
vLLM
```

分别进行：

```text
Concurrency
1
4
8
16
```

四档测试。

实验控制：

```text
Input Length ≈ 512 Tokens
Output Length ≈ 128 Tokens
Requests = 100
Request Rate = inf
```

主要观察两类指标：

```text
User Experience
├── TTFT
├── TPOT
├── ITL
└── E2E Latency

System Capacity
├── Requests/s
└── Output Tokens/s
```

实验结果显示，在 Concurrency=1 时，两种 Runtime 的 Output Throughput 比较接近：

```text
Ollama : 55.91 tokens/s
vLLM   : 57.51 tokens/s
```

随着并发提高，两者的 Scaling Behavior 开始明显不同。

```text
Output Tokens/s

Concurrency     Ollama      vLLM
1               55.91       57.51
4               53.39      161.08
8               51.41      228.58
16              49.90      287.27
```

同时，TTFT、TPOT、ITL 和 Tail Latency 也会随着并发产生不同变化。

这说明评价一个 LLM Server 不能只看单请求 tokens/s，而需要同时观察：

```text
Latency
+
Throughput
+
Concurrency
+
Percentile
```

原始 Benchmark 输出保存在：

```text
10-serving-benchmark/
```

其中：

```text
ollama-c*.log
vllm-c*.log
```

分别对应 Ollama 和 vLLM 在不同并发下的完整 Benchmark 输出。

Note: 本次 vLLM Benchmark 过程中，实验服务器出现过偶发的整机卡死/重启，因此本次 vLLM 实验的 Concurrency=1、4、8、16 的结果来自多次实验中分别完成的成功测试，并非一次连续 Benchmark Run。当前硬件稳定性问题仍在进一步排查，因此这些数据主要用于学习和观察 Scaling Behavior，而不是严格的硬件性能基准。

## About

这个项目主要用于个人学习和实验。

目标不是第一天就搭建完整 AI 平台，而是按照：

```text
遇到问题
 ↓
理解原理
 ↓
做实验
 ↓
观察结果
 ↓
再加入下一层 Infra
```

逐步理解完整的 AI Infra 技术栈。





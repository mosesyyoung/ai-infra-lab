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

下一阶段将进一步观察一次 LLM 推理请求内部发生的事情：

```text
Prompt
 ↓
Chat Template
 ↓
Tokenizer
 ↓
Context
 ↓
Prefill
 ├── Build KV Cache
 └── Output Token 1
          ↓
       Decode
          ↓
   Update KV Cache
          ↓
   Next Output Token
```

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

下一阶段将继续研究一个非常现实的问题：

```text
12GB VRAM
     ↓
FP16 / BF16
     ↓
INT8
     ↓
INT4
     ↓
Quantization
     ↓
How Large a Model Can We Run?
```

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





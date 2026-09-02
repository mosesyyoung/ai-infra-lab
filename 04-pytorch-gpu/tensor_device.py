import torch

x_cpu = torch.randn(8192, 8192)

print(x_cpu.shape)
print(x_cpu.dtype)
print(x_cpu.device)

x_gpu = x_cpu.to("cuda")

print(x_gpu.device)

print(
    torch.cuda.memory_allocated() / 1024**2,
    "MiB allocated"
)

print(
    torch.cuda.memory_reserved() / 1024**2,
    "MiB reserved"
)


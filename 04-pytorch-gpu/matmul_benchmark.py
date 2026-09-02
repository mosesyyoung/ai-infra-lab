import time
import torch

N = 4096
REPEAT = 5

print("Matrix :", f"{N} x {N}")
print("GPU    :", torch.cuda.get_device_name(0))

# -------------------
# CPU
# -------------------

a_cpu = torch.randn(N, N)
b_cpu = torch.randn(N, N)

# warm up
_ = a_cpu @ b_cpu

start = time.perf_counter()

for _ in range(REPEAT):
    c_cpu = a_cpu @ b_cpu

cpu_time = (time.perf_counter() - start) / REPEAT

# -------------------
# GPU
# -------------------

a_gpu = a_cpu.to("cuda")
b_gpu = b_cpu.to("cuda")

# warm up
for _ in range(5):
    _ = a_gpu @ b_gpu

torch.cuda.synchronize()

start = time.perf_counter()

for _ in range(REPEAT):
    c_gpu = a_gpu @ b_gpu

torch.cuda.synchronize()

gpu_time = (time.perf_counter() - start) / REPEAT

print(f"CPU : {cpu_time * 1000:.2f} ms")
print(f"GPU : {gpu_time * 1000:.2f} ms")
print(f"Speedup : {cpu_time / gpu_time:.2f} x")

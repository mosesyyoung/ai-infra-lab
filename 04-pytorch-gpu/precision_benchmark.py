import time
import torch

N = 4096
REPEAT = 20
DEVICE = "cuda"

print("GPU :", torch.cuda.get_device_name(0))

def benchmark(dtype):

    a = torch.randn(
        N, N,
        device=DEVICE,
        dtype=dtype
    )

    b = torch.randn(
        N, N,
        device=DEVICE,
        dtype=dtype
    )

    # warm up
    for _ in range(5):
        c = a @ b

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(REPEAT):
        c = a @ b

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() - start
    ) / REPEAT

    tensor_size = (
        a.numel()
        * a.element_size()
        / 1024**2
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    print()
    print("dtype      :", dtype)
    print("element    :", a.element_size(), "Bytes")
    print("one tensor :", f"{tensor_size:.2f} MiB")
    print("matmul     :", f"{elapsed * 1000:.3f} ms")
    print("peak VRAM  :", f"{peak_memory:.2f} MiB")


benchmark(torch.float32)
benchmark(torch.float16)

if torch.cuda.is_bf16_supported():
    benchmark(torch.bfloat16)

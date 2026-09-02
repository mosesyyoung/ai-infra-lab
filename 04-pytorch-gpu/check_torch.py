import torch

print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())
print("PyTorch CUDA    :", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU Name        :", torch.cuda.get_device_name(0))
    print("GPU Capability  :", torch.cuda.get_device_capability(0))
    print("BF16 Supported  :", torch.cuda.is_bf16_supported())

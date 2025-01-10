import torch

# Get the PyTorch version
torch_version = torch.__version__

# Get the CUDA version (if available)
cuda_version = torch.version.cuda if torch.version.cuda else "CUDA is not available"

# Print the versions
print(f"PyTorch Version: {torch_version}")
print(f"CUDA Version: {cuda_version}")

# Check if CUDA is available and print details
if torch.cuda.is_available():
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    print(f"Default CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available on this system.")
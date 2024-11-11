import torch

if torch.cuda.is_available():
    print("CUDA is available. You are using", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

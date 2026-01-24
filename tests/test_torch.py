import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    
    # Create tensors on GPU
    x = torch.rand(1000, 1000, device=device)
    y = torch.rand(1000, 1000, device=device)
    
    # Matrix multiplication on GPU
    z = x @ y
    
    print(f"\nMatrix multiplication on MPS successful!")
    print(f"Result shape: {z.shape}")

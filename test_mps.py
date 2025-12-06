import torch
import sys

print(f"Python: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Torch compiled with MPS: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print("✅ MPS 完全可用！")
    print(x)
else:
    print("❌ MPS 不可用")

import torch
try:
    import diffusers
except ImportError:
    pass
import transformers
import sentence_transformers

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
if torch.cuda.is_available():
    print("VRAM total GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)

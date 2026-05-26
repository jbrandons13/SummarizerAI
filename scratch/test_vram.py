import sys
import torch
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.vram import VRAMManager
from src.models.siglip import SigLIPEncoder

def test():
    device = 0
    # 1. Instantiate VRAMManager
    vram = VRAMManager(device_id=device)
    
    # Measure memory before loading
    mem_before = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Memory before load: {mem_before:.4f} GB")
    
    # 2. Load model
    print("Loading SigLIP model...")
    encoder = SigLIPEncoder(vram)
    encoder._load()
    
    mem_after = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Memory after load: {mem_after:.4f} GB")
    
    # 3. Unload model
    print("Deleting encoder reference...")
    del encoder
    
    print("Unloading model from VRAMManager...")
    vram.unload_current_model()
    
    mem_after_unload = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Memory after unload: {mem_after_unload:.4f} GB")
    
    success = mem_after_unload < (mem_after - 0.1)  # significant drop
    print(f"Unload working: {success}")

if __name__ == "__main__":
    test()

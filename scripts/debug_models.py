import torch
from transformers import AutoProcessor, AutoModel
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")

model_name = "google/siglip2-so400m-patch16-naflex"
print(f"\nAttempting to load {model_name}...")
try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print("Successfully loaded SigLIP 2!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"Failed to load SigLIP 2: {e}")

qwen_name = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
print(f"\nAttempting to load {qwen_name}...")
try:
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(qwen_name, trust_remote_code=True, device_map="cpu")
    print("Successfully loaded Qwen2.5-VL!")
except Exception as e:
    print(f"Failed to load Qwen2.5-VL: {e}")

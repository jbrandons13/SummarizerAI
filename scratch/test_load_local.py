from diffusers import LTXImageToVideoPipeline
import torch
import os

model_path = "/home/wins053/models/ltx_video_distilled"
print("Checking model path:", model_path)
print("Exists:", os.path.exists(model_path))

try:
    pipe = LTXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    print("Loaded successfully!")
    print("Class:", pipe.__class__)
except Exception as e:
    print("Error:", e)

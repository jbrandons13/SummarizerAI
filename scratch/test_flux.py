import torch
from diffusers import FluxPipeline
from PIL import Image
import os

print("Setting up test script...")
model_id = "black-forest-labs/FLUX.1-schnell"
cache_dir = os.path.expanduser("~/models/flux_schnell")

print(f"Loading {model_id} from Hugging Face with cache_dir={cache_dir}...")
pipe = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
)

print("Enabling sequential CPU offload...")
pipe.enable_model_cpu_offload()

print("Loading IP-Adapter...")
try:
    pipe.load_ip_adapter(
        "XLabs-AI/flux-ip-adapter",
        weight_name="ip_adapter.safetensors",
        image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
    )
    print("IP-Adapter loaded successfully!")
except Exception as e:
    print(f"Failed to load IP-Adapter: {e}")

print("Done.")

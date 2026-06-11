import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

prompts = ["A cartoon Sun"] * 16

try:
    images = pipe(prompts, height=768, width=768, num_inference_steps=2).images
    print(f"Success: batch size {len(prompts)} fits at 768x768!")
except Exception as e:
    print(f"Failed with batch size {len(prompts)} at 768x768: {e}")

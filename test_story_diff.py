import sys
sys.path.append("./StoryDiffusion")
import torch
from diffusers import StableDiffusionXLPipeline
from StoryDiffusion.utils.gradio_utils import set_attention_processor

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

prompts = [
    "A cartoon Sun with a smiling face",
    "A cartoon Sun wearing sunglasses",
    "A cartoon Sun holding a book"
]

# Vanilla
images_vanilla = pipe(prompts, num_inference_steps=20).images
for i, img in enumerate(images_vanilla):
    img.save(f"vanilla_{i}.png")

# Consistent
set_attention_processor(pipe.unet, len(prompts), is_ipadapter=False)
images_consistent = pipe(prompts, num_inference_steps=20).images
for i, img in enumerate(images_consistent):
    img.save(f"consistent_{i}.png")

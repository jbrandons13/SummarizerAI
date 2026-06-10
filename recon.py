import torch
import diffusers
import sys
import os
import yaml
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection

def main():
    print('Torch:', torch.__version__)
    print('Diffusers:', diffusers.__version__)

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_model = config["phase4"]["image_gen"]["base_model"]
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        image_encoder=image_encoder
    )
    
    ip_config = config["phase4"]["image_gen"].get("ip_adapter", {})
    repo = ip_config.get("repo", "h94/IP-Adapter")
    subfolder = ip_config.get("subfolder", "sdxl_models")
    weight_name = ip_config.get("weight_name", "ip-adapter-plus_sdxl_vit-h.safetensors")
    pipe.load_ip_adapter(repo, subfolder=subfolder, weight_name=weight_name)

    print("Pipeline Class:", pipe.__class__.__name__)
    print("Scheduler Class:", pipe.scheduler.__class__.__name__)

    try:
        ip_layer = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        print("IP-Adapter Variant Class:", ip_layer.__class__.__name__)
    except Exception as e:
        print("IP-Adapter Variant Check Failed:", e)
        
    try:
        # Does pipe.set_ip_adapter_scale accept dict?
        pipe.set_ip_adapter_scale({"down": 0.5})
        print("pipe.set_ip_adapter_scale accepts dict: YES")
    except Exception as e:
        print("pipe.set_ip_adapter_scale accepts dict: NO (", e, ")")
        
    with open("unet_attn_map.txt", "w") as f:
        for k in pipe.unet.attn_processors.keys():
            f.write(k + "\n")
    print("Saved unet_attn_map.txt")

if __name__ == "__main__":
    main()

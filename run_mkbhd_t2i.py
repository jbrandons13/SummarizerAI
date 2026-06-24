import os
import json
import torch
from diffusers import StableDiffusionXLPipeline

def main():
    json_path = "data/intermediate/iGeXGdYE7UE/summary_script.json"
    out_dir = "data/intermediate/iGeXGdYE7UE/t2i_images"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    shots = data.get("shots") or data.get("sentences") or data.get("segments") or []
    
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")
    
    # We use a base subject to give context
    base_subject = "Tech reviewer Marques Brownlee in a studio setup. "
    
    for s in shots:
        shot_id = s.get("id")
        text = s.get("text", "")
        keywords = ", ".join(s.get("keywords", []))
        
        prompt = base_subject + text + " " + keywords + ", high quality, 4k, photorealistic"
        out_path = os.path.join(out_dir, f"shot_{shot_id:03d}.png")
        
        if os.path.exists(out_path):
            print(f"Skipping {out_path}, already exists.")
            continue
            
        print(f"Generating T2I for shot_{shot_id:03d}...")
        print(f"Prompt: {prompt}")
        
        # MKBHD videos are usually 16:9, so 1024x576 or 1152x648. Let's use 1024x576.
        image = pipe(prompt=prompt, num_inference_steps=30, width=1024, height=576).images[0]
        image.save(out_path)
        print(f"Saved {out_path}")
        
    print("T2I Generation Complete!")

if __name__ == "__main__":
    main()

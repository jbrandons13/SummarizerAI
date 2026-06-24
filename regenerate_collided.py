import json
import os
import torch
import yaml
from src.phase4.image_gen import load_pipeline, get_deterministic_seed

def main():
    video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    sb_path = f"data/intermediate/{video}/phase4/storyboard.json"
    with open(sb_path, "r") as f:
        sb = json.load(f)["shots"]
        
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    from diffusers import EulerDiscreteScheduler
    pipe, _ = load_pipeline(config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    from src.phase4.image_gen import _prep_reference
    from PIL import Image
    ref_path = "runs/geology/reference.png"
    ref_image = Image.open(ref_path).convert("RGB")
    ip_img = _prep_reference(ref_image, mode="crop")
    
    # Old seeds: 
    seeds = {f"shot_{i:03d}": get_deterministic_seed(f"shot_{i:03d}") for i in range(1, 17)}
    w_grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    
    run_dir = "runs/G0_A0_geology_collided"
    os.makedirs(run_dir, exist_ok=True)
    
    for shot in sb:
        shot_id = shot["shot_id"] # shot_001
        seed = seeds[shot_id]
        prompt = "fca style, " + shot["image_prompt"]
        neg = "text, letters, words, numbers, captions, labels, infographic, diagram, panels, charts, table, watermark, signature, gibberish text"
        
        # Original latents behavior:
        shape = (1, 4, 768 // 8, 1344 // 8)
        
        for w in w_grid:
            img_dir = os.path.join(run_dir, f"w{w:.2f}")
            os.makedirs(img_dir, exist_ok=True)
            out_path = os.path.join(img_dir, f"{shot_id}.png")
            
            if os.path.exists(out_path): continue
            
            gen = torch.Generator(device="cuda").manual_seed(seed)
            latents = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float16)
            
            pipe.set_ip_adapter_scale(w)
            out = pipe(
                prompt=prompt,
                negative_prompt=neg,
                width=1344, height=768,
                num_inference_steps=30,
                guidance_scale=7.0,
                latents=latents,
                ip_adapter_image=ip_img if w > 0 else Image.new("RGB", (224, 224), "black")
            ).images[0].resize((832, 480), Image.LANCZOS)
            out.save(out_path)

if __name__ == "__main__":
    main()

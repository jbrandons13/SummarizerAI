import sys
sys.modules['gptqmodel'] = None
sys.modules['awq'] = None
sys.modules['bitsandbytes'] = None
import peft.import_utils
peft.import_utils.is_auto_awq_available = lambda: False
peft.import_utils.is_gptqmodel_available = lambda: False
peft.import_utils.is_auto_gptq_available = lambda: False

import torch
import json
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import os
from PIL import Image
import yaml
from pipeline.facet.scoring_wrap import ScoringWrap

def main():
    video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    sb_path = f"data/intermediate/{video}/phase4/storyboard.json"
    with open(sb_path, "r") as f:
        sb = json.load(f)["shots"]

    with open("pipeline/facet/seeds.json", "r") as f:
        seeds = json.load(f)

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    from src.phase4.image_gen import load_pipeline, generate_image, _prep_reference
    
    pipe, _ = load_pipeline(config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    ref_path = "runs/geology/reference.png"
    ref_image = Image.open(ref_path).convert("RGB")

    shots_to_probe = ["shot_005", "shot_011"]
    weights = [0.0, 0.4]

    scorer = ScoringWrap()

    os.makedirs("runs/probe", exist_ok=True)
    
    print("Running probe...")

    for shot_id in shots_to_probe:
        seed = seeds[shot_id]
        shot = next(s for s in sb if s["shot_id"] == shot_id)

        prompt = "fca style, " + shot["image_prompt"]
        neg = "text, letters, words, numbers, captions, labels, infographic, diagram, panels, charts, table, watermark, signature, gibberish text"

        # Initialize explicit latent
        shape = (1, 4, 768 // 8, 1344 // 8)
        gen1 = torch.Generator(device="cuda").manual_seed(seed)
        latent_explicit = torch.randn(shape, generator=gen1, device="cuda", dtype=torch.float16)

        for w in weights:
            # 1. Original repo code path (generate_image)
            # generate_image doesn't take latents, it just takes seed.
            config["phase4"]["image_gen"]["concept_anchor_ref_weight"] = w
            # we need to simulate generate_image without literally calling it because we need exactly what it does
            gen2 = torch.Generator(device="cuda").manual_seed(seed)
            ip_img = _prep_reference(ref_image, mode="crop")
            pipe.set_ip_adapter_scale(w)
            
            # This simulates original code path exactly:
            out_orig = pipe(
                prompt=prompt,
                negative_prompt=neg,
                width=1344, height=768,
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=gen2,
                ip_adapter_image=ip_img if w > 0 else Image.new("RGB", (224, 224), "black")
            ).images[0].resize((832, 480), Image.LANCZOS)
            p_orig = f"runs/probe/{shot_id}_w{w}_orig.png"
            out_orig.save(p_orig)

            # 2. runner.py code path
            gen3 = torch.Generator(device="cuda").manual_seed(seed)
            pipe.set_ip_adapter_scale(w)
            
            out_runner = pipe(
                prompt=prompt,
                negative_prompt=neg,
                width=1344, height=768,
                num_inference_steps=30,
                guidance_scale=7.0,
                latents=latent_explicit.clone(),
                ip_adapter_image=ip_img if w > 0 else Image.new("RGB", (224, 224), "black")
            ).images[0].resize((832, 480), Image.LANCZOS)
            p_runner = f"runs/probe/{shot_id}_w{w}_runner.png"
            out_runner.save(p_runner)
            
            # Compare DINO similarity between the two
            emb1 = scorer.embed_dino(p_orig)
            emb2 = scorer.embed_dino(p_runner)
            sim = float((emb1 * emb2).sum().item())
            print(f"{shot_id} @ w={w}: DINO cos sim = {sim:.4f}")

if __name__ == "__main__":
    main()

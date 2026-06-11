import os
import json
import torch
import yaml
import time
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from pipeline.facet.scoring_wrap import ScoringWrap
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor

def main():
    os.makedirs("runs/blockprobe", exist_ok=True)
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("pipeline/facet/seeds.json", "r") as f:
        seeds = json.load(f)
        
    def get_prompts(video):
        with open(f"data/intermediate/{video}/phase4/storyboard.json", "r") as f:
            return json.load(f)["shots"]

    geo_sb = get_prompts("lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge")
    eco_sb = get_prompts("8JntNOAhKjM_crash-course-ecology-2")
    
    # 6 shots: 3 geo, 3 eco
    probe_shots = [
        ("geo", "geo_001", "runs/geology/reference.png", geo_sb),
        ("geo", "geo_002", "runs/geology/reference.png", geo_sb),
        ("geo", "geo_003", "runs/geology/reference.png", geo_sb),
        ("eco", "eco_001", "runs/ecology/reference.png", eco_sb),
        ("eco", "eco_002", "runs/ecology/reference.png", eco_sb),
        ("eco", "eco_003", "runs/ecology/reference.png", eco_sb),
    ]

    from src.phase4.image_gen import load_pipeline, _prep_reference
    
    pipe, _ = load_pipeline(config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    scorer = ScoringWrap()

    # We use pipe.set_ip_adapter_scale with a dict to route to specific layers.
    # The valid keys for diffusers are the processor names in pipe.unet.attn_processors
    
    sites = {
        "global": "global",
        "down2.att0": "down_blocks.2.attentions.0",
        "down2.att1": "down_blocks.2.attentions.1",
        "mid.att0": "mid_block.attentions.0",
        "up0.att0": "up_blocks.0.attentions.0",
        "up0.att1": "up_blocks.0.attentions.1",
        "up0.att2": "up_blocks.0.attentions.2"
    }

    scales = [0.3, 0.5, 0.8]
    
    results = {site: {} for site in sites}
    w0_cache = {}

    start_time = time.time()
    total_gens = len(probe_shots) * (1 + len(sites) * len(scales))
    gen_idx = 0
    
    for domain, shot_id, ref_path, sb in probe_shots:
        ref_image = Image.open(ref_path).convert("RGB")
        ip_img = _prep_reference(ref_image, mode="crop")
        
        shot_num = shot_id.split("_")[1]
        shot_data = next(s for s in sb if s["shot_id"] == f"shot_{shot_num}")
        prompt = "fca style, " + shot_data["image_prompt"]
        neg = "text, letters, words, numbers, captions, labels, infographic, diagram, panels, charts, table, watermark, signature, gibberish text"
        seed = seeds[shot_id]
        
        # Shared w=0 render
        pipe.set_ip_adapter_scale(0.0)
        gen = torch.Generator("cuda").manual_seed(seed)
        w0_img = pipe(prompt=prompt, negative_prompt=neg, width=1344, height=768, num_inference_steps=30, guidance_scale=7.0, generator=gen, ip_adapter_image=ip_img).images[0]
        w0_path = f"runs/blockprobe/{shot_id}_w0.png"
        w0_img.save(w0_path)
        w0_emb = scorer.embed_dino(w0_path)
        w0_cache[shot_id] = (w0_path, w0_emb)
        
        gen_idx += 1
        print(f"[{gen_idx}/{total_gens}] w=0 {shot_id}")
        
        for site_name, prefix in sites.items():
            results[site_name][shot_id] = {}
            for w in scales:
                if site_name == "global":
                    pipe.set_ip_adapter_scale(w)
                else:
                    scales_dict = {}
                    for proc_name in pipe.unet.attn_processors.keys():
                        if "attn2" in proc_name:
                            scales_dict[proc_name] = w if prefix in proc_name else 0.0
                    pipe.set_ip_adapter_scale(scales_dict)
                    
                gen = torch.Generator("cuda").manual_seed(seed)
                img = pipe(prompt=prompt, negative_prompt=neg, width=1344, height=768, num_inference_steps=30, guidance_scale=7.0, generator=gen, ip_adapter_image=ip_img).images[0]
                img_path = f"runs/blockprobe/{shot_id}_{site_name}_w{w}.png"
                img.save(img_path)
                
                emb = scorer.embed_dino(img_path)
                c_s = float((emb * w0_emb).sum().item())
                ref_sim = float((emb * scorer.embed_dino(ref_path)).sum().item())
                
                results[site_name][shot_id][w] = {
                    "c_s": c_s,
                    "ref_sim": ref_sim,
                    "path": img_path
                }
                gen_idx += 1
                if gen_idx % 10 == 0:
                    print(f"Heartbeat: [{gen_idx}/{total_gens}] - ETA: {(time.time()-start_time)/gen_idx * (total_gens-gen_idx):.0f}s")
                    
    with open("runs/blockprobe/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate contact sheets per site
    import matplotlib.pyplot as plt
    for site_name in sites:
        images = []
        for domain, shot_id, ref_path, _ in probe_shots:
            row = [Image.open(w0_cache[shot_id][0])]
            for w in scales:
                row.append(Image.open(results[site_name][shot_id][w]["path"]))
            images.extend(row)
        
        tensors = [to_tensor(img) for img in images]
        grid = make_grid(tensors, nrow=4, padding=2) # w0 + 3 scales = 4 cols
        import torchvision
        torchvision.transforms.functional.to_pil_image(grid).save(f"runs/blockprobe/site_{site_name}.png")

if __name__ == "__main__":
    main()

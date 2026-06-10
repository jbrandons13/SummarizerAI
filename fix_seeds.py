import json
import os
import torch
import hashlib
from src.phase4.image_gen import get_deterministic_seed

def main():
    geo_video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    eco_video = "2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology"
    
    geo_path = f"data/intermediate/{geo_video}/phase4/storyboard.json"
    eco_path = f"data/intermediate/{eco_video}/phase4/storyboard.json"
    
    if os.path.exists(geo_path):
        with open(geo_path, "r") as f:
            geo_data = json.load(f)
        for i, shot in enumerate(geo_data["shots"]):
            shot["shot_id"] = f"geo_{i+1:03d}"
            shot["id"] = shot["shot_id"]
        with open(geo_path, "w") as f:
            json.dump(geo_data, f, indent=4)
            
    if os.path.exists(eco_path):
        with open(eco_path, "r") as f:
            eco_data = json.load(f)
        for i, shot in enumerate(eco_data["shots"]):
            shot["shot_id"] = f"eco_{i+1:03d}"
            shot["id"] = shot["shot_id"]
        with open(eco_path, "w") as f:
            json.dump(eco_data, f, indent=4)

    seeds = {}
    ledger = {}
    
    latents_dir = "pipeline/facet/latents_cache"
    os.makedirs(latents_dir, exist_ok=True)
    
    # Generate 30 seeds
    for i in range(1, 15):
        sid = f"geo_{i:03d}"
        seed = get_deterministic_seed(sid)
        seeds[sid] = seed
        
    for i in range(1, 17):
        sid = f"eco_{i:03d}"
        seed = get_deterministic_seed(sid)
        seeds[sid] = seed
        
    with open("pipeline/facet/seeds.json", "w") as f:
        json.dump(seeds, f, indent=4)
        
    # Generate latents and hashes
    shape = (1, 4, 768 // 8, 1344 // 8)
    for sid, seed in seeds.items():
        latent_path = os.path.join(latents_dir, f"{sid}.pt")
        gen = torch.Generator(device="cuda").manual_seed(seed)
        latents = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float16)
        torch.save(latents, latent_path)
        
        # Hash it
        with open(latent_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        ledger[sid] = h
        
    with open("pipeline/facet/ledger.json", "w") as f:
        json.dump(ledger, f, indent=4)

if __name__ == "__main__":
    main()

import sys
import os
import time
import json
import hashlib

sys.modules['gptqmodel'] = None
sys.modules['awq'] = None
sys.modules['bitsandbytes'] = None
import peft.import_utils
peft.import_utils.is_auto_awq_available = lambda: False
peft.import_utils.is_gptqmodel_available = lambda: False
peft.import_utils.is_auto_gptq_available = lambda: False

import torch
import diffusers
import yaml
from PIL import Image

sys.path.append(os.path.abspath("."))
from src.phase4.image_gen import load_pipeline, generate_image, _prep_reference

def get_run_dir(run_stamp):
    return os.path.join("runs", run_stamp)

def init_latents(pipe, seeds, run_dir, resolution):
    import hashlib
    latents_dir = "pipeline/facet/latents_cache"
    latents_map = {}
    
    with open("pipeline/facet/ledger.json", "r") as f:
        ledger = json.load(f)
        
    for shot_id, seed in seeds.items():
        latent_path = os.path.join(latents_dir, f"{shot_id}.pt")
        if not os.path.exists(latent_path):
            print(f"Warning: Latent missing for {shot_id}")
            continue
            
        with open(latent_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            
        assert h == ledger.get(shot_id), f"Latent hash mismatch for {shot_id}! Expected {ledger.get(shot_id)}, got {h}"
        
        latents = torch.load(latent_path, map_location="cuda")
        latents_map[shot_id] = latents
        
    return latents_map

def run_arm(stamp, arm, video, w_grid, tau, kwargs=None):
    print(f"Torch: {torch.__version__}, Diffusers: {diffusers.__version__}")
    
    run_dir = get_run_dir(stamp)
    os.makedirs(run_dir, exist_ok=True)
    
    with open("configs/facet.yaml", "r") as f:
        facet_conf = yaml.safe_load(f)
    with open("configs/default.yaml", "r") as f:
        default_conf = yaml.safe_load(f)
        
    with open("pipeline/facet/seeds.json", "r") as f:
        seeds = json.load(f)
        
    pipe, _ = load_pipeline(default_conf)
    pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    try:
        pipe.set_ip_adapter_scale({"down": 0.5})
        print("pipe.set_ip_adapter_scale accepts dict: YES")
    except Exception as e:
        print(f"pipe.set_ip_adapter_scale accepts dict: NO ({e})")
        
    with open(os.path.join(run_dir, "unet_attn_map.txt"), "w") as f:
        for k in pipe.unet.attn_processors.keys():
            f.write(k + "\n")
    print(f"Saved unet_attn_map.txt to {run_dir}/unet_attn_map.txt")
    
    latents_map = init_latents(pipe, seeds, run_dir, facet_conf["resolution"])
    
    sb_path = f"data/intermediate/{video}/phase4/storyboard.json"
    summ_path = f"data/intermediate/{video}/summary_script.json"
    with open(sb_path, "r") as f:
        sb = json.load(f)["shots"]
    with open(summ_path, "r") as f:
        summ = json.load(f)["sentences"]
    
    narration_map = {}
    for shot in sb:
        sid = shot["shot_id"]
        narration_map[sid] = shot["visual_description"] 
        
    if video == "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge":
        ref_path = "runs/geology/reference.png" 
    else:
        ref_path = "runs/ecology/reference.png"
        
    if not os.path.exists(ref_path):
        print(f"Warning: {ref_path} not found. Using empty ref.")
        ref_image = Image.new("RGB", (224, 224), "black")
    else:
        ref_image = Image.open(ref_path).convert("RGB")
        
    records = []
    
    for shot in sb:
        shot_id = shot["shot_id"]
        seed = seeds.get(shot_id)
        if seed is None:
            continue
        
        for w in w_grid:
            img_dir = os.path.join(run_dir, "images", arm, f"w{w:.2f}")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{shot_id}.png")
            
            rec = {
                "run_id": stamp,
                "stage": 0,
                "arm": arm,
                "video": video,
                "concept_id": shot.get("topic_tag", "concept"),
                "shot_id": shot_id,
                "seed": seed,
                "knobs": {"w": w},
                "unet_calls": facet_conf["steps"] * 2,
                "paths": {"image": img_path, "latents": os.path.join(run_dir, "latents", f"{shot_id}.pt")}
            }
            
            if os.path.exists(img_path):
                # We still append the record if image exists so the JSONL has full data
                records.append(rec)
                continue
                
            default_conf["phase4"]["image_gen"]["concept_anchor_ref_weight"] = w
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_t = time.time()
            
            prompt = shot["image_prompt"]
            if facet_conf.get("cartoon_lora_path"):
                prompt = "fca style, " + prompt
                
            generator = torch.Generator(device="cuda").manual_seed(seed)
            ip_img = _prep_reference(ref_image, mode="crop")
            pipe.set_ip_adapter_scale(w)
            
            out = pipe(
                prompt=prompt,
                negative_prompt="text, letters, words, numbers, captions, labels, infographic, diagram, panels, charts, table, watermark, signature, gibberish text",
                width=facet_conf["resolution"][0],
                height=facet_conf["resolution"][1],
                num_inference_steps=facet_conf["steps"],
                guidance_scale=facet_conf["guidance_scale"],
                latents=latents_map[shot_id].clone(),
                ip_adapter_image=ip_img
            )
            
            img = out.images[0]
            img = img.resize((832, 480), Image.LANCZOS)
            img.save(img_path)
            
            gen_time = time.time() - start_t
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            rec["gen_time_s"] = gen_time
            rec["vram_peak_gb"] = peak_vram
            
            records.append(rec)
            
            with open(os.path.join(run_dir, "records.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
                
    return records

if __name__ == "__main__":
    stamp = "G0_A0_geology"
    video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    run_arm(stamp, "A0", video, [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], 0.70)

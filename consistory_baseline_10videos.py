import os
import csv
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
from diffusers import StableDiffusionXLPipeline
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import gc
import sys

# Import ConsiStory components
sys.path.append(os.path.join(os.path.dirname(__file__), "consistory"))
from consistory.consistory_run import load_pipeline, run_anchor_generation, run_extra_generation

# ==========================================================
# Config
# ==========================================================
VIDEOS_CONFIG = [
    ("Heart",     "runs/heart/storyboard.json",                  "all",    "a human heart", "heart", "fullrun_results/data/V4_Heart_adaptive_anchor.csv", 0.257, 0.856),
    ("Eye",       "data/intermediate/V12/phase4/storyboard.json", "all",    "a human eye", "eye", "fullrun_results/data/V12_Eye_adaptive_anchor.csv", 0.284, 0.807),
    ("Sun",       "runs/sun/storyboard.json",                    "all",    "the Sun", "Sun", "fullrun_results/data/V3_Sun_adaptive_anchor.csv", 0.251, 0.799),
    ("Geology",   "runs/geology/storyboard.json",                "anchor", "rocks", "rocks", "fullrun_results/data/V1_Geology_adaptive_anchor.csv", 0.339, 0.790),
    ("Ecology",   "runs/ecology/storyboard.json",                "anchor", "a dripping water cave", "cave", "fullrun_results/data/V2_Ecology_adaptive_anchor.csv", 0.310, 0.828),
    ("Neuron",    "data/intermediate/V10/phase4/storyboard.json", "anchor", "a neuron", "neuron", "fullrun_results/data/V10_Neuron_adaptive_anchor.csv", 0.252, 0.826),
    ("Volcano",   "data/intermediate/V11/phase4/storyboard.json", "anchor", "an erupting volcano", "volcano", "fullrun_results/data/V11_Volcano_adaptive_anchor.csv", 0.283, 0.843),
    ("BlackHole", "data/intermediate/V6/phase4/storyboard.json",  "anchor", "a black hole", "hole", "fullrun_results/data/V6_BlackHole_adaptive_anchor.csv", 0.287, 0.837),
    ("Hurricane", "data/intermediate/V13/phase4/storyboard.json", "anchor", "a hurricane", "hurricane", "replacement_results_2026-06-15/3videos_results/V13_Hurricane_adaptive_anchor.csv", 0.322, 0.801),
    ("Rocket",    "data/intermediate/V15/phase4/storyboard.json", "anchor", "a rocket", "rocket", "15videos_results/V15_Rocket_adaptive_anchor.csv", 0.286, 0.804),
]

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 50
GUIDANCE = 7.0
RES = 1024

def read_anchor(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    concept_text = rows[0][1].replace('"', '').strip()
    shot_ids = [r[0].replace('"', '').strip() for r in rows if len(r) > 0 and r[0].replace('"', '').strip().startswith("shot_")]
    return concept_text, shot_ids

def get_shots(sb_path, selection, anchor_csv, subject):
    with open(sb_path, 'r') as f:
        data = json.load(f)
        shots = data["shots"] if "shots" in data else data
        
    _, anchor_ids = read_anchor(anchor_csv)
    by_id = {s["shot_id"]: s["image_prompt"] for s in shots}
    
    if selection == "all":
        ids = [s["shot_id"] for s in shots]
    else:
        ids = anchor_ids
        
    # As per ConsiStory CLI, prepend the subject to the setting
    ordered = [(sid, f"{subject} {by_id[sid]}", int(sid.split("_")[1]) * 100) for sid in ids]
    return ordered

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_vanilla():
    device = "cuda"
    out_dir = "outputs/consistory_baseline_10videos"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading SDXL {SDXL_ID} for Vanilla Baseline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(SDXL_ID, torch_dtype=torch.float16, use_safetensors=True).to(device)
    pipeline.set_progress_bar_config(disable=True)

    for vid_name, sb_path, selection, subj, token, anchor_csv, dc, dp in VIDEOS_CONFIG:
        print(f"--- [Vanilla] Processing Video: {vid_name} ---")
        ordered = get_shots(sb_path, selection, anchor_csv, subj)
        n_shots = len(ordered)
        
        v_dir = os.path.join(out_dir, vid_name, "vanilla")
        os.makedirs(v_dir, exist_ok=True)

        for i, (sid, prompt, seed) in enumerate(ordered):
            fp = os.path.join(v_dir, f"shot_{i:03d}.png")
            if not os.path.exists(fp):
                torch.cuda.empty_cache()
                set_seed(seed)
                # SDXL native generator
                generator = torch.Generator(device=device).manual_seed(seed)
                img = pipeline(prompt, generator=generator, num_inference_steps=50, guidance_scale=GUIDANCE, height=RES, width=RES).images[0]
                img.save(fp)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

def generate_consistory(target_vid=None):
    out_dir = "outputs/consistory_baseline_10videos"
    
    for vid_name, sb_path, selection, subject, token, anchor_csv, dc, dp in VIDEOS_CONFIG:
        if target_vid is not None and vid_name != target_vid:
            continue
            
        print(f"--- [ConsiStory] Processing Video: {vid_name} ---")
        ordered = get_shots(sb_path, selection, anchor_csv, subject)
        n_shots = len(ordered)
        
        c_dir = os.path.join(out_dir, vid_name, "consistent")
        os.makedirs(c_dir, exist_ok=True)
        
        last_c = os.path.join(c_dir, f"shot_{n_shots-1:03d}.png")
        if os.path.exists(last_c):
            print(f"[{vid_name}] Images already exist, skipping...")
            continue

        print("Loading ConsiStory Pipeline...")
        story_pipeline = load_pipeline(gpu_id=0)
        story_pipeline.enable_vae_slicing()
        story_pipeline.enable_vae_tiling()
            
        prompts = [item[1] for item in ordered]
        seeds = [item[2] for item in ordered]
        concept_tokens = [token]
        
        print(f"Generating anchors for {vid_name}...")
        anchor_prompts = prompts[:2]
        anchor_seeds = seeds[:2]
        extra_prompts = prompts[2:]
        extra_seeds = seeds[2:]

        torch.cuda.empty_cache()
        gc.collect()

        # Generate anchors
        anchor_out_images, _, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(
            story_pipeline, anchor_prompts, concept_tokens, 
            seed=anchor_seeds, mask_dropout=0.5, same_latent=False,
            cache_cpu_offloading=True
        )

        for i, image in enumerate(anchor_out_images):
            image.save(os.path.join(c_dir, f"shot_{i:03d}.png"))
        
        # Generate extra shots
        for i, (extra_prompt, extra_seed) in enumerate(zip(extra_prompts, extra_seeds)):
            print(f"Generating extra shot {i+2}/{n_shots-1} for {vid_name}...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Note: ConsiStory run_extra_generation expects `seed` to be a single value OR list.
            # We pass a list of length 1 for this extra prompt so that latents use this seed.
            # Let's pass a single-element list.
            extra_out_images, _ = run_extra_generation(
                story_pipeline, [extra_prompt], concept_tokens, 
                anchor_cache_first_stage, anchor_cache_second_stage, 
                seed=[extra_seed], mask_dropout=0.5, same_latent=False, 
                cache_cpu_offloading=True
            )
            extra_out_images[0].save(os.path.join(c_dir, f"shot_{i+2:03d}.png"))

        del story_pipeline
        gc.collect()
        torch.cuda.empty_cache()

def evaluate():
    device = "cuda"
    out_dir = "outputs/consistory_baseline_10videos"
    
    print("Loading models for measurement...")
    dinov2 = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    dinov2_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def get_dino(imgs):
        inputs = dinov2_proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = dinov2(**inputs).last_hidden_state[:, 0, :]
        return f / f.norm(p=2, dim=-1, keepdim=True)
        
    def get_clip_img(imgs):
        inputs = clip_proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = clip.get_image_features(**inputs)
        return f / f.norm(p=2, dim=-1, keepdim=True)
        
    def get_clip_txt(txt):
        inputs = clip_proc(text=[txt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            f = clip.get_text_features(**inputs)
        return f / f.norm(p=2, dim=-1, keepdim=True)

    metrics_per_shot = []
    metrics_agg = []
    fair_coords = []
    summary_lines = []
    
    for vid_name, sb_path, selection, subj, token, anchor_csv, dc, dp in VIDEOS_CONFIG:
        concept_text, _ = read_anchor(anchor_csv)
        ordered = get_shots(sb_path, selection, anchor_csv, subj)
        n_shots = len(ordered)
        shot_ids = [sid for sid, _, _ in ordered]
        
        v_dir = os.path.join(out_dir, vid_name, "vanilla")
        c_dir = os.path.join(out_dir, vid_name, "consistent")
        
        v_imgs = [Image.open(os.path.join(v_dir, f"shot_{i:03d}.png")).convert("RGB") for i in range(n_shots)]
        c_imgs = [Image.open(os.path.join(c_dir, f"shot_{i:03d}.png")).convert("RGB") for i in range(n_shots)]
        
        v_feats = get_dino(v_imgs)
        c_feats = get_dino(c_imgs)
        txt_feat = get_clip_txt(concept_text)
        c_clip_feats = get_clip_img(c_imgs)
        v_clip_feats = get_clip_img(v_imgs)
        
        preservation = F.cosine_similarity(c_feats, v_feats, dim=-1).cpu().numpy()
        concept = F.cosine_similarity(c_clip_feats, txt_feat.expand_as(c_clip_feats), dim=-1).cpu().numpy()
        v_concept = F.cosine_similarity(v_clip_feats, txt_feat.expand_as(v_clip_feats), dim=-1).cpu().numpy()
        
        v_inter, c_inter = [], []
        for i in range(n_shots):
            for j in range(i+1, n_shots):
                v_inter.append(F.cosine_similarity(v_feats[i:i+1], v_feats[j:j+1]).item())
                c_inter.append(F.cosine_similarity(c_feats[i:i+1], c_feats[j:j+1]).item())
                
        mean_v_inter = np.mean(v_inter) if v_inter else 0
        mean_c_inter = np.mean(c_inter) if c_inter else 0
        mean_pres = np.mean(preservation)
        mean_conc = np.mean(concept)
        mean_v_conc = np.mean(v_concept)
        
        homogenization = mean_c_inter - mean_v_inter

        for i in range(n_shots):
            metrics_per_shot.append([vid_name, shot_ids[i], f"{concept[i]:.4f}", f"{preservation[i]:.4f}"])
            
        metrics_agg.append([vid_name, f"{mean_v_conc:.4f}", f"{mean_conc:.4f}", f"{mean_pres:.4f}", f"{mean_pres:.4f}", f"{homogenization:.4f}"])
        fair_coords.append([vid_name, f"{mean_conc:.4f}", f"{mean_pres:.4f}"])
        
        summary_lines.append(f"[{vid_name}] Concept={mean_conc:.4f} Content={mean_pres:.4f} Homogenization={homogenization:.4f}")
        
        # Contact sheet
        sheet = Image.new("RGB", (RES*2, RES*n_shots), "white")
        for i in range(n_shots):
            sheet.paste(v_imgs[i], (0, i*RES))
            sheet.paste(c_imgs[i], (RES, i*RES))
        sheet = sheet.resize((1024, 512*n_shots))
        sheet.save(os.path.join(out_dir, f"external_{vid_name}_contactsheet.png"))
        
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/consistory_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Base concept", "ConsiStory concept", "Base content", "ConsiStory content", "Homogenization"])
        writer.writerows(metrics_agg)
        
    with open("outputs/consistory_fairplane.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "ConsiStory concept", "ConsiStory content"])
        writer.writerows(fair_coords)
        
    with open("outputs/consistory_run.log", "w") as f:
        f.write("ConsiStory Baseline - 10 VIDEOS\n")
        f.write("\n".join(summary_lines))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "evaluate":
            evaluate()
        else:
            generate_consistory(sys.argv[1])
    else:
        generate_consistory()
        evaluate()

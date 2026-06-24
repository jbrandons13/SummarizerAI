import os
import json
import subprocess
import yt_dlp
import shutil
import csv
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

videos = [
    {"id": "V6", "name": "BlackHole", "url": "https://www.youtube.com/watch?v=e9TLbZuvsko"},
    {"id": "V7", "name": "Immune", "url": "https://www.youtube.com/watch?v=PSRJfaAYkW4"},
    {"id": "V8", "name": "DNA", "url": "https://www.youtube.com/watch?v=0_b80fHmuWw"},
    {"id": "V9", "name": "Photosynthesis", "url": "https://www.youtube.com/watch?v=Da8XxlCfTuU"},
    {"id": "V10", "name": "Neuron", "url": "https://www.youtube.com/watch?v=uU_4uA6-zcE"},
    {"id": "V11", "name": "Volcano", "url": "https://www.youtube.com/watch?v=LQwZwKS9RPs"}
]

os.makedirs("data/raw_videos", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("fullrun_results/data", exist_ok=True)

device = "cuda"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"ERROR running {cmd}")
        return False
    return True

def download_video(url, out_path):
    if os.path.exists(out_path):
        return True
    print(f"Downloading {url} to {out_path} via CLI...")
    cmd = f"yt-dlp --extractor-args 'youtube:player_client=android' -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' '{url}' -o '{out_path}'"
    res = subprocess.run(cmd, shell=True)
    if res.returncode == 0:
        return True
    else:
        # Fallback to format 18 if bestvideo fails
        cmd = f"yt-dlp --extractor-args 'youtube:player_client=android' -f 18 '{url}' -o '{out_path}'"
import gc

def calc_metrics_and_csv(vid, vname, storyboard, run_dir):
    print("Loading metrics models...")
    dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Read sweeps
    shots = [s["shot_id"] for s in storyboard]
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    # Generate canonical concepts
    concepts = {s["shot_id"]: s["topic_tag"] for s in storyboard}
    
    # Find canonical
    freq = {}
    for shot in storyboard:
        t = shot["topic_tag"]
        freq[t] = freq.get(t, 0) + 1
    max_freq = max(freq.values())
    canonical_topic = [k for k, v in freq.items() if v == max_freq][0]
    
    with torch.no_grad():
        # Load canonical reference (this is w0 image for the canonical shot!)
        ref_img = Image.open(f"{run_dir}/reference.png").convert("RGB")
        ref_feat = dinov2_model(**dinov2_processor(images=ref_img, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]
        ref_feat = ref_feat / ref_feat.norm(p=2, dim=-1, keepdim=True)
        
        metrics_csv = [["shot", "weight", "sim_to_reference", "sim_to_own_w0"]]
        agg_csv = [["weight", "mean_sim_to_reference", "mean_sim_to_own_w0", "mean_inter_shot_sim"]]
        
        adaptive_csv = [["concept_text", f'"{canonical_topic}"'], ["tau", 0.7], [], ["shot", "adaptive_w*", "content_at_w*(sim_to_own)", "concept_at_w*(CLIP)"]]
        
        collapse_data = {w: {"sim_ref": [], "sim_own": [], "inter": [], "clip_t": []} for w in weights}
        shot_daca = {}
        
        for s_idx, sid in enumerate(shots):
            topic_text = concepts[sid]
            text_feat = clip_model.get_text_features(**clip_processor(text=[topic_text], padding=True, return_tensors="pt").to(device))
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            
            # Load w0 image as own reference
            w0_img_path = f"{run_dir}/sweep/{sid}_w0.0.png"
            if not os.path.exists(w0_img_path): continue
            w0_img = Image.open(w0_img_path).convert("RGB")
            w0_feat = dinov2_model(**dinov2_processor(images=w0_img, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]
            w0_feat = w0_feat / w0_feat.norm(p=2, dim=-1, keepdim=True)
            
            best_w = 0.0
            best_content = 1.0
            best_clip = 0.0
            
            for w in weights:
                img_path = f"{run_dir}/sweep/{sid}_w{w}.png"
                if not os.path.exists(img_path): continue
                img = Image.open(img_path).convert("RGB")
                
                feat = dinov2_model(**dinov2_processor(images=img, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                
                clip_img = clip_model.get_image_features(**clip_processor(images=img, return_tensors="pt").to(device))
                clip_img = clip_img / clip_img.norm(p=2, dim=-1, keepdim=True)
                
                sim_ref = F.cosine_similarity(feat, ref_feat, dim=-1).item()
                sim_own = F.cosine_similarity(feat, w0_feat, dim=-1).item()
                clip_t = F.cosine_similarity(clip_img, text_feat, dim=-1).item()
                
                metrics_csv.append([f'"{sid}"', w, f"{sim_ref:.4f}", f"{sim_own:.4f}"])
                
                collapse_data[w]["sim_ref"].append(sim_ref)
                collapse_data[w]["sim_own"].append(sim_own)
                
                if sim_own >= 0.7:
                    best_w = w
                    best_content = sim_own
                    best_clip = clip_t
                    
            adaptive_csv.append([f'"{sid}"', best_w, f"{best_content:.4f}", f"{best_clip:.4f}"])
            shot_daca[sid] = {"w": best_w, "content": best_content, "concept": best_clip}
            
        # Inter-shot sim
        for w in weights:
            # compute all pairs
            feats = []
            for sid in shots:
                img_path = f"{run_dir}/sweep/{sid}_w{w}.png"
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    feat = dinov2_model(**dinov2_processor(images=img, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]
                    feats.append(feat / feat.norm(p=2, dim=-1, keepdim=True))
            if len(feats) > 1:
                feats = torch.cat(feats, dim=0)
                sim_matrix = torch.matmul(feats, feats.T)
                # upper triangle
                idx = torch.triu_indices(len(feats), len(feats), offset=1)
                inter_sim = sim_matrix[idx[0], idx[1]].mean().item()
            else:
                inter_sim = 0.0
            
            m_ref = np.mean(collapse_data[w]["sim_ref"]) if collapse_data[w]["sim_ref"] else 0
            m_own = np.mean(collapse_data[w]["sim_own"]) if collapse_data[w]["sim_own"] else 0
            agg_csv.append([w, f"{m_ref:.4f}", f"{m_own:.4f}", f"{inter_sim:.4f}"])
            
    with open(f"fullrun_results/data/{vid}_{vname}_collapse_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(metrics_csv)
        writer.writerow([])
        writer.writerows(agg_csv)
        
    adaptive_csv.append([])
    adaptive_csv.append(["scheme", "mean_concept(CLIP)", "mean_content(sim_to_own)"])
    adaptive_csv.append(['"adaptive"', f"{np.mean([x['concept'] for x in shot_daca.values()]):.4f}", f"{np.mean([x['content'] for x in shot_daca.values()]):.4f}"])
    with open(f"fullrun_results/data/{vid}_{vname}_adaptive_anchor.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(adaptive_csv)

    # Contact sheet
    try:
        w_h = 1.0
        res_w, res_h = 1024, 1024
        contact_sheet = Image.new("RGB", (res_w * 3, res_h * len(shots)))
        for i, sid in enumerate(shots):
            w0 = Image.open(f"{run_dir}/sweep/{sid}_w0.0.png")
            best_w = shot_daca[sid]["w"]
            w_opt = Image.open(f"{run_dir}/sweep/{sid}_w{best_w}.png")
            w_high = Image.open(f"{run_dir}/sweep/{sid}_w{w_h}.png")
            
            contact_sheet.paste(w0, (0, i * res_h))
            contact_sheet.paste(w_opt, (res_w, i * res_h))
            contact_sheet.paste(w_high, (res_w*2, i * res_h))
        contact_sheet.save(f"fullrun_results/data/{vid}_{vname}_adaptive_grid.png")
    except Exception as e:
        print(f"Error making contact sheet: {e}")
        
    del dinov2_model
    del clip_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return shot_daca, run_dir

def process_video(v):
    vid = v["id"]
    vname = v["name"]
    url = v["url"]
    
    print(f"=== Processing {vid} {vname} ===")
    vid_path = f"data/raw_videos/{vid}.mp4"
    run_dir = f"data/intermediate/{vid}/phase4"
    os.makedirs(run_dir, exist_ok=True)
    
    if not download_video(url, vid_path): return None, None
    
    # Phase 1 & 2
    if not os.path.exists(f"data/intermediate/{vid}/summary_script.json"):
        if not run_cmd(f"PYTHONPATH=. python scripts/run_pipeline.py {vid_path} --phases 1,2"):
            return None, None
            
    # Phase 4 Segmenter
    if not os.path.exists(f"{run_dir}/shots.json"):
        if not run_cmd(f"PYTHONPATH=. python src/phase4/segmenter.py --video-id {vid}"):
            return None, None
            
    # Phase 4 Storyboard
    if not os.path.exists(f"{run_dir}/storyboard.json"):
        if not run_cmd(f"PYTHONPATH=. python src/phase4/storyboard.py --video-id {vid}"):
            return None, None
            
    with open(f"{run_dir}/storyboard.json") as f:
        storyboard = json.load(f)["shots"]
        
    if len(storyboard) == 0:
        print(f"No shots generated for {vid}")
        return None, None
        
    # Find canonical shot
    freq = {}
    for shot in storyboard:
        t = shot["topic_tag"]
        freq[t] = freq.get(t, 0) + 1
    max_freq = max(freq.values())
    canonical_topic = [k for k, v in freq.items() if v == max_freq][0]
    canonical_sid = [s["shot_id"] for s in storyboard if s["topic_tag"] == canonical_topic][0]
    
    # 1. Generate Canonical Reference at w=0
    if not os.path.exists(f"{run_dir}/reference.png"):
        os.makedirs(f"{run_dir}/sweep", exist_ok=True)
        # Dummy reference just to pass arg
        Image.new('RGB', (1024, 1024)).save(f"{run_dir}/dummy.png")
        run_cmd(f"PYTHONPATH=. python weight_sweep.py --config configs/default.yaml --storyboard {run_dir}/storyboard.json --reference {run_dir}/dummy.png --shots {canonical_sid} --weights 0.0 --out {run_dir}/sweep")
        shutil.copy(f"{run_dir}/sweep/{canonical_sid}_w0.0.png", f"{run_dir}/reference.png")
        
    # 2. Run Full Sweep
    all_sids = ",".join([s["shot_id"] for s in storyboard])
    # check if all sweep files exist
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    sweep_complete = True
    for s in storyboard:
        for w in weights:
            if not os.path.exists(f"{run_dir}/sweep/{s['shot_id']}_w{w}.png"):
                sweep_complete = False
                break
    if not sweep_complete:
        w_str = ",".join(map(str, weights))
        run_cmd(f"PYTHONPATH=. python weight_sweep.py --config configs/default.yaml --storyboard {run_dir}/storyboard.json --reference {run_dir}/reference.png --shots {all_sids} --weights {w_str} --out {run_dir}/sweep")
        
    # 3. Compute Metrics
    print("Computing metrics...")
    shot_daca, run_dir = calc_metrics_and_csv(vid, vname, storyboard, run_dir)
    print(f"Finished {vid}")
    return shot_daca, run_dir

# First step: Compute metrics for all 6 videos
video_results = {}
for v in videos:
    shot_daca, run_dir = process_video(v)
    if shot_daca is not None:
        video_results[v["id"]] = {"daca": shot_daca, "dir": run_dir, "vname": v["name"]}

print("=== All metrics computations finished ===")

# Second step: Run full I2V for ALL videos
for v in videos:
    vid = v["id"]
    if vid not in video_results: continue
    print(f"=== Starting I2V generation for {vid} ===")
    vid_path = f"data/raw_videos/{vid}.mp4"
    run_dir = video_results[vid]["dir"]
    vname = video_results[vid]["vname"]
    shot_daca = video_results[vid]["daca"]
    
    # Run Phase 3 (TTS)
    if not os.path.exists(f"data/intermediate/{vid}/audio_manifest.json"):
        run_cmd(f"PYTHONPATH=. python scripts/run_pipeline.py {vid_path} --phases 3")
        
    # Prepare best images directory
    best_img_dir = f"{run_dir}/best_images"
    os.makedirs(best_img_dir, exist_ok=True)
    for sid, daca in shot_daca.items():
        best_w = daca["w"]
        shutil.copy(f"{run_dir}/sweep/{sid}_w{best_w}.png", f"{best_img_dir}/{sid}.png")
        
    # Run Phase 5 I2V rendering
    final_vid_path = f"fullrun_results/data/{vid}_{vname}_final_i2v.mp4"
    if not os.path.exists(final_vid_path):
        i2v_cmd = (
            f"PYTHONPATH=. python render_summary_video.py --all-i2v "
            f"--storyboard {run_dir}/storyboard.json "
            f"--script data/intermediate/{vid}/summary_script.json "
            f"--images-dir {best_img_dir} "
            f"--audio-dir data/intermediate/{vid}/audio "
            f"--work data/intermediate/{vid}/i2v_workspace "
            f"--final {final_vid_path} "
            f"--workflow scripts/wan_i2v_workflow.json"
        )
        run_cmd(i2v_cmd)
        
print("=== All requested processing finished! ===")

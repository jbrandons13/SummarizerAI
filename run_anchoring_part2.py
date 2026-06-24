import os
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import subprocess

device = "cuda"

videos = [
    {"id": "V12", "name": "Eye", "concept": "a colorful cartoon illustration of a human eye, the organ of sight"},
    {"id": "V13", "name": "Hurricane", "concept": "a colorful cartoon illustration of a hurricane, a large swirling storm system"},
    {"id": "V14", "name": "Reef", "concept": "a colorful cartoon illustration of a coral reef, an underwater ecosystem of corals and fish"}
]

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"ERROR running {cmd}")
        return False
    return True

# 1. Run Sweeps
for v in videos:
    vid = v["id"]
    run_dir = f"data/intermediate/{vid}/phase4"
    
    with open(f"{run_dir}/storyboard.json") as f:
        storyboard = json.load(f)["shots"]
        
    sids = ",".join([s["shot_id"] for s in storyboard])
    w_str = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0"
    
    print(f"\n--- SWEEPING {vid} ---")
    run_cmd(f"PYTHONPATH=. python weight_sweep.py --config configs/default.yaml --storyboard {run_dir}/storyboard.json --reference gate_check/{vid}_{v['name']}_reference.png --shots {sids} --weights {w_str} --out {run_dir}/sweep")

# 2. Compute Metrics
print("\n--- COMPUTING METRICS ---")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

for v in videos:
    vid = v["id"]
    vname = v["name"]
    concept_text = v["concept"]
    run_dir = f"data/intermediate/{vid}/phase4"
    
    with open(f"{run_dir}/storyboard.json") as f:
        storyboard = json.load(f)["shots"]
    shots = [s["shot_id"] for s in storyboard]
    
    with torch.no_grad():
        ref_img = Image.open(f"gate_check/{vid}_{vname}_reference.png").convert("RGB")
        ref_feat = dinov2_model(**dinov2_processor(images=ref_img, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]
        ref_feat = ref_feat / ref_feat.norm(p=2, dim=-1, keepdim=True)
        
        text_feat = clip_model.get_text_features(**clip_processor(text=[concept_text], padding=True, return_tensors="pt").to(device))
        text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
        
        metrics_csv = [["shot", "weight", "sim_to_reference", "sim_to_own_w0"]]
        agg_csv = [["weight", "mean_sim_to_reference", "mean_sim_to_own_w0", "mean_inter_shot_sim"]]
        adaptive_csv = [["concept_text", f'"{concept_text}"'], ["tau", 0.7], [], ["shot", "adaptive_w*", "content_at_w*(sim_to_own)", "concept_at_w*(CLIP)"]]
        
        collapse_data = {w: {"sim_ref": [], "sim_own": [], "inter": [], "clip_t": []} for w in weights}
        shot_daca = {}
        
        for sid in shots:
            w0_path = f"{run_dir}/sweep/{sid}_w0.0.png"
            if not os.path.exists(w0_path): continue
            w0_img = Image.open(w0_path).convert("RGB")
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
                collapse_data[w]["clip_t"].append(clip_t)
                
                if sim_own >= 0.7:
                    best_w = w
                    best_content = sim_own
                    best_clip = clip_t
                    
            adaptive_csv.append([f'"{sid}"', best_w, f"{best_content:.4f}", f"{best_clip:.4f}"])
            shot_daca[sid] = {"w": best_w, "content": best_content, "concept": best_clip}
            
        for w in weights:
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
                idx = torch.triu_indices(len(feats), len(feats), offset=1)
                inter_sim = sim_matrix[idx[0], idx[1]].mean().item()
            else:
                inter_sim = 0.0
                
            m_ref = np.mean(collapse_data[w]["sim_ref"]) if collapse_data[w]["sim_ref"] else 0
            m_own = np.mean(collapse_data[w]["sim_own"]) if collapse_data[w]["sim_own"] else 0
            agg_csv.append([w, f"{m_ref:.4f}", f"{m_own:.4f}", f"{inter_sim:.4f}"])
            
    # Write output
    os.makedirs("3videos_results", exist_ok=True)
    with open(f"3videos_results/V{vid[1:]}_{vname}_collapse_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(metrics_csv)
        writer.writerow([])
        writer.writerows(agg_csv)
        
    adaptive_csv.append([])
    adaptive_csv.append(["scheme", "mean_concept(CLIP)", "mean_content(sim_to_own)"])
    adaptive_csv.append(['"adaptive"', f"{np.mean([x['concept'] for x in shot_daca.values()]):.4f}", f"{np.mean([x['content'] for x in shot_daca.values()]):.4f}"])
    with open(f"3videos_results/V{vid[1:]}_{vname}_adaptive_anchor.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(adaptive_csv)
        
    # Generate Contact Sheet
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
    contact_sheet.save(f"3videos_results/V{vid[1:]}_{vname}_adaptive_grid.png")
    
    # Save to global summary
    with open("3videos_results/concept_remeasure_3videos.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Video", "Concept Text", "Mean CLIP w=0.2", "Mean CLIP w=0.4", "Mean CLIP w=0.6", "Mean CLIP w*"])
        writer.writerow([
            vname, concept_text,
            f"{np.mean(collapse_data[0.2]['clip_t']):.4f}",
            f"{np.mean(collapse_data[0.4]['clip_t']):.4f}",
            f"{np.mean(collapse_data[0.6]['clip_t']):.4f}",
            f"{np.mean([x['concept'] for x in shot_daca.values()]):.4f}"
        ])

print("\n--- METRICS COMPLETE ---")
del dinov2_model
del clip_model
torch.cuda.empty_cache()

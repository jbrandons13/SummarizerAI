import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

videos = [
    {"id": "V6", "name": "BlackHole", "concept": "a colorful cartoon illustration of a black hole, a dark region in space with strong gravity"},
    {"id": "V7", "name": "Immune", "concept": "a colorful cartoon illustration of the immune system, white blood cells defending the body"},
    {"id": "V8", "name": "DNA", "concept": "a colorful cartoon illustration of a DNA double helix molecule"},
    {"id": "V9", "name": "Photosynthesis", "concept": "a colorful cartoon illustration of photosynthesis, a green leaf turning sunlight into energy"},
    {"id": "V10", "name": "Neuron", "concept": "a colorful cartoon illustration of a neuron, a nerve cell that transmits signals"},
    {"id": "V11", "name": "Volcano", "concept": "a colorful cartoon illustration of an erupting volcano"}
]

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

summary_csv = [["Video", "Concept Text", "Mean CLIP w=0.2", "Mean CLIP w=0.4", "Mean CLIP w=0.6", "Mean CLIP w*"]]

for v in videos:
    vid = v["id"]
    vname = v["name"]
    concept_text = v["concept"]
    print(f"Processing {vid} {vname}...")
    
    # Text feature
    with torch.no_grad():
        text_inputs = clip_processor(text=[concept_text], padding=True, return_tensors="pt").to(device)
        text_feat = clip_model.get_text_features(**text_inputs)
        text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
    
    run_dir = f"data/intermediate/{vid}/phase4"
    adaptive_path = f"fullrun_results/data/{vid}_{vname}_adaptive_anchor.csv"
    remeasure_path = f"fullrun_results/data/{vid}_{vname}_clip_remeasure.csv"
    
    with open(adaptive_path, "r") as f:
        adaptive_rows = list(csv.reader(f))
    
    shots_wstar = {}
    header_idx = None
    for i, row in enumerate(adaptive_rows):
        if len(row) > 0 and row[0] == "shot":
            header_idx = i
            break
            
    shot_start_idx = header_idx + 1
    shot_end_idx = None
    for i in range(shot_start_idx, len(adaptive_rows)):
        if len(adaptive_rows[i]) == 0 or adaptive_rows[i][0].replace('"', '') == "scheme":
            shot_end_idx = i
            break
            
    if shot_end_idx is None: shot_end_idx = len(adaptive_rows)
    
    for i in range(shot_start_idx, shot_end_idx):
        row = adaptive_rows[i]
        if len(row) < 2: continue
        sid = row[0].replace('"', '')
        w_star = float(row[1])
        shots_wstar[sid] = w_star
        
    sids = list(shots_wstar.keys())
    
    remeasure_csv = [["shot", "weight", "concept_clip_t"]]
    
    w_scores = {w: [] for w in weights}
    wstar_scores = []
    
    with torch.no_grad():
        for sid in sids:
            for w in weights:
                img_path = f"{run_dir}/sweep/{sid}_w{w}.png"
                if not os.path.exists(img_path): continue
                img = Image.open(img_path).convert("RGB")
                img_inputs = clip_processor(images=img, return_tensors="pt").to(device)
                clip_img = clip_model.get_image_features(**img_inputs)
                clip_img = clip_img / clip_img.norm(p=2, dim=-1, keepdim=True)
                
                clip_t = F.cosine_similarity(clip_img, text_feat, dim=-1).item()
                remeasure_csv.append([f'"{sid}"', w, f"{clip_t:.4f}"])
                
                w_scores[w].append(clip_t)
                if abs(w - shots_wstar[sid]) < 0.001:
                    wstar_scores.append(clip_t)
                    for i in range(shot_start_idx, shot_end_idx):
                        if adaptive_rows[i][0].replace('"', '') == sid:
                            adaptive_rows[i][3] = f"{clip_t:.4f}"
                            break
    
    for i in range(shot_end_idx, len(adaptive_rows)):
        if len(adaptive_rows[i]) > 0 and adaptive_rows[i][0].replace('"', '') == "adaptive":
            adaptive_rows[i][1] = f"{np.mean(wstar_scores):.4f}"
            break
            
    adaptive_rows[0][1] = f'"{concept_text}"'
    
    with open(remeasure_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(remeasure_csv)
        
    with open(adaptive_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(adaptive_rows)
        
    m02 = np.mean(w_scores[0.2]) if w_scores[0.2] else 0
    m04 = np.mean(w_scores[0.4]) if w_scores[0.4] else 0
    m06 = np.mean(w_scores[0.6]) if w_scores[0.6] else 0
    mws = np.mean(wstar_scores) if wstar_scores else 0
    summary_csv.append([vid, concept_text, f"{m02:.4f}", f"{m04:.4f}", f"{m06:.4f}", f"{mws:.4f}"])
    print(f"  -> w* mean clip-t: {mws:.4f}")

with open("fullrun_results/data/n10_concept_remeasure_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(summary_csv)

print("Finished remeasurement.")

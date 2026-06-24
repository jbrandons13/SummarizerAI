import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

device = "cuda"
vid = "V15"
vname = "Rocket"
concept_text = "a colorful cartoon illustration of a rocket launching into space"

# The sweep directory still contains the Rocket images for w > 0 because River skipped them!
sweep_dir = f"data/intermediate/{vid}/phase4/sweep"

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

weights = [0.2, 0.4, 0.6]

with torch.no_grad():
    text_feat = clip_model.get_text_features(**clip_processor(text=[concept_text], padding=True, return_tensors="pt").to(device))
    text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
    
    means = {}
    for w in weights:
        clip_scores = []
        for i in range(1, 17):
            sid = f"shot_{i:03d}"
            img_path = os.path.join(sweep_dir, f"{sid}_w{w}.png")
            if not os.path.exists(img_path):
                print(f"ERROR: {img_path} not found!")
                continue
                
            img = Image.open(img_path).convert("RGB")
            img_feat = clip_model.get_image_features(**clip_processor(images=img, return_tensors="pt").to(device))
            img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
            
            sim = F.cosine_similarity(img_feat, text_feat, dim=-1).item()
            clip_scores.append(sim)
            
        means[w] = np.mean(clip_scores) if clip_scores else 0.0

# Read w* from V15_Rocket_adaptive_anchor.csv to sanity-check and report
w_star_mean = 0.0
with open("15videos_results/V15_Rocket_adaptive_anchor.csv", "r") as f:
    lines = list(csv.reader(f))
    last_line = lines[-1]
    if "adaptive" in last_line[0]:
        w_star_mean = float(last_line[1])

print(f"Mean CLIP at w=0.2: {means[0.2]:.4f}")
print(f"Mean CLIP at w=0.4: {means[0.4]:.4f}")
print(f"Mean CLIP at w=0.6: {means[0.6]:.4f}")
print(f"Mean CLIP at w*: {w_star_mean:.4f}")

out_csv = [
    ["Video", "Concept Text", "Mean CLIP w=0.2", "Mean CLIP w=0.4", "Mean CLIP w=0.6", "Mean CLIP w*"],
    [vname, f'"{concept_text}"', f"{means[0.2]:.4f}", f"{means[0.4]:.4f}", f"{means[0.6]:.4f}", f"{w_star_mean:.4f}"]
]

with open("concept_remeasure_rocket.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(out_csv)

print("Saved to concept_remeasure_rocket.csv")

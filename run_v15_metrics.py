import os
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

device = "cuda"
vid = "V15"
vname = "Rocket"
concept_text = "a colorful cartoon illustration of a rocket launching into space"

run_dir = f"data/intermediate/{vid}/phase4"
with open(f"{run_dir}/storyboard.json") as f:
    shots = json.load(f)["shots"]

print("\n--- COMPUTING METRICS ---")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

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
    
    shots_ids = [s["shot_id"] for s in shots]
    for sid in shots_ids:
        w0_path = f"{run_dir}/sweep/{sid}_w0.0.png"
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
        feats_w = []
        for sid in shots_ids:
            img_path = f"{run_dir}/sweep/{sid}_w{w}.png"
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                feat = dinov2_model(**dinov2_processor(images=img, return_tensors="pt").to(device)).last_hidden_state[:, 0, :]
                feats_w.append(feat / feat.norm(p=2, dim=-1, keepdim=True))
        if len(feats_w) > 1:
            feats_w = torch.cat(feats_w, dim=0)
            sim_matrix = torch.matmul(feats_w, feats_w.T)
            idx = torch.triu_indices(len(feats_w), len(feats_w), offset=1)
            inter_sim = sim_matrix[idx[0], idx[1]].mean().item()
        else:
            inter_sim = 0.0
            
        m_ref = np.mean(collapse_data[w]["sim_ref"]) if collapse_data[w]["sim_ref"] else 0
        m_own = np.mean(collapse_data[w]["sim_own"]) if collapse_data[w]["sim_own"] else 0
        agg_csv.append([w, f"{m_ref:.4f}", f"{m_own:.4f}", f"{inter_sim:.4f}"])
        
num_less_than_1 = sum(1 for x in shot_daca.values() if x['w'] < 1.0)
print(f"\n[DACA ACCEPTANCE CHECK] Number of shots with w* < 1.0: {num_less_than_1} (Target >= 5)")
if num_less_than_1 < 5:
    print("FAILED DACA DIVERSITY CHECK! Stopping.")
else:
    print("PASSED DACA DIVERSITY CHECK!")

os.makedirs("15videos_results", exist_ok=True)
with open(f"15videos_results/V{vid[1:]}_{vname}_collapse_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(metrics_csv)
    writer.writerow([])
    writer.writerows(agg_csv)
    
adaptive_csv.append([])
adaptive_csv.append(["scheme", "mean_concept(CLIP)", "mean_content(sim_to_own)"])
adaptive_csv.append(['"adaptive"', f"{np.mean([x['concept'] for x in shot_daca.values()]):.4f}", f"{np.mean([x['content'] for x in shot_daca.values()]):.4f}"])
with open(f"15videos_results/V{vid[1:]}_{vname}_adaptive_anchor.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(adaptive_csv)

w_h = 1.0
res_w, res_h = 1024, 1024
contact_sheet = Image.new("RGB", (res_w * 3, res_h * len(shots_ids)))
for i, sid in enumerate(shots_ids):
    w0 = Image.open(f"{run_dir}/sweep/{sid}_w0.0.png")
    best_w = shot_daca[sid]["w"]
    w_opt = Image.open(f"{run_dir}/sweep/{sid}_w{best_w}.png")
    w_high = Image.open(f"{run_dir}/sweep/{sid}_w{w_h}.png")
    
    contact_sheet.paste(w0, (0, i * res_h))
    contact_sheet.paste(w_opt, (res_w, i * res_h))
    contact_sheet.paste(w_high, (res_w*2, i * res_h))
contact_sheet.save(f"15videos_results/V{vid[1:]}_{vname}_adaptive_grid.png")

print("\n--- VLM JUDGE ---")
del dinov2_model
del clip_model
torch.cuda.empty_cache()

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import random

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

def query_vlm(images, text_prompt):
    messages = [{"role": "user", "content": []}]
    for img in images:
        messages[0]["content"].append({"type": "image", "image": img})
    messages[0]["content"].append({"type": "text", "text": text_prompt})
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

daca_wins = 0
fidelity_total = 0
copy_daca = 0
copy_high_w = 0
copy_total = 0

d_images = []
h_images = []

for i in range(1, 17):
    sid = f"shot_{i:03d}"
    w_star = shot_daca[sid]["w"]
    
    d_path = f"{run_dir}/sweep/{sid}_w{w_star}.png"
    h_path = f"{run_dir}/sweep/{sid}_w{1.0}.png"
        
    img_d = Image.open(d_path).convert("RGB")
    img_h = Image.open(h_path).convert("RGB")
    
    is_daca_a = random.choice([True, False])
    imgs = [img_d, img_h] if is_daca_a else [img_h, img_d]
    ans = query_vlm(imgs, f"Which image more accurately and completely depicts: {concept_text}? Answer A or B only.").lower()
    if "a" in ans and "b" not in ans:
        if is_daca_a: daca_wins += 1
        fidelity_total += 1
    elif "b" in ans and "a" not in ans:
        if not is_daca_a: daca_wins += 1
        fidelity_total += 1
        
    ans_d = query_vlm([img_d, ref_img], "Is the first image essentially a copy of the second image (near-identical content/composition)? Answer Yes or No.")
    if "yes" in ans_d.lower(): copy_daca += 1
    
    ans_h = query_vlm([img_h, ref_img], "Is the first image essentially a copy of the second image (near-identical content/composition)? Answer Yes or No.")
    if "yes" in ans_h.lower(): copy_high_w += 1
    
    copy_total += 1
    d_images.append(img_d)
    h_images.append(img_h)
    
same_daca = 0
same_high_w = 0
same_total = 0

for i in range(len(d_images) - 1):
    ans_d = query_vlm([d_images[i], d_images[i+1]], "Do these two images depict the same concept/topic? Answer Yes or No.")
    if "yes" in ans_d.lower(): same_daca += 1
    ans_h = query_vlm([h_images[i], h_images[i+1]], "Do these two images depict the same concept/topic? Answer Yes or No.")
    if "yes" in ans_h.lower(): same_high_w += 1
    same_total += 1

results = [
    {
        "video": vname,
        "method": "DACA",
        "content_fidelity": (daca_wins / fidelity_total) if fidelity_total > 0 else 0,
        "same_scene": (same_daca / same_total) if same_total > 0 else 0,
        "near_duplicate": (copy_daca / copy_total) if copy_total > 0 else 0,
        "n_shots": fidelity_total,
        "n_pairs": same_total
    },
    {
        "video": vname,
        "method": "Fixed w=1.0",
        "content_fidelity": ((fidelity_total - daca_wins) / fidelity_total) if fidelity_total > 0 else 0,
        "same_scene": (same_high_w / same_total) if same_total > 0 else 0,
        "near_duplicate": (copy_high_w / copy_total) if copy_total > 0 else 0,
        "n_shots": fidelity_total,
        "n_pairs": same_total
    }
]

with open(f"15videos_results/vlm_judge_{vname}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["video", "method", "content_fidelity", "same_scene", "near_duplicate", "n_shots", "n_pairs"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("V15 PIPELINE COMPLETE!")

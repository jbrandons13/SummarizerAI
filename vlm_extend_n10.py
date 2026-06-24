import os
import json
import random
import csv
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"Loading VLM: {model_id}")
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

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
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()

videos = {
    "V1": "Geology",
    "V2": "Ecology",
    "V6": "BlackHole",
    "V7": "Immune",
    "V8": "DNA",
    "V9": "Photosynthesis",
    "V10": "Neuron",
    "V11": "Volcano"
}

results = []

for vid, vname in videos.items():
    print(f"Processing {vid} {vname}...")
    storyboard_path = f"data/intermediate/{vid}/phase4/storyboard.json"
    daca_csv = f"fullrun_results/data/{vid}_{vname}_adaptive_anchor.csv"
    
    if not os.path.exists(storyboard_path):
        print(f"Missing storyboard for {vid}")
        continue
        
    with open(storyboard_path) as f:
        sb = json.load(f)
        storyboard = sb["shots"] if "shots" in sb else sb
        
    daca_picks = {}
    with open(daca_csv) as f:
        lines = f.readlines()
        in_shots = False
        for line in lines:
            if line.startswith("shot,adaptive_w*"):
                in_shots = True; continue
            if in_shots and line.strip() == "": break
            if in_shots:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    daca_picks[parts[0].strip('"')] = float(parts[1])
                    
    high_w = 1.0
    
    daca_wins = 0
    fidelity_total = 0
    copy_daca = 0
    copy_high_w = 0
    copy_total = 0
    
    # We will just pair shot i and shot i+1
    d_images = []
    h_images = []
    
    for shot in storyboard:
        sid = shot.get("shot_id") or shot.get("id")
        caption = shot.get("visual_description", "") or shot.get("image_prompt", "")
        
        w_star = daca_picks.get(sid, 0.2)
        
        # Paths
        d_path = f"data/intermediate/{vid}/phase4/sweep/{sid}_w{w_star}.png"
        h_path = f"data/intermediate/{vid}/phase4/sweep/{sid}_w{high_w}.png"
        
        if not os.path.exists(d_path) or not os.path.exists(h_path):
            continue
            
        img_d = Image.open(d_path).convert("RGB")
        img_h = Image.open(h_path).convert("RGB")
        
        # Reference image is w=0.0
        ref_path = f"data/intermediate/{vid}/phase4/sweep/{sid}_w0.0.png"
        ref_img = Image.open(ref_path).convert("RGB") if os.path.exists(ref_path) else None
        
        # 1. CONTENT FIDELITY (A/B)
        is_daca_a = random.choice([True, False])
        imgs = [img_d, img_h] if is_daca_a else [img_h, img_d]
        prompt = f"Which image more accurately and completely depicts: {caption}? Answer A or B only."
        ans = query_vlm(imgs, prompt).lower()
        if "a" in ans and "b" not in ans:
            if is_daca_a: daca_wins += 1
            fidelity_total += 1
        elif "b" in ans and "a" not in ans:
            if not is_daca_a: daca_wins += 1
            fidelity_total += 1
            
        # 2. COPY FLAG (near duplicate of reference)
        if ref_img:
            ans_d = query_vlm([img_d, ref_img], "Is the first image essentially a copy of the second image (near-identical content/composition)? Answer Yes or No.")
            if "yes" in ans_d.lower(): copy_daca += 1
            
            ans_h = query_vlm([img_h, ref_img], "Is the first image essentially a copy of the second image (near-identical content/composition)? Answer Yes or No.")
            if "yes" in ans_h.lower(): copy_high_w += 1
            
            copy_total += 1
            
        d_images.append(img_d)
        h_images.append(img_h)
        
    # 3. SAME-CONCEPT RATE (Same-scene across consecutive shots)
    same_daca = 0
    same_high_w = 0
    same_total = 0
    
    for i in range(len(d_images) - 1):
        ans_d = query_vlm([d_images[i], d_images[i+1]], "Do these two images depict the same concept/topic? Answer Yes or No.")
        if "yes" in ans_d.lower(): same_daca += 1
        
        ans_h = query_vlm([h_images[i], h_images[i+1]], "Do these two images depict the same concept/topic? Answer Yes or No.")
        if "yes" in ans_h.lower(): same_high_w += 1
        
        same_total += 1
        
    results.append({
        "video": vname,
        "method": "DACA",
        "content_fidelity": (daca_wins / fidelity_total) if fidelity_total > 0 else 0,
        "same_scene": (same_daca / same_total) if same_total > 0 else 0,
        "near_duplicate": (copy_daca / copy_total) if copy_total > 0 else 0,
        "n_shots": fidelity_total,
        "n_pairs": same_total
    })
    results.append({
        "video": vname,
        "method": "Fixed w=1.0",
        "content_fidelity": ((fidelity_total - daca_wins) / fidelity_total) if fidelity_total > 0 else 0,
        "same_scene": (same_high_w / same_total) if same_total > 0 else 0,
        "near_duplicate": (copy_high_w / copy_total) if copy_total > 0 else 0,
        "n_shots": fidelity_total,
        "n_pairs": same_total
    })
    print(f"  DACA fidelity={results[-2]['content_fidelity']:.2f}, same_scene={results[-2]['same_scene']:.2f}")
    print(f"  Fixed w=1.0 same_scene={results[-1]['same_scene']:.2f}")
    
with open("vlm_judge_n10.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["video", "method", "content_fidelity", "same_scene", "near_duplicate", "n_shots", "n_pairs"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)
        
print("VLM evaluation complete.")

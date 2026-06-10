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

def run_task_a():
    os.makedirs("addon_results", exist_ok=True)
    videos = {"geology": "V1", "ecology": "V2", "sun": "V3", "heart": "V4", "iphone": "V5"}
    
    results = []
    
    # Write question templates
    with open("addon_results/vlm_questions.txt", "w") as f:
        f.write("CONTENT FIDELITY (A/B):\nWhich image more accurately and completely depicts: [caption]? Answer A or B only.\n")
        f.write("Counterbalancing applied: Image A and B randomly swapped between DACA and high-w, recorded properly.\n\n")
        f.write("SAME-CONCEPT RATE:\nDo these two images depict the same concept/topic? Yes or No.\n\n")
        f.write("COPY FLAG:\nIs this image essentially a copy of this reference (near-identical content/composition)? Yes or No.\n")
        
    with open("addon_results/vlm_model.txt", "w") as f:
        f.write(f"VLM: {model_id}\n")

    for vid, v_label in videos.items():
        print(f"Processing {v_label} ({vid})...")
        manifest_path = f"runs/{vid}/sweep/manifest.json"
        storyboard_path = f"runs/{vid}/storyboard.json"
        ref_path = f"runs/{vid}/reference.png"
        
        if not os.path.exists(manifest_path) or not os.path.exists(storyboard_path):
            print(f"  Missing data for {vid}, skipping.")
            continue
            
        with open(storyboard_path) as f:
            storyboard = json.load(f)["shots"]
            
        with open(manifest_path) as f:
            manifest = json.load(f)
            
        ref_img = Image.open(ref_path).convert("RGB") if os.path.exists(ref_path) else None
        
        # Collect paths for DACA and high-w
        # For high-w, we just pick weight == 1.0. For V1/V2, sweep might be structured differently
        # Let's read daca/adaptive_anchor.csv to get w*
        daca_picks = {}
        daca_csv = f"runs/{vid}/daca/adaptive_anchor.csv"
        # Since V1/V2 daca might be in data/intermediate, let's try to find it
        if not os.path.exists(daca_csv):
            # fallback for V1/V2
            if vid == "geology":
                daca_csv = "data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4/collapse_evidence_fine/adaptive_anchor.csv"
            elif vid == "ecology":
                daca_csv = "data/intermediate/2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology/phase4/collapse_evidence/adaptive_anchor.csv"
        
        if os.path.exists(daca_csv):
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
        
        concept_groups = {} # topic_tag -> [(daca_img, high_w_img), ...]
        
        for shot in storyboard:
            sid = shot["shot_id"]
            caption = shot.get("visual_description", "")
            topic = shot.get("topic_tag", "")
            
            w_star = daca_picks.get(sid, 0.2)
            
            # Find paths
            d_path = None
            h_path = None
            if isinstance(manifest, list):
                for item in manifest:
                    if item["shot_id"] == sid and abs(item["weight"] - w_star) < 0.01:
                        d_path = item["path"]
                    if item["shot_id"] == sid and abs(item["weight"] - high_w) < 0.01:
                        h_path = item["path"]
            else:
                for item in manifest.get(sid, []):
                    if abs(item["weight"] - w_star) < 0.01:
                        d_path = item["image_path"]
                    if abs(item["weight"] - high_w) < 0.01:
                        h_path = item["image_path"]
                        
            if not d_path or not os.path.exists(d_path) or not h_path or not os.path.exists(h_path):
                continue
                
            img_d = Image.open(d_path).convert("RGB")
            img_h = Image.open(h_path).convert("RGB")
            
            # 1. CONTENT FIDELITY (A/B)
            # Counterbalance
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
                
            # 2. COPY FLAG
            if ref_img:
                ans_d = query_vlm([img_d, ref_img], "Is the first image essentially a copy of the second image (near-identical content/composition)? Answer Yes or No.")
                if "yes" in ans_d.lower(): copy_daca += 1
                
                ans_h = query_vlm([img_h, ref_img], "Is the first image essentially a copy of the second image (near-identical content/composition)? Answer Yes or No.")
                if "yes" in ans_h.lower(): copy_high_w += 1
                
                copy_total += 1
                
            # Group for concept
            if topic not in concept_groups:
                concept_groups[topic] = []
            concept_groups[topic].append((img_d, img_h))
            
        # 3. SAME-CONCEPT RATE
        same_daca = 0
        same_high_w = 0
        same_total = 0
        for topic, pairs in concept_groups.items():
            for i in range(len(pairs)-1):
                ans_d = query_vlm([pairs[i][0], pairs[i+1][0]], "Do these two images depict the same concept/topic? Answer Yes or No.")
                if "yes" in ans_d.lower(): same_daca += 1
                ans_h = query_vlm([pairs[i][1], pairs[i+1][1]], "Do these two images depict the same concept/topic? Answer Yes or No.")
                if "yes" in ans_h.lower(): same_high_w += 1
                same_total += 1
                
        results.append({
            "video": v_label,
            "method": "DACA",
            "content_fidelity_winrate": (daca_wins / fidelity_total) if fidelity_total > 0 else 0,
            "same_concept_rate": (same_daca / same_total) if same_total > 0 else 0,
            "copy_rate": (copy_daca / copy_total) if copy_total > 0 else 0,
            "n_shots": fidelity_total
        })
        results.append({
            "video": v_label,
            "method": "high-w",
            "content_fidelity_winrate": ((fidelity_total - daca_wins) / fidelity_total) if fidelity_total > 0 else 0,
            "same_concept_rate": (same_high_w / same_total) if same_total > 0 else 0,
            "copy_rate": (copy_high_w / copy_total) if copy_total > 0 else 0,
            "n_shots": fidelity_total
        })
        
    with open("addon_results/vlm_results_aggregate.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "method", "content_fidelity_winrate", "same_concept_rate", "copy_rate", "n_shots"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
if __name__ == "__main__":
    run_task_a()

import os
import shutil
import csv

OUT_DIR = "comparison"
os.makedirs(OUT_DIR, exist_ok=True)

videos = {
    "heart": {
        "vid_id": "V4",
        "vid_name": "Heart",
        "shots": [
            ("shot_010", "shot_009.png"), 
            ("shot_011", "shot_010.png"), 
            ("shot_012", "shot_011.png")  
        ]
    },
    "sun": {
        "vid_id": "V3",
        "vid_name": "Sun",
        "shots": [
            ("shot_009", "shot_008.png"), 
            ("shot_010", "shot_009.png"), 
            ("shot_011", "shot_010.png")  
        ]
    }
}

daca_weights = {}
for v_key, v_info in videos.items():
    csv_path = f"fullrun_results/data/{v_info['vid_id']}_{v_info['vid_name']}_adaptive_anchor.csv"
    daca_weights[v_key] = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1 and row[0].replace('"', '').startswith('shot_'):
                sid = row[0].replace('"', '')
                w = float(row[1])
                if w.is_integer():
                    w_str = f"{int(w)}.0"
                else:
                    w_str = str(w)
                daca_weights[v_key][sid] = w_str

for v_key, v_info in videos.items():
    v_name = v_info["vid_name"]
    
    ref_src = f"runs/{v_key}/reference.png"
    ref_dst = f"{OUT_DIR}/{v_key}_reference.png"
    if os.path.exists(ref_src):
        shutil.copy(ref_src, ref_dst)
    else:
        print(f"Missing {ref_src}")
        
    for n_idx, (sid, ext_name) in enumerate(v_info["shots"]):
        N = n_idx + 1
        
        tasks = [
            (f"runs/{v_key}/sweep/{sid}_w0.0.png", f"{OUT_DIR}/{v_key}_ipa_w0.0_shot{N}.png"),
            (f"runs/{v_key}/sweep/{sid}_w0.5.png", f"{OUT_DIR}/{v_key}_ipa_w0.5_shot{N}.png"),
            (f"runs/{v_key}/sweep/{sid}_w1.0.png", f"{OUT_DIR}/{v_key}_ipa_w1.0_shot{N}.png"),
            (f"runs/{v_key}/sweep/{sid}_w{daca_weights[v_key][sid]}.png", f"{OUT_DIR}/{v_key}_daca_shot{N}.png"),
            (f"outputs/consistory_baseline_10videos/{v_name}/consistent/{ext_name}", f"{OUT_DIR}/{v_key}_consistory_shot{N}.png"),
            (f"runs/A3_storydiffusion/{v_name}/consistent/{ext_name}", f"{OUT_DIR}/{v_key}_storydiffusion_shot{N}.png")
        ]
        
        for src, dst in tasks:
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Missing {src}")

print("Retrieval completed successfully.")

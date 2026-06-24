import os
import matplotlib.pyplot as plt
from PIL import Image

videos = [
    ("V7", "Immune", ["shot_003", "shot_006", "shot_010"]),
    ("V8", "DNA", ["shot_002", "shot_007", "shot_012"]),
    ("V9", "Photosynthesis", ["shot_004", "shot_008", "shot_014"])
]

fig, axes = plt.subplots(3, 4, figsize=(16, 9), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

for i, (vid, name, shots) in enumerate(videos):
    # Reference
    ref_path = f"data/intermediate/{vid}/phase4/sweep/shot_001_w0.0.png"
    if not os.path.exists(ref_path):
        ref_path = f"data/intermediate/{vid}/phase4/sweep/shot_002_w0.0.png"
    if not os.path.exists(ref_path):
        ref_path = f"data/intermediate/{vid}/phase4/sweep/shot_003_w0.0.png"
        
    ax = axes[i, 0]
    if os.path.exists(ref_path):
        ax.imshow(Image.open(ref_path))
    if i == 0:
        ax.set_title("Reference Concept", fontsize=14, fontweight="bold")
    ax.set_ylabel(name, fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Get w* for this video
    daca_picks = {}
    daca_csv = f"fullrun_results/data/{vid}_{name}_adaptive_anchor.csv"
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
                    
    for j, sid in enumerate(shots):
        ax = axes[i, j+1]
        w_star = daca_picks.get(sid, 0.2)
        img_path = f"data/intermediate/{vid}/phase4/sweep/{sid}_w{w_star}.png"
        if os.path.exists(img_path):
            ax.imshow(Image.open(img_path))
        if i == 0:
            ax.set_title(f"DACA Shot {j+1}", fontsize=14, fontweight="bold")
        ax.axis("off")

plt.savefig("biology_visual_audit.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved biology_visual_audit.png")

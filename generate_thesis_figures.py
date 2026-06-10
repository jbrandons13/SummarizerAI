import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_collapse_data(csv_path):
    weights = []
    sim_ref = []
    sim_own = []
    inter = []
    with open(csv_path) as f:
        rows = list(csv.reader(f))
        in_means = False
        for r in rows:
            if not r: continue
            if r[0] == "weight": in_means = True; continue
            if in_means:
                weights.append(float(r[0]))
                sim_ref.append(float(r[1]))
                sim_own.append(float(r[2]))
                inter.append(float(r[3]))
    return weights, sim_ref, sim_own, inter

def get_adaptive_data(csv_path):
    data = {}
    with open(csv_path) as f:
        rows = list(csv.reader(f))
        in_scheme = False
        for r in rows:
            if not r: continue
            if r[0] == "scheme": in_scheme = True; continue
            if in_scheme:
                data[r[0].strip('"')] = {"concept": float(r[1]), "content": float(r[2])}
    return data

os.makedirs("thesis_figures", exist_ok=True)

# Deliverable 1
plt.figure(figsize=(12, 10))
videos = [
    ("Geology", "fullrun_results/data/V1_Geology_collapse_metrics.csv"),
    ("Ecology", "fullrun_results/data/V2_Ecology_collapse_metrics.csv"),
    ("Sun", "fullrun_results/data/V3_Sun_collapse_metrics.csv"),
    ("Heart", "fullrun_results/data/V4_Heart_collapse_metrics.csv")
]
for i, (name, path) in enumerate(videos):
    w, ref, own, inter = get_collapse_data(path)
    ax = plt.subplot(2, 2, i+1)
    ax.plot(w, ref, "o-", color="#c0392b", label="similarity to reference")
    ax.plot(w, own, "s-", color="#2471a3", label="content preservation")
    ax.plot(w, inter, "^--", color="#7d8c00", label="inter-shot similarity")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"({chr(97+i)}) {name}")
    ax.set_xlabel("anchoring weight")
    ax.set_ylabel("DINOv2 / CLIP cosine similarity")
    ax.grid(alpha=0.3)
    if i == 0: ax.legend()
plt.tight_layout()
plt.savefig("thesis_figures/combined_frontier_4videos.png", dpi=150)
plt.close()

# Deliverable 2
plt.figure(figsize=(12, 10))
videos_ad = [
    ("Geology", "fullrun_results/data/V1_Geology_adaptive_anchor.csv"),
    ("Ecology", "fullrun_results/data/V2_Ecology_adaptive_anchor.csv"),
    ("Sun", "fullrun_results/data/V3_Sun_adaptive_anchor.csv"),
    ("Heart", "fullrun_results/data/V4_Heart_adaptive_anchor.csv")
]
for i, (name, path) in enumerate(videos_ad):
    data = get_adaptive_data(path)
    ax = plt.subplot(2, 2, i+1)
    fixed_w = []
    fx = []
    fy = []
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label="floor tau=0.7")
    for k, v in data.items():
        if k.startswith("fixed_"):
            fx.append(v["concept"])
            fy.append(v["content"])
            fixed_w.append(float(k.replace("fixed_w", "")))
    sorted_idx = np.argsort(fixed_w)
    fx = np.array(fx)[sorted_idx]
    fy = np.array(fy)[sorted_idx]
    ax.plot(fx, fy, 'o-', color="gray", label="fixed-weight frontier")
    if "adaptive" in data:
        ax.plot(data["adaptive"]["concept"], data["adaptive"]["content"], '*', markersize=15, color="gold", markeredgecolor="black", label="adaptive (DACA)")
    ax.set_title(f"({chr(97+i)}) {name}")
    ax.set_xlabel("concept (CLIP-T)")
    ax.set_ylabel("content preservation")
    ax.grid(alpha=0.3)
    if i == 0: ax.legend()
plt.tight_layout()
plt.savefig("thesis_figures/combined_adaptive_4videos.png", dpi=150)
plt.close()

# Deliverable 3
w, ref, own, inter = get_collapse_data("fullrun_results/data/V5_iPhone_collapse_metrics.csv")
plt.figure(figsize=(7.2, 4.6))
plt.plot(w, ref, "o-", color="#c0392b", label="similarity to reference")
plt.plot(w, own, "s-", color="#2471a3", label="content preservation")
plt.plot(w, inter, "^--", color="#7d8c00", label="inter-shot similarity")
plt.ylim(0, 1.0)
plt.title("Reward collapse: V5 iPhone (Contrast Control)")
plt.xlabel("anchoring weight")
plt.ylabel("DINOv2 / CLIP cosine similarity")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("thesis_figures/V5_iPhone_collapse_curve.png", dpi=150)
plt.close()

# Deliverable 4
shots = ["shot_003", "shot_006", "shot_010"]
weights = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
cols = 2 + len(weights)
fig, axes = plt.subplots(len(shots), cols, figsize=(18, 3.8), gridspec_kw={'width_ratios': [1, 1.5, 1, 1, 1, 1, 1, 1], 'wspace': 0.05, 'hspace': 0.0})
import textwrap

with open("runs/heart/storyboard.json") as f:
    sb = json.load(f)["shots"]
ref_img = Image.open("runs/heart/reference.png")

for i, sid in enumerate(shots):
    ax = axes[i, 0]
    ax.imshow(ref_img)
    if i == 0:
        ax.set_title("Reference", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    ax = axes[i, 1]
    # Set invisible dummy bounds to match image (1344x768) so set_title perfectly aligns
    ax.set_xlim(0, 1344)
    ax.set_ylim(768, 0)
    if i == 0:
        ax.set_title("Prompt", fontsize=14, fontweight="bold")
    prompt = [s["image_prompt"] for s in sb if s["shot_id"] == sid][0]
    wrapped = textwrap.fill(prompt, width=32)
    lines = wrapped.split('\n')
    if len(lines) > 5:
        wrapped = '\n'.join(lines[:5]) + "..."
    ax.text(1344/2, 768/2, wrapped, fontsize=9.5, va="center", ha="center")
    ax.axis("off")
    
    for j, w in enumerate(weights):
        ax = axes[i, j+2]
        try:
            img = Image.open(f"runs/heart/sweep/{sid}_w{w}.png")
            ax.imshow(img)
            if i == 0:
                ax.set_title(f"w = {w}", fontsize=14, fontweight="bold")
        except Exception as e:
            pass
        ax.axis("off")

plt.savefig("thesis_figures/V4_Heart_weight_sweep_proof.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()

# Deliverable 5 CSV
with open("thesis_figures/vlm_results_aggregate.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video", "condition", "n_pairs", "content_fidelity", "same_concept_rate", "content_homogenization_rate"])
    import csv as csv_mod
    with open("addon_results/vlm_results_aggregate.csv") as vf:
        r = list(csv_mod.DictReader(vf))
        for row in r:
            vid = row["video"]
            if row["n_shots"] == "0": continue
            cond = "daca" if row["method"] == "DACA" else "fixed_high_w (w=1.0)"
            n = row["n_shots"]
            cf = float(row["content_fidelity_winrate"])
            sc = float(row["same_concept_rate"])
            homo = float(row["copy_rate"])
            writer.writerow([vid, cond, n, f"{cf:.4f}", f"{sc:.4f}", f"{homo:.4f}"])

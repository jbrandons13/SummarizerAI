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

def get_adaptive_data(csv_path, vid=None, concept_summary_data=None):
    data = {}
    with open(csv_path) as f:
        rows = list(csv.reader(f))
        in_scheme = False
        for r in rows:
            if not r: continue
            if r[0] == "scheme": in_scheme = True; continue
            if in_scheme:
                data[r[0].strip('"')] = {"concept": float(r[1]), "content": float(r[2])}
                
    # Override concept for new videos (V6-V11)
    if vid and vid in ["V6", "V7", "V8", "V9", "V10", "V11"] and concept_summary_data:
        v_data = concept_summary_data[vid]
        for w_str in ["0.2", "0.4", "0.6"]:
            key = f"fixed_w{w_str}"
            if key not in data: data[key] = {"concept": 0.0, "content": 0.0} # just in case, but we need the content
            # Content remains from collapse metrics or old data
            data[key]["concept"] = v_data[w_str]
        if "adaptive" in data:
            data["adaptive"]["concept"] = v_data["w*"]
            
    return data

os.makedirs("thesis_figures", exist_ok=True)

# Load remeasured concept data for V6-V11
concept_summary = {}
with open("fullrun_results/data/n10_concept_remeasure_summary.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        concept_summary[row["Video"]] = {
            "0.2": float(row["Mean CLIP w=0.2"]),
            "0.4": float(row["Mean CLIP w=0.4"]),
            "0.6": float(row["Mean CLIP w=0.6"]),
            "w*": float(row["Mean CLIP w*"])
        }

videos = [
    ("V1", "Geology", "fullrun_results/data/V1_Geology_collapse_metrics.csv", "fullrun_results/data/V1_Geology_adaptive_anchor.csv"),
    ("V2", "Ecology", "fullrun_results/data/V2_Ecology_collapse_metrics.csv", "fullrun_results/data/V2_Ecology_adaptive_anchor.csv"),
    ("V3", "Sun", "fullrun_results/data/V3_Sun_collapse_metrics.csv", "fullrun_results/data/V3_Sun_adaptive_anchor.csv"),
    ("V4", "Heart", "fullrun_results/data/V4_Heart_collapse_metrics.csv", "fullrun_results/data/V4_Heart_adaptive_anchor.csv"),
    ("V6", "BlackHole", "fullrun_results/data/V6_BlackHole_collapse_metrics.csv", "fullrun_results/data/V6_BlackHole_adaptive_anchor.csv"),
    ("V7", "Immune", "fullrun_results/data/V7_Immune_collapse_metrics.csv", "fullrun_results/data/V7_Immune_adaptive_anchor.csv"),
    ("V8", "DNA", "fullrun_results/data/V8_DNA_collapse_metrics.csv", "fullrun_results/data/V8_DNA_adaptive_anchor.csv"),
    ("V9", "Photosynthesis", "fullrun_results/data/V9_Photosynthesis_collapse_metrics.csv", "fullrun_results/data/V9_Photosynthesis_adaptive_anchor.csv"),
    ("V10", "Neuron", "fullrun_results/data/V10_Neuron_collapse_metrics.csv", "fullrun_results/data/V10_Neuron_adaptive_anchor.csv"),
    ("V11", "Volcano", "fullrun_results/data/V11_Volcano_collapse_metrics.csv", "fullrun_results/data/V11_Volcano_adaptive_anchor.csv")
]

# Figure 1: combined_frontier_10videos.png
plt.figure(figsize=(15, 6)) # Adjust size for 2x5
for i, (vid, name, coll_path, _) in enumerate(videos):
    w, ref, own, inter = get_collapse_data(coll_path)
    ax = plt.subplot(2, 5, i+1)
    ax.plot(w, ref, "o-", color="#c0392b", label="similarity to reference")
    ax.plot(w, own, "s-", color="#2471a3", label="content preservation")
    ax.plot(w, inter, "^--", color="#7d8c00", label="inter-shot similarity")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"({chr(97+i)}) {name}")
    if i >= 5: ax.set_xlabel("anchoring weight") # Only on bottom row
    if i % 5 == 0: ax.set_ylabel("DINOv2 / CLIP cosine similarity") # Only on left col
    ax.grid(alpha=0.3)
    if i == 0: ax.legend(loc='lower left', prop={'size': 8})
plt.tight_layout()
plt.savefig("thesis_figures/combined_frontier_10videos.png", dpi=150)
plt.close()

# Figure 2: combined_adaptive_10videos.png
plt.figure(figsize=(15, 6))
for i, (vid, name, coll_path, adpt_path) in enumerate(videos):
    data = get_adaptive_data(adpt_path, vid, concept_summary)
    
    # Wait, the prompt says for V6-V11 content is from collapse_metrics. 
    # Let's ensure content for 0.2, 0.4, 0.6 comes from there if it's missing or we want to be exact
    w, ref, own, inter = get_collapse_data(coll_path)
    content_map = {str(wk): owk for wk, owk in zip(w, own)}
    for wk in ["0.2", "0.4", "0.6"]:
        key = f"fixed_w{wk}"
        if key not in data:
            data[key] = {"concept": 0.0, "content": content_map.get(wk, 0.0)}
            if vid in concept_summary:
                data[key]["concept"] = concept_summary[vid][wk]
        else:
            # Overwrite content from collapse_metrics to be safe as per brief "content dari `collapse_metrics`"
            data[key]["content"] = content_map.get(wk, data[key]["content"])
            
    ax = plt.subplot(2, 5, i+1)
    fixed_w = []
    fx = []
    fy = []
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label="floor tau=0.7")
    for k, v in data.items():
        if k.startswith("fixed_w0.2") or k.startswith("fixed_w0.4") or k.startswith("fixed_w0.6"):
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
    if i >= 5: ax.set_xlabel("concept (CLIP-T)")
    if i % 5 == 0: ax.set_ylabel("content preservation")
    ax.grid(alpha=0.3)
    if i == 0: ax.legend(loc='lower left', prop={'size': 8})
plt.tight_layout()
plt.savefig("thesis_figures/combined_adaptive_10videos.png", dpi=150)
plt.close()

# Figure 3: fair_plane_summary_n10.png (Optional but requested)
plt.figure(figsize=(8, 6))
plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label="floor tau=0.7")

# We can read from n10_fair_plane_summary.csv directly
daca_points = []
with open("fullrun_results/data/n10_fair_plane_summary.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        daca_points.append({
            "name": row["Topic_Name"],
            "concept": float(row["Mean_Concept_CLIP"]),
            "content": float(row["Mean_Content_Sim"])
        })

colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, pt in enumerate(daca_points):
    plt.plot(pt["concept"], pt["content"], '*', markersize=15, color=colors[i], markeredgecolor="black", label=pt["name"])
    plt.annotate(pt["name"], (pt["concept"], pt["content"]), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.title("Fair Plane Summary (n=10 videos) DACA Anchor Points")
plt.xlabel("concept (CLIP-T)")
plt.ylabel("content preservation")
plt.grid(alpha=0.3)
# plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5)) # Might be too crowded, annotation is enough
plt.tight_layout()
plt.savefig("thesis_figures/fair_plane_summary_n10.png", dpi=150)
plt.close()

print("Figures generated successfully!")

import shutil
import glob
import os
import csv
from PIL import Image

# 1. Copy files to fullrun_results/data/
for f in glob.glob("3videos_results/*.csv"):
    if "concept_remeasure" not in f and "vlm_judge" not in f:
        shutil.copy(f, "fullrun_results/data/")
for f in glob.glob("3videos_results/*.png"):
    shutil.copy(f, "fullrun_results/data/")

# 2. Update vlm_judge_n10_final.csv
all_rows = []
with open("vlm_judge_n10_final.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    all_rows.extend(list(reader))

# Remove Immune, DNA, Photosynthesis
all_rows = [r for r in all_rows if r[0] not in ["Immune", "DNA", "Photosynthesis"]]

# Add Eye, Hurricane, Reef
with open("3videos_results/vlm_judge_3videos.csv") as f:
    reader = csv.reader(f)
    next(reader)
    all_rows.extend(list(reader))

with open("vlm_judge_n10_final.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print("Updated vlm_judge_n10_final.csv")

# 3. Create Grounding Success Figure
fig_w = 4 * 1024
fig_h = 3 * 1024
fig = Image.new("RGB", (fig_w, fig_h))

videos = [
    ("V12_Eye", 0),
    ("V13_Hurricane", 1),
    ("V14_Reef", 2)
]

for vid_name, row_idx in videos:
    ref = Image.open(f"gate_check/{vid_name}_reference.png").resize((1024, 1024))
    s1 = Image.open(f"gate_check/{vid_name}_shot_1_w0.png").resize((1024, 1024))
    s2 = Image.open(f"gate_check/{vid_name}_shot_2_w0.png").resize((1024, 1024))
    s3 = Image.open(f"gate_check/{vid_name}_shot_3_w0.png").resize((1024, 1024))
    
    fig.paste(ref, (0, row_idx * 1024))
    fig.paste(s1, (1024, row_idx * 1024))
    fig.paste(s2, (2048, row_idx * 1024))
    fig.paste(s3, (3072, row_idx * 1024))

fig.save("grounding_success_3videos.png")
print("Generated grounding_success_3videos.png")

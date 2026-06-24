import csv
import shutil
import os

# 1. Update vlm_judge_n10_final.csv
all_rows = []
with open("vlm_judge_n10_final.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        if r[0] != "Reef":
            all_rows.append(r)

with open("15videos_results/vlm_judge_River.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    all_rows.extend(list(reader))

with open("vlm_judge_n10_final.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print("Updated vlm_judge_n10_final.csv with River")

# 2. Update n10_fair_plane_summary.csv
summary_rows = []
with open("fullrun_results/data/n10_fair_plane_summary.csv", "r") as f:
    reader = csv.reader(f)
    header_sum = next(reader)
    for r in reader:
        if r[0] != "V14":
            summary_rows.append(r)

with open("15videos_results/V15_River_adaptive_anchor.csv", "r") as f:
    lines = list(csv.reader(f))
    last_line = lines[-1]
    mean_concept = last_line[1]
    mean_content = last_line[2]

summary_rows.append(["V15", "River", mean_concept, mean_content])

with open("fullrun_results/data/n10_fair_plane_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header_sum)
    writer.writerows(summary_rows)

print("Updated n10_fair_plane_summary.csv with V15 River")

# 3. Copy files to results folder
os.makedirs("replacement_results_2026-06-15/15videos_results", exist_ok=True)
for file in os.listdir("15videos_results"):
    shutil.copy(os.path.join("15videos_results", file), "replacement_results_2026-06-15/15videos_results/")

shutil.copy("vlm_judge_n10_final.csv", "replacement_results_2026-06-15/")
shutil.copy("fullrun_results/data/n10_fair_plane_summary.csv", "replacement_results_2026-06-15/")

print("Files backed up to replacement_results_2026-06-15")

import csv
import os

v_ids = [1, 2, 3, 4, 6, 10, 11, 12, 13, 14]
data_dir = "fullrun_results/data"
all_files = os.listdir(data_dir)

results = []
results.append(["Video_ID", "Topic_Name", "Mean_Concept_CLIP", "Mean_Content_Sim"])

for vid in v_ids:
    prefix = f"V{vid}_"
    file_name = None
    for f in all_files:
        if f.startswith(prefix) and f.endswith("adaptive_anchor.csv"):
            file_name = f
            break
    
    if not file_name:
        print(f"Warning: File for V{vid} not found!")
        continue
        
    parts = file_name.split("_")
    v_name = parts[1]
    
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)
        
        last_line = None
        for line in reversed(lines):
            if len(line) >= 3 and "adaptive" in line[0]:
                last_line = line
                break
                
        if last_line:
            mean_concept = last_line[1]
            mean_content = last_line[2]
            results.append([f"V{vid}", v_name, mean_concept, mean_content])
        else:
            print(f"Warning: Could not find 'adaptive' row in {file_name}")

# Also include V5 for reference if user wants? The prompt said "kumpulkan 10 video itu", so n=10 (excluding V5).

out_path = "fullrun_results/data/n10_fair_plane_summary.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print(f"Aggregated n=10 data saved to {out_path}")
for row in results:
    print(row)

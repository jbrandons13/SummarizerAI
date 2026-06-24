import csv

all_rows = []
with open("vlm_judge_n10.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    all_rows.extend(list(reader))
    
with open("vlm_judge_v1_v2.csv") as f:
    reader = csv.reader(f)
    next(reader)
    all_rows.extend(list(reader))
    
# Keep only the requested 8 videos: Geology, Ecology, BlackHole, Immune, DNA, Photosynthesis, Neuron, Volcano
# Note: In the csv they are named "Geology", "Ecology", "BlackHole", "Immune", etc.
valid_videos = ["Geology", "Ecology", "BlackHole", "Immune", "DNA", "Photosynthesis", "Neuron", "Volcano"]
filtered_rows = [row for row in all_rows if row[0] in valid_videos]

with open("vlm_judge_n10_final.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(filtered_rows)
    
print("Combined into vlm_judge_n10_final.csv")

import json
import matplotlib.pyplot as plt
import os
import yaml

def main():
    # 1. Histogram of concepts
    video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    sb_path = f"data/intermediate/{video}/phase4/storyboard.json"
    with open(sb_path, "r") as f:
        sb = json.load(f)["shots"]
        
    concept_counts = {}
    for shot in sb:
        tag = shot.get("topic_tag", "unknown")
        concept_counts[tag] = concept_counts.get(tag, 0) + 1
        
    tags = list(concept_counts.keys())
    counts = [concept_counts[t] for t in tags]
    
    plt.figure(figsize=(10, 6))
    plt.bar(tags, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Shots")
    plt.title("Shots per Concept (Geology Video)")
    plt.tight_layout()
    plt.savefig("runs/G0_A0_geology/concept_histogram.png")
    
    # 2. Verify 144 attention processors
    with open("runs/G0_A0_geology/unet_attn_map.txt", "r") as f:
        lines = f.read().splitlines()
        
    attn1_count = sum(1 for l in lines if "attn1" in l)
    attn2_count = sum(1 for l in lines if "attn2" in l)
    other_count = len(lines) - attn1_count - attn2_count
    
    print(f"Total processors: {len(lines)}")
    print(f"attn1 count: {attn1_count}")
    print(f"attn2 count: {attn2_count}")
    print(f"other count: {other_count}")
    
    if other_count > 0:
        others = [l for l in lines if "attn1" not in l and "attn2" not in l]
        print("Other processors:")
        for o in others:
            print(f"  {o}")
            
    # Check extra keys vs 140
    # SDXL UNet usually has 140 (70 blocks * 2).
    # Why 144? 
    # Let's print out the structure.
    
if __name__ == "__main__":
    main()

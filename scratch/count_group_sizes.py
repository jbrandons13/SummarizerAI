import json
from pathlib import Path

def main():
    intermediate_dir = Path("data/intermediate")
    if not intermediate_dir.exists():
        print("Intermediate directory not found.")
        return

    assignment_files = list(intermediate_dir.glob("*/p4_assignments.json"))
    
    if not assignment_files:
        print("No p4_assignments.json files found.")
        return

    print(f"Found {len(assignment_files)} processed video folders.\n")
    print("| Video ID | Total Groups | Groups with > 1 Sentence | Max Group Size | Details (Sentence Count per Group) |")
    print("|---|---|---|---|---|")
    
    total_groups_all = 0
    multi_sentence_groups_all = 0

    for fpath in sorted(assignment_files):
        video_id = fpath.parent.name
        with open(fpath) as f:
            assignments = json.load(f)
            
        group_sizes = [len(a["sentence_ids"]) for a in assignments]
        total_groups = len(assignments)
        multi_sentence_groups = sum(1 for sz in group_sizes if sz > 1)
        max_size = max(group_sizes) if group_sizes else 0
        
        total_groups_all += total_groups
        multi_sentence_groups_all += multi_sentence_groups
        
        print(f"| {video_id} | {total_groups} | {multi_sentence_groups} | {max_size} | {group_sizes} |")

    print(f"\n**Aggregated Summary:**")
    print(f"- Total groups analyzed across all videos: {total_groups_all}")
    print(f"- Groups containing more than 1 sentence: {multi_sentence_groups_all} ({multi_sentence_groups_all/total_groups_all*100:.1f}%)")

if __name__ == "__main__":
    main()

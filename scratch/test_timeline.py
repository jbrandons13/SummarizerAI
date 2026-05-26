import json
from pathlib import Path

def main():
    metadata_path = Path("data/output/review_1/summary_grouping_gate_metadata.json")
    with open(metadata_path) as f:
        meta = json.load(f)
        
    assignments_path = Path("data/intermediate/review_1/p4_assignments.json")
    with open(assignments_path) as f:
        assignments = json.load(f)
        
    padding_s = 0.15 # 150ms from configs/default.yaml tts.silence_padding_ms
    
    segments = meta["segments"]
    current_time = 0.0
    
    print("| Group ID | Start (s) | End (s) | Action | Similarity Score |")
    print("|----------|-----------|---------|--------|------------------|")
    
    for i, seg in enumerate(segments):
        v_dur = seg["source_time_range"][1] - seg["source_time_range"][0]
        a_dur = seg["group_audio_duration"]
        
        duration_in_concat = v_dur
        if i < len(segments) - 1:
            spacer_duration = (a_dur + padding_s) - v_dur
            if spacer_duration > 0.05:
                duration_in_concat += spacer_duration
        else:
            if a_dur > v_dur + 0.01:
                spacer_duration = a_dur - v_dur
                duration_in_concat += spacer_duration
                
        action = assignments[i]["action"].upper()
        score = seg["similarity_score"]
        
        print(f"| {i} | {current_time:.3f} | {current_time + duration_in_concat:.3f} | {action} | {score:.4f} |")
        current_time += duration_in_concat

if __name__ == "__main__":
    main()

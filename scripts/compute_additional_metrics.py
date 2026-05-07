import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

def compute_max_consecutive(sequence):
    if not sequence:
        return 0
    max_count = 1
    current_count = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_count += 1
        else:
            max_count = max(max_count, current_count)
            current_count = 1
    return max(max_count, current_count)

def compute_strict_viscoher(matches, frame_embeddings):
    consecutive_sims = []
    
    def find_nearest_embedding(scene_id, ts, frame_embeddings):
        best_key = None
        min_dist = float('inf')
        for k in frame_embeddings.keys():
            if k[0] == scene_id:
                dist = abs(k[1] - ts)
                if dist < min_dist:
                    min_dist = dist
                    best_key = k
        if best_key:
            return frame_embeddings[best_key]
        return None

    for i in range(len(matches) - 1):
        scene_a = matches[i]['matched_scene_id']
        scene_b = matches[i+1]['matched_scene_id']
        
        # EXCLUDE SAME-SCENE PAIRS
        if scene_a == scene_b:
            continue
            
        ts_a = matches[i].get('best_frame_timestamp', 0.0)
        ts_b = matches[i+1].get('best_frame_timestamp', 0.0)

        # Convert keys to tuples if they are lists (sometimes json does this)
        key_a = (scene_a, ts_a)
        key_b = (scene_b, ts_b)

        emb_a = frame_embeddings.get(key_a)
        if emb_a is None:
            emb_a = find_nearest_embedding(scene_a, ts_a, frame_embeddings)
            
        emb_b = frame_embeddings.get(key_b)
        if emb_b is None:
            emb_b = find_nearest_embedding(scene_b, ts_b, frame_embeddings)

        if emb_a is None or emb_b is None:
            continue

        norm_a, norm_b = np.linalg.norm(emb_a), np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            continue
        consecutive_sims.append(float(np.dot(emb_a, emb_b) / (norm_a * norm_b)))

    if not consecutive_sims:
        return 0.0 

    return float(np.mean(consecutive_sims))

def main():
    csv_path = "results/aggregated_20260507_083942/ablation_results.csv"
    df = pd.read_csv(csv_path)
    
    # Remove existing columns if they exist to avoid duplicates
    cols_to_add = ["num_unique_scenes_used", "num_sentences", "scene_diversity", "max_consecutive_reuse", "viscoher_strict"]
    for col in cols_to_add:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            
    new_data = []
    model_slug = "google_siglip2_so400m_patch16_naflex"
    
    for idx, row in df.iterrows():
        video_id = row['video_id']
        arm = row['arm']
        
        matches_path = Path(f"data/intermediate/{video_id}/scene_matches_{arm}.json")
        if not matches_path.exists():
            print(f"Warning: {matches_path} not found")
            new_data.append({
                "num_unique_scenes_used": 0,
                "num_sentences": 0,
                "scene_diversity": 0.0,
                "max_consecutive_reuse": 0,
                "viscoher_strict": 0.0
            })
            continue
            
        with open(matches_path, "r") as f:
            matches_data = json.load(f)
            matches = matches_data['matches']
            
        scene_ids = [m['matched_scene_id'] for m in matches]
        num_unique = len(set(scene_ids))
        num_sentences = len(matches)
        diversity = num_unique / num_sentences if num_sentences > 0 else 0.0
        max_consecutive = compute_max_consecutive(scene_ids)
        
        # Strict VisCoher
        emb_path = Path(f"data/intermediate/{video_id}/embeddings_{model_slug}.joblib")
        if emb_path.exists():
            frame_embs = joblib.load(emb_path)
            strict_vis = compute_strict_viscoher(matches, frame_embs)
        else:
            strict_vis = 0.0
            
        new_data.append({
            "num_unique_scenes_used": num_unique,
            "num_sentences": num_sentences,
            "scene_diversity": diversity,
            "max_consecutive_reuse": max_consecutive,
            "viscoher_strict": strict_vis
        })
        
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([df, new_df], axis=1)
    
    combined_df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with new metrics.")

if __name__ == "__main__":
    main()

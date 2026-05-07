import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def get_sim_matrix(vid):
    manifest_path = Path(f"data/intermediate/{vid}/keyframes_manifest.json")
    summary_path = Path(f"data/intermediate/{vid}/summary_script.json")
    cap_path = Path(f"data/intermediate/{vid}/keyframes_captions.json")
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    with open(summary_path, "r") as f:
        summary = json.load(f)
    with open(cap_path, "r") as f:
        captions = json.load(f)
        
    sentences = [s["text"] for s in summary["sentences"]]
    num_sentences = len(sentences)
    scenes = manifest["scenes"]
    num_scenes = len(scenes)
    
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    sent_embs = st_model.encode(sentences, convert_to_tensor=True)
    
    sim_matrix = np.zeros((num_sentences, num_scenes))
    for j, scene in enumerate(scenes):
        scene_keys = [f"{scene['id']}_{ts}" for ts in scene["multi_frame_timestamps"][:3]]
        scene_caps = [captions[k] for k in scene_keys if k in captions]
        if not scene_caps:
            scene_caps = [""] # dummy
        cap_embs = st_model.encode(scene_caps, convert_to_tensor=True)
        scores = util.cos_sim(sent_embs, cap_embs).cpu().numpy()
        for i in range(num_sentences):
            s = scores[i]
            k = min(2, len(s))
            top_k_indices = np.argsort(-s)[:k]
            sim_matrix[i, j] = np.mean(s[top_k_indices])
            
    # Apply temporal
    from src.phase4_retrieve import compute_temporal_scores, min_max_normalize
    kf_timestamps = [s["keyframe_timestamp"] for s in scenes]
    new_sim_matrix = np.zeros_like(sim_matrix)
    for i in range(num_sentences):
        semantic_scores = sim_matrix[i]
        hint = summary["sentences"][i].get("source_timestamp_hint")
        temporal_scores = compute_temporal_scores(hint, kf_timestamps, sigma=30.0)
        new_sim_matrix[i] = (1 - 0.3) * min_max_normalize(semantic_scores) + 0.3 * min_max_normalize(temporal_scores)
        
    return new_sim_matrix

if __name__ == "__main__":
    print("Computing review_7...")
    mat = get_sim_matrix("review_7")
    print(mat.shape)
    print("Done")

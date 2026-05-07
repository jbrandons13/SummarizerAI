import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from src.phase4_retrieve import RetrievalBackend, compute_temporal_scores, min_max_normalize
from src.schemas import KeyframeScene
import joblib
from scripts.compute_additional_metrics import compute_strict_viscoher

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
    kf_timestamps = [s["keyframe_timestamp"] for s in scenes]
    new_sim_matrix = np.zeros_like(sim_matrix)
    for i in range(num_sentences):
        semantic_scores = sim_matrix[i]
        hint = summary["sentences"][i].get("source_timestamp_hint")
        temporal_scores = compute_temporal_scores(hint, kf_timestamps, sigma=30.0)
        new_sim_matrix[i] = (1 - 0.3) * min_max_normalize(semantic_scores) + 0.3 * min_max_normalize(temporal_scores)
        
    # parse scenes into KeyframeScene objects
    scene_objs = [KeyframeScene(**s) for s in scenes]
    video_dur = max(s.end_seconds for s in scene_objs)
    return new_sim_matrix, scene_objs, video_dur

def run_tests():
    print("Starting tests for CCMA...")
    test_videos = ["review_2", "review_5", "review_7"]
    sim_matrices = {}
    print("Extracting sim_matrices...")
    for vid in test_videos:
        print(f"Loading {vid}...")
        sim_matrices[vid] = get_sim_matrix(vid)
        print(f"Done loading {vid}")
        
    class DummyBackend(RetrievalBackend):
        def retrieve(self, summary, manifest, progress_callback=None):
            pass
            
    backend = DummyBackend({})
    
    def compute_max_consecutive(assign):
        if not assign: return 0
        max_consec = 1
        cur = 1
        for i in range(1, len(assign)):
            if assign[i] == assign[i-1]:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 1
        return max(max_consec, cur)

    print("\n--- TEST 1: Constraint satisfaction on all videos ---")
    for vid in test_videos:
        sim_matrix, scenes, video_dur = sim_matrices[vid]
        assign_ccma = backend.ccma_align_sequence(
            sim_matrix, scenes, video_dur,
            c_max=3, reuse_penalty=0.2, forward_jump_penalty=0.1, backward_jump_penalty=2.0
        )
        max_consec = compute_max_consecutive(assign_ccma)
        assert max_consec <= 3, f"Constraint violated for {vid}: {max_consec} > 3"
        print(f"PASS: K_max=3 constraint satisfied for {vid}, max_consec={max_consec}")
    
    print("\n--- TEST 2: Looping case fix on review_7 ---")
    sim_matrix, scenes, video_dur = sim_matrices["review_7"]
    assign_dp_r7 = backend.dp_sequence_align(
        sim_matrix, scenes, video_dur,
        jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5
    )
    assign_ccma_r7 = backend.ccma_align_sequence(
        sim_matrix, scenes, video_dur,
        c_max=3, reuse_penalty=0.2, forward_jump_penalty=0.1, backward_jump_penalty=2.0
    )
    print(f"Vanilla DP: {assign_dp_r7}")
    print(f"CCMA (K=3): {assign_ccma_r7}")
    assert assign_dp_r7 != assign_ccma_r7, "CCMA did not change the assignment for review_7"
    print("PASS: CCMA changed the looping assignment")
    
    print("\n--- TEST 3: Hyperparameter sweep ---")
    c_vals = [2, 3]
    fwd_vals = [0.05, 0.1, 0.2]
    
    results = []
    
    for vid in test_videos:
        sim_matrix, scenes, video_dur = sim_matrices[vid]
        
        # Load frame embeddings for strict viscoher
        model_slug = "google_siglip2_so400m_patch16_naflex"
        emb_path = Path(f"data/intermediate/{vid}/embeddings_{model_slug}.joblib")
        if emb_path.exists():
            frame_embs = joblib.load(emb_path)
        else:
            frame_embs = {}
            
        for c in c_vals:
            for fwd in fwd_vals:
                assign = backend.ccma_align_sequence(
                    sim_matrix, scenes, video_dur,
                    c_max=c, reuse_penalty=0.2, forward_jump_penalty=fwd, backward_jump_penalty=2.0
                )
                
                max_c = compute_max_consecutive(assign)
                diversity = len(set(assign)) / len(assign) if len(assign) > 0 else 0
                
                # Mock matches for compute_strict_viscoher
                matches = []
                for i, scene_idx in enumerate(assign):
                    scene = scenes[scene_idx]
                    matches.append({
                        'matched_scene_id': scene_idx,
                        'best_frame_timestamp': scene.keyframe_timestamp
                    })
                    
                strict_vis = compute_strict_viscoher(matches, frame_embs)
                
                results.append({
                    "video": vid,
                    "c_max": c,
                    "fwd_jump": fwd,
                    "max_consec": max_c,
                    "diversity": diversity,
                    "viscoher_strict": strict_vis,
                    "assignment": assign
                })
                
    with open("test_results_ccma.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n| Video | C_max | Fwd Jump | Max Consec | Scene Diversity | VisCoher_strict | Assignment |")
    print("|-------|-------|----------|------------|-----------------|-----------------|------------|")
    for r in results:
        print(f"| {r['video']} | {r['c_max']} | {r['fwd_jump']} | {r['max_consec']} | {r['diversity']:.3f} | {r['viscoher_strict']:.3f} | {r['assignment']} |")
        
    print("\nSweep complete!")

if __name__ == "__main__":
    run_tests()

import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from src.phase4_retrieve import RetrievalBackend, compute_temporal_scores, min_max_normalize
from src.schemas import KeyframeScene

# Disable CUDA for tests to avoid VRAM issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_test_data(vid, track="caption_temporal"):
    manifest_path = Path(f"data/intermediate/{vid}/keyframes_manifest.json")
    summary_path = Path(f"data/intermediate/{vid}/summary_script.json")
    cap_path = Path(f"data/intermediate/{vid}/keyframes_captions.json")
    
    if not manifest_path.exists() or not summary_path.exists() or not cap_path.exists():
        raise FileNotFoundError(f"Missing data for {vid}")

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
            
    # Apply temporal guidance if requested
    kf_timestamps = [s["keyframe_timestamp"] for s in scenes]
    if "temporal" in track:
        new_sim_matrix = np.zeros_like(sim_matrix)
        for i in range(num_sentences):
            semantic_scores = sim_matrix[i]
            hint = summary["sentences"][i].get("source_timestamp_hint")
            temporal_scores = compute_temporal_scores(hint, kf_timestamps, sigma=30.0)
            new_sim_matrix[i] = (1 - 0.3) * min_max_normalize(semantic_scores) + 0.3 * min_max_normalize(temporal_scores)
        sim_matrix = new_sim_matrix
    else:
        # Just normalize
        for i in range(num_sentences):
            sim_matrix[i] = min_max_normalize(sim_matrix[i])
        
    # parse scenes into KeyframeScene objects
    scene_objs = [KeyframeScene(**s) for s in scenes]
    video_dur = max(s.end_seconds for s in scene_objs)
    return sim_matrix, scene_objs, video_dur

def compute_max_consecutive(seq):
    if not seq: return 0
    max_c, cur = 1, 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 1
    return max_c

class DummyBackend(RetrievalBackend):
    def retrieve(self, summary, manifest, progress_callback=None):
        pass

def run_verification():
    backend = DummyBackend({})
    all_10_video_ids = [f"review_{i}" for i in range(1, 11)]

    print("--- Test 1: Reduction Property ---")
    for video_id in ["review_2", "review_5", "review_7"]:
        sim_matrix, scenes, video_dur = load_test_data(video_id, "caption_temporal")
        
        # Vanilla DP
        assign_dp = backend.dp_sequence_align(
            sim_matrix, scenes, video_dur,
            jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5
        )
        
        # CCMA with relaxed constraints — should equal DP
        assign_ccma_relaxed = backend.ccma_align_sequence(
            sim_matrix, scenes, video_dur,
            c_max=1000,
            reuse_penalty=-0.01,  # equal to -reuse_bonus of DP (bonus = negative penalty)
            jump_penalty=0.01,
            backward_penalty=0.5
        )
        
        assert assign_dp == assign_ccma_relaxed, (
            f"FAIL on {video_id}:\n"
            f"  DP:   {assign_dp}\n"
            f"  CCMA: {assign_ccma_relaxed}"
        )
        print(f"PASS: {video_id} — reduction property holds")

    print("\n--- Test 2: Constraint Satisfaction ---")
    for c_max in [2, 3]:
        for video_id in all_10_video_ids:
            for track in ["caption_temporal"]: # Skipping siglip to save time, logic is the same
                try:
                    sim_matrix, scenes, video_dur = load_test_data(video_id, track)
                except FileNotFoundError:
                    continue
                assign = backend.ccma_align_sequence(
                    sim_matrix, scenes, video_dur,
                    c_max=c_max, reuse_penalty=0.2,
                    jump_penalty=0.01, backward_penalty=0.5
                )
                mc = compute_max_consecutive(assign)
                assert mc <= c_max, f"VIOLATION: {video_id} {track} c_max={c_max} got {mc}"
                
    print("PASS: Constraint satisfaction across all available videos")

    print("\n--- Test 3: Determinism ---")
    sim_matrix, scenes, video_dur = load_test_data("review_7", "caption_temporal")
    results = []
    for _ in range(5):
        assign = backend.ccma_align_sequence(
            sim_matrix, scenes, video_dur,
            c_max=3, reuse_penalty=0.2,
            jump_penalty=0.01, backward_penalty=0.5
        )
        results.append(assign)

    assert all(r == results[0] for r in results), "FAIL: non-deterministic output"
    print("PASS: Determinism across 5 runs")

    print("\n--- Test 4: Looping case still fixed ---")
    sim_matrix, scenes, video_dur = load_test_data("review_7", "caption_temporal")

    assign_dp = backend.dp_sequence_align(
        sim_matrix, scenes, video_dur,
        jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5
    )

    assign_ccma = backend.ccma_align_sequence(
        sim_matrix, scenes, video_dur,
        c_max=3, reuse_penalty=0.2,
        jump_penalty=0.01, backward_penalty=0.5
    )

    mc_dp = compute_max_consecutive(assign_dp)
    mc_ccma = compute_max_consecutive(assign_ccma)

    assert mc_ccma <= 3, f"CCMA constraint violated: max_consec={mc_ccma}"
    print(f"PASS: review_7 max_consec — DP={mc_dp}, CCMA={mc_ccma}")
    print(f"  DP assignment:   {assign_dp}")
    print(f"  CCMA assignment: {assign_ccma}")

if __name__ == "__main__":
    run_verification()

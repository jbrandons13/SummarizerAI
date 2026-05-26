#!/usr/bin/env python
import os
import sys
import json
import time
import logging
import subprocess
import traceback
import csv
from pathlib import Path
import numpy as np
import yaml
import torch

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.vram import VRAMManager
from src.models.siglip import SigLIPEncoder
from src.phase4_retrieve import Sentence as P4Sentence, Scene as P4Scene
from src.phase5_assemble import Phase5Assembler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ablation_study_16_arms")

# --- Helper Math Functions ---

def _cosine_to_all(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = float(np.linalg.norm(query))
    if q_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    return (matrix @ query) / (row_norms * q_norm)

def _gaussian_temporal_weights(scene_centers: np.ndarray, hint_center: float, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.ones_like(scene_centers, dtype=np.float32)
    delta = scene_centers - float(hint_center)
    return np.exp(-(delta ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-6:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min)

# --- Alignment Algorithm Implementations ---

def dp_sequence_align(
    sim_matrix: np.ndarray,
    scene_timestamps: np.ndarray,
    video_duration: float,
    jump_penalty: float = 0.01,
    reuse_bonus: float = 0.01,
    backward_penalty: float = 0.5,
) -> list:
    N, M = sim_matrix.shape
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1e9)

    if N == 1:
        return [int(np.argmax(sim_matrix[0]))]

    dt_matrix = (scene_timestamps[None, :] - scene_timestamps[:, None]) / max(video_duration, 1e-6)

    transition_matrix = np.where(
        dt_matrix >= 0,
        jump_penalty * dt_matrix,
        jump_penalty * np.abs(dt_matrix) + backward_penalty,
    )
    np.fill_diagonal(transition_matrix, -reuse_bonus)

    dp = np.full((N, M), -np.inf)
    backptr = np.full((N, M), -1, dtype=int)
    dp[0] = sim_matrix[0]

    for i in range(1, N):
        candidates = dp[i - 1][:, None] - transition_matrix
        dp[i] = sim_matrix[i] + candidates.max(axis=0)
        backptr[i] = candidates.argmax(axis=0)

    assignment = [0] * N
    assignment[N - 1] = int(np.argmax(dp[N - 1]))
    for i in range(N - 2, -1, -1):
        assignment[i] = int(backptr[i + 1][assignment[i + 1]])

    return assignment

def ccma_align_sequence(
    sim_matrix: np.ndarray,
    scene_timestamps: np.ndarray,
    video_duration: float,
    c_max: int = 3,
    reuse_penalty: float = 0.2,
    jump_penalty: float = 0.01,
    backward_penalty: float = 0.5,
) -> list:
    N, M = sim_matrix.shape
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1e9)

    if N == 1:
        return [int(np.argmax(sim_matrix[0]))]

    dp = np.full((N, M, c_max), -np.inf)
    bp_j = np.full((N, M, c_max), -1, dtype=int)
    bp_c = np.full((N, M, c_max), -1, dtype=int)

    dp[0, :, 0] = sim_matrix[0]

    for i in range(1, N):
        prev_best_c = np.max(dp[i-1], axis=1)
        prev_best_c_idx = np.argmax(dp[i-1], axis=1)

        for j in range(M):
            best_score = -np.inf
            best_j_prev = -1
            best_c_prev = -1

            for j_prev in range(M):
                if j_prev == j:
                    continue

                dt = (scene_timestamps[j] - scene_timestamps[j_prev]) / max(video_duration, 1e-6)
                if dt >= 0:
                    cost = jump_penalty * dt
                else:
                    cost = jump_penalty * abs(dt) + backward_penalty

                score = prev_best_c[j_prev] - cost
                if score > best_score:
                    best_score = score
                    best_j_prev = j_prev
                    best_c_prev = prev_best_c_idx[j_prev]

            if best_score > -np.inf:
                dp[i, j, 0] = sim_matrix[i, j] + best_score
                bp_j[i, j, 0] = best_j_prev
                bp_c[i, j, 0] = best_c_prev

            for c_idx in range(1, min(i + 1, c_max)):
                prev_score = dp[i-1, j, c_idx-1]
                if prev_score > -np.inf:
                    dp[i, j, c_idx] = sim_matrix[i, j] + prev_score - reuse_penalty
                    bp_j[i, j, c_idx] = j
                    bp_c[i, j, c_idx] = c_idx - 1

    flat_idx = np.argmax(dp[N-1])
    final_j, final_c = np.unravel_index(flat_idx, (M, c_max))

    assignment = [0] * N
    assignment[N-1] = int(final_j)

    cur_j, cur_c = int(final_j), int(final_c)
    for i in range(N-1, 0, -1):
        prev_j = bp_j[i, cur_j, cur_c]
        prev_c = bp_c[i, cur_j, cur_c]
        assignment[i-1] = int(prev_j)
        cur_j, cur_c = int(prev_j), int(prev_c)

    return assignment

# --- Greedy Grouping Implementation ---

def greedy_grouping(sentences, scene_matrix, scene_centers, siglip, sigma=30.0, extend_epsilon=0.03, max_group_size=5, join_sep=" "):
    n = len(sentences)
    groups = []
    i = 0
    while i < n:
        group_ids = [i]
        joined_text = sentences[i].text
        joined_emb = siglip.encode(joined_text)

        starts = [sentences[sid].timestamp_hint[0] for sid in group_ids]
        ends = [sentences[sid].timestamp_hint[1] for sid in group_ids]
        hint_center = (min(starts) + max(ends)) / 2.0

        raw = _cosine_to_all(joined_emb, scene_matrix)
        weights = _gaussian_temporal_weights(scene_centers, hint_center, sigma)
        weighted = raw * weights

        locked_idx = int(np.argmax(weighted))
        best_weighted = float(weighted[locked_idx])
        sim_trail = [best_weighted]

        while i + len(group_ids) < n and len(group_ids) < max_group_size:
            next_idx = i + len(group_ids)
            candidate_text = joined_text + join_sep + sentences[next_idx].text
            candidate_emb = siglip.encode(candidate_text)

            candidate_ids = group_ids + [next_idx]
            cand_starts = [sentences[sid].timestamp_hint[0] for sid in candidate_ids]
            cand_ends = [sentences[sid].timestamp_hint[1] for sid in candidate_ids]
            candidate_hint_center = (min(cand_starts) + max(cand_ends)) / 2.0

            cand_raw = _cosine_to_all(candidate_emb, scene_matrix)
            cand_weights = _gaussian_temporal_weights(scene_centers, candidate_hint_center, sigma)
            cand_weighted = cand_raw * cand_weights

            candidate_best_idx = int(np.argmax(cand_weighted))
            candidate_locked_weighted = float(cand_weighted[locked_idx])

            same_scene = (candidate_best_idx == locked_idx)
            tolerable_drop = (candidate_locked_weighted >= best_weighted - extend_epsilon)

            if not (same_scene and tolerable_drop):
                break

            group_ids.append(next_idx)
            joined_text = candidate_text
            joined_emb = candidate_emb
            best_weighted = candidate_locked_weighted
            sim_trail.append(best_weighted)

        groups.append({
            "sentence_ids": group_ids,
            "scene_idx": locked_idx,
            "text": joined_text,
            "hint_range": (min(starts), max(ends)),
            "similarity_trail": sim_trail
        })
        i += len(group_ids)

    return groups

# --- 16 arms sweep orchestrator ---

ARM_CONFIGS = {
    # 2 Normalizations x 8 combinations = 16 Arms
    # (grouping, matching_algorithm, gating, normalize)
    "raw_full_retrieval_siglip":                  (False, "dp",   False, False),
    "raw_full_retrieval_ccma":                    (False, "ccma", False, False),
    "raw_full_retrieval_siglip_grouping":         (True,  "dp",   False, False),
    "raw_full_retrieval_ccma_grouping":           (True,  "ccma", False, False),
    "raw_hybrid_retrieval_siglip_gating":         (False, "dp",   True,  False),
    "raw_hybrid_retrieval_ccma_gating":           (False, "ccma", True,  False),
    "raw_hybrid_retrieval_siglip_grouping_gating": (True,  "dp",   True,  False),
    "raw_hybrid_retrieval_ccma_grouping_gating":   (True,  "ccma", True,  False),
    
    "minmax_full_retrieval_siglip":                  (False, "dp",   False, True),
    "minmax_full_retrieval_ccma":                    (False, "ccma", False, True),
    "minmax_full_retrieval_siglip_grouping":         (True,  "dp",   False, True),
    "minmax_full_retrieval_ccma_grouping":           (True,  "ccma", False, True),
    "minmax_hybrid_retrieval_siglip_gating":         (False, "dp",   True,  True),
    "minmax_hybrid_retrieval_ccma_gating":           (False, "ccma", True,  True),
    "minmax_hybrid_retrieval_siglip_grouping_gating": (True,  "dp",   True,  True),
    "minmax_hybrid_retrieval_ccma_grouping_gating":   (True,  "ccma", True,  True),
}
import argparse

def run_ablation_sweep():
    parser = argparse.ArgumentParser(description="16-Arm Ablation Study Runner")
    parser.add_argument("--video", type=str, default="all", help="Specific video ID (e.g. review_1) or 'all' to run all 10 videos")
    args = parser.parse_args()

    # 1. Load config
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )

    # Load SigLIP model to get encoder
    siglip_model_id = config.get("models", {}).get("siglip", {}).get("model_name", "google/siglip2-so400m-patch16-naflex")
    siglip = SigLIPEncoder(vram_manager, siglip_model_id)

    # 2. Get sorted list of all 10 video IDs
    if args.video == "all":
        video_ids = [f"review_{i}" for i in range(1, 11)]
    else:
        video_ids = [args.video]
    logger.info(f"Starting 16-arm ablation sweep for videos: {video_ids}")

    # Results table
    sweep_results = []

    # Loop over videos
    for video_id in video_ids:
        logger.info(f"\n" + "="*50 + f"\nPROCESSING VIDEO: {video_id}\n" + "="*50)
        
        # Load directories
        intermediate_dir = Path("data/intermediate") / video_id
        output_dir = Path("data/output") / video_id
        
        keyframes_path = intermediate_dir / "keyframes_manifest.json"
        summary_path = intermediate_dir / "summary_script.json"
        audio_manifest_path = intermediate_dir / "audio_manifest.json"
        original_video_path = Path("data/eval_videos") / f"{video_id}.mp4"
        
        # Verify required input paths
        if not keyframes_path.exists() or not summary_path.exists() or not audio_manifest_path.exists() or not original_video_path.exists():
            logger.error(f"Missing essential pipeline files for {video_id}. Skipping.")
            continue
            
        # Load datasets
        with open(keyframes_path, "r") as f:
            manifest_data = json.load(f)
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
            
        # 3. Setup keyframe/scene representations and embeddings
        # Load cached frame embeddings
        emb_slug = siglip_model_id.replace("/", "_").replace("-", "_")
        cache_path = intermediate_dir / f"embeddings_{emb_slug}.joblib"
        if not cache_path.exists():
            logger.error(f"Missing cached SigLIP embeddings for {video_id}. Skipping.")
            continue
            
        import joblib
        frame_embeddings = joblib.load(cache_path)
        
        # Setup scenes
        p4_scenes = []
        scene_timestamps = []
        for sc in manifest_data["scenes"]:
            embs = [frame_embeddings[(sc["id"], ts)] for ts in sc["multi_frame_timestamps"]]
            if embs:
                scene_emb = np.mean(embs, axis=0)
                norm = np.linalg.norm(scene_emb)
                if norm > 0:
                    scene_emb = scene_emb / norm
            else:
                scene_emb = np.zeros(siglip.get_embedding_dim())
                
            p4_scenes.append(P4Scene(
                id=sc["id"],
                start=sc["start_seconds"],
                end=sc["end_seconds"],
                embedding=scene_emb
            ))
            scene_timestamps.append((sc["start_seconds"] + sc["end_seconds"]) / 2.0)
            
        scene_matrix = np.stack([s.embedding for s in p4_scenes], axis=0)
        scene_centers = np.array(scene_timestamps, dtype=np.float32)
        
        # Setup sentences
        sentences = [
            P4Sentence(
                id=s["id"],
                text=s["text"],
                timestamp_hint=(s["source_timestamp_hint"][0], s["source_timestamp_hint"][1])
            )
            for s in summary_data["sentences"]
        ]
        
        video_duration = max(sc.end for sc in p4_scenes)
        
        # 4. Generate greedy grouping topology
        logger.info(f"Generating grouping topology for {video_id} using SigLIP...")
        greedy_groups = greedy_grouping(sentences, scene_matrix, scene_centers, siglip, sigma=config.get("phase4", {}).get("temporal_sigma", 30.0))
        
        # PRECOMPUTE EMBEDDINGS CACHE FOR ALL SENTENCES AND GROUPS
        logger.info(f"Precomputing visual-text embeddings cache...")
        text_cache = {}
        for sent in sentences:
            if sent.text not in text_cache:
                text_cache[sent.text] = siglip.encode(sent.text)
        for g in greedy_groups:
            if g["text"] not in text_cache:
                text_cache[g["text"]] = siglip.encode(g["text"])
                
        # Unload SigLIP completely to save VRAM and prevent slow loading inside the sweep!
        vram_manager.unload_current_model()
        siglip.model = None
        siglip.processor = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # 5. Run each of the 16 arms
        for arm_name, (grouping, matching_algo, gating, normalize) in ARM_CONFIGS.items():
            logger.info(f"Running arm '{arm_name}' on {video_id}...")
            
            # Select correct topology
            if grouping:
                arm_groups = greedy_groups
            else:
                # Ungrouped: each sentence is its own group
                arm_groups = []
                for idx, sent in enumerate(sentences):
                    # Use precomputed embedding from cache!
                    sent_emb = text_cache[sent.text]
                    raw_cosine = _cosine_to_all(sent_emb, scene_matrix)
                    weights = _gaussian_temporal_weights(scene_centers, (sent.timestamp_hint[0] + sent.timestamp_hint[1]) / 2.0, 30.0)
                    weighted = raw_cosine * weights
                    locked_idx = int(np.argmax(weighted))
                    
                    arm_groups.append({
                        "sentence_ids": [idx],
                        "scene_idx": locked_idx,
                        "text": sent.text,
                        "hint_range": sent.timestamp_hint,
                        "similarity_trail": [float(weighted[locked_idx])]
                    })
            
            # Construct similarity matrix (num_groups, num_scenes) using cached text embeddings
            num_arm_groups = len(arm_groups)
            num_scenes = len(p4_scenes)
            sim_matrix = np.zeros((num_arm_groups, num_scenes))
            raw_cosine_matrix = np.zeros((num_arm_groups, num_scenes))
            temporal_weight_matrix = np.zeros((num_arm_groups, num_scenes))
            
            for k, group in enumerate(arm_groups):
                group_text = group["text"]
                group_emb = text_cache[group_text] # Use cached embedding!
                
                # Coherence to all scenes
                raw_cos = _cosine_to_all(group_emb, scene_matrix)
                
                # Temporal prior
                hint_center = sum(group["hint_range"]) / 2.0
                weights = _gaussian_temporal_weights(scene_centers, hint_center, config.get("phase4", {}).get("temporal_sigma", 30.0))
                
                raw_cosine_matrix[k] = raw_cos
                temporal_weight_matrix[k] = weights
                sim_matrix[k] = raw_cos * weights
                
            # Normalize similarities if required
            score_matrix = sim_matrix.copy()
            if normalize:
                for k in range(num_arm_groups):
                    score_matrix[k] = min_max_normalize(score_matrix[k])
                    
            # Run sequence alignment matching algorithm
            if matching_algo == "dp":
                assigned_scenes = dp_sequence_align(
                    score_matrix, scene_centers, video_duration,
                    jump_penalty=config.get("retrieval", {}).get("dp_jump_penalty", 0.01),
                    reuse_bonus=config.get("retrieval", {}).get("dp_reuse_bonus", 0.01),
                    backward_penalty=config.get("retrieval", {}).get("dp_backward_penalty", 0.5)
                )
            elif matching_algo == "ccma":
                assigned_scenes = ccma_align_sequence(
                    score_matrix, scene_centers, video_duration,
                    c_max=config.get("retrieval", {}).get("ccma_c_max", 3),
                    reuse_penalty=config.get("retrieval", {}).get("ccma_reuse_penalty", 0.2),
                    jump_penalty=config.get("retrieval", {}).get("dp_jump_penalty", 0.01),
                    backward_penalty=config.get("retrieval", {}).get("dp_backward_penalty", 0.5)
                )
                
            # Build assignments list
            assignments_list = []
            for k, group in enumerate(arm_groups):
                scene_idx = assigned_scenes[k]
                score = float(score_matrix[k, scene_idx])
                raw_cos = float(raw_cosine_matrix[k, scene_idx])
                weight = float(temporal_weight_matrix[k, scene_idx])
                
                # PATCH (Bug #2): Gating decision MUST use raw weighted cosine, NOT normalized score.
                # Min-max normalization per-group destroys the absolute signal needed for gating,
                # because the best-match score always becomes 1.0 after normalization.
                raw_weighted = raw_cos * weight  # equivalent to sim_matrix[k, scene_idx] before normalization
                
                if gating:
                    action = "retrieve" if raw_weighted >= 0.12 else "generate"
                else:
                    action = "retrieve"
                    
                assignments_list.append({
                    "sentence_ids": group["sentence_ids"],
                    "scene_id": p4_scenes[scene_idx].id,
                    "best_similarity": score,
                    "raw_cosine": raw_cos,
                    "temporal_weight": weight,
                    "action": action,
                    "timestamp_hint_merged": [float(group["hint_range"][0]), float(group["hint_range"][1])],
                    "similarity_trail": group.get("similarity_trail", [score])
                })
                
            # Save arm-specific assignments JSON
            out_assignments = {
                "retrieval_method": arm_name,
                "groups": assignments_list
            }
            
            assignments_file = intermediate_dir / f"scene_matches_{arm_name}.json"
            with open(assignments_file, "w") as f:
                json.dump(out_assignments, f, indent=2)
                
            logger.info(f"Saved retrieval assignments to: {assignments_file}")
            
            # 6. Assemble final video summary
            try:
                logger.info(f"Stitching video summary_{arm_name}.mp4...")
                p5 = Phase5Assembler(config, vram_manager)
                p5.run(
                    original_video_path,
                    audio_manifest_path,
                    keyframes_path,
                    assignments_file
                )
                logger.info(f"Video assembly success: summary_{arm_name}.mp4")
            except Exception as ae:
                logger.error(f"Video assembly failed for {video_id} arm {arm_name}: {ae}")
                traceback.print_exc()
                
            # 7. Evaluate generated summary video
            try:
                logger.info(f"Evaluating generated video for {video_id} arm {arm_name} using unified evaluation...")
                # We launch the evaluation in a separate subprocess to guarantee clean GPU memory release after execution
                cmd = [
                    sys.executable,
                    "-m", "src.eval.unified_evaluation",
                    "--video", video_id,
                    "--arm", arm_name
                ]
                subprocess.run(cmd, check=True)
                
                # Load evaluated metrics CSV
                csv_path = Path("data/evaluation") / f"unified_eval_{arm_name}.csv"
                if csv_path.exists():
                    with open(csv_path, "r", encoding="utf-8") as csvf:
                        reader = csv.DictReader(csvf)
                        for r in reader:
                            if r["video_id"] == video_id:
                                sweep_results.append(r)
                                break
            except Exception as ee:
                logger.error(f"Unified evaluation failed for {video_id} arm {arm_name}: {ee}")
                traceback.print_exc()
                
        logger.info(f"Completed all 16 arms for {video_id}.")
        
    # 8. Output final aggregated results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    aggregate_csv = results_dir / "final_ablation_results.csv"
    
    if sweep_results:
        # Write to aggregate CSV
        fieldnames = list(sweep_results[0].keys())
        with open(aggregate_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sweep_results)
            
        logger.info(f"\n" + "="*80 + f"\nSWEEP COMPLETE! All aggregated ablation results saved to: {aggregate_csv}\n" + "="*80)
        
        # Display Markdown Aggregate Table
        print("\n### Aggregated Ablation Study Performance Table")
        print("| Video ID | Arm | CLIPScore | BLIPScore | SceneDiversity | MaxConsecutiveReuse | TempAcc@15s |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for r in sweep_results:
            c_score = f"{float(r['clipscore_mean']):.4f}" if r['clipscore_mean'] != 'NaN' else 'NaN'
            b_score = f"{float(r['blipscore_mean']):.4f}" if r['blipscore_mean'] != 'NaN' else 'NaN'
            div = f"{float(r['scene_diversity']):.4f}" if r['scene_diversity'] != 'NaN' else 'NaN'
            reuse = f"{r['max_consecutive_reuse']}" if r['max_consecutive_reuse'] != 'NaN' else 'NaN'
            temp = f"{float(r['temporal_accuracy_15s']):.4f}" if r['temporal_accuracy_15s'] != 'NaN' else 'NaN'
            print(f"| {r['video_id']} | {r['arm']} | {c_score} | {b_score} | {div} | {reuse} | {temp} |")
        print("\n")
    else:
        logger.warning("Sweep complete, but no evaluation results were aggregated successfully.")

if __name__ == "__main__":
    run_ablation_sweep()

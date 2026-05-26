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
import gc
import joblib

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.vram import VRAMManager
from src.models.siglip import SigLIPEncoder
from src.phase4_retrieve import Sentence as P4Sentence, Scene as P4Scene
from src.phase5_assemble import Phase5Assembler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gating_threshold_investigation")

# --- Helper Math & Alignment Functions ---

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

import argparse

def run_investigation():
    parser = argparse.ArgumentParser(description="Gating Decision Gating Threshold Investigation Sweep")
    parser.add_argument("--video", type=str, default="all", help="Specific video ID (e.g. review_1) or 'all'")
    parser.add_argument("--run-eval", action="store_true", help="If set, physically run video stitching and VLM evaluation")
    parser.add_argument("--eval-thresholds", type=str, default="static_0.08,static_0.12,static_0.16,z_0.5,z_1.0,z_1.5",
                        help="Comma-separated configurations to evaluate if --run-eval is set")
    args = parser.parse_args()

    # 1. Load default config
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )

    siglip_model_id = config.get("models", {}).get("siglip", {}).get("model_name", "google/siglip2-so400m-patch16-naflex")
    siglip = SigLIPEncoder(vram_manager, siglip_model_id)

    # Sorted list of videos to sweep
    if args.video == "all":
        video_ids = [f"review_{i}" for i in range(1, 11)]
    else:
        video_ids = [args.video]

    logger.info(f"Initiating mathematical sweep profiling for videos: {video_ids}")

    # Threshold values to sweep
    static_thresholds = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]
    z_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]

    # Pre-select matching architectures to investigate
    architectures = [
        # (grouping, matching_algo, name)
        (True, "ccma", "ccma_grouping"),
        (True, "dp", "siglip_grouping"),
        (False, "ccma", "ccma_ungrouped"),
        (False, "dp", "siglip_ungrouped")
    ]

    sweep_rows = []

    # Cache for video representations to avoid redundant files load
    video_data_cache = {}

    for video_id in video_ids:
        logger.info(f"Loading and pre-computing embeddings for: {video_id}")
        
        intermediate_dir = Path("data/intermediate") / video_id
        keyframes_path = intermediate_dir / "keyframes_manifest.json"
        summary_path = intermediate_dir / "summary_script.json"
        audio_manifest_path = intermediate_dir / "audio_manifest.json"
        original_video_path = Path("data/eval_videos") / f"{video_id}.mp4"
        
        if not keyframes_path.exists() or not summary_path.exists() or not audio_manifest_path.exists() or not original_video_path.exists():
            logger.warning(f"Skipping {video_id}: Manifest files not found.")
            continue
            
        with open(keyframes_path, "r") as f:
            manifest_data = json.load(f)
        with open(summary_path, "r") as f:
            summary_data = json.load(f)

        emb_slug = siglip_model_id.replace("/", "_").replace("-", "_")
        cache_path = intermediate_dir / f"embeddings_{emb_slug}.joblib"
        if not cache_path.exists():
            logger.warning(f"Skipping {video_id}: No cached SigLIP embeddings found.")
            continue
            
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
                
            p4_scenes.append(P4Scene(id=sc["id"], start=sc["start_seconds"], end=sc["end_seconds"], embedding=scene_emb))
            scene_timestamps.append((sc["start_seconds"] + sc["end_seconds"]) / 2.0)
            
        scene_matrix = np.stack([s.embedding for s in p4_scenes], axis=0)
        scene_centers = np.array(scene_timestamps, dtype=np.float32)
        
        # Setup sentences
        sentences = [
            P4Sentence(id=s["id"], text=s["text"], timestamp_hint=(s["source_timestamp_hint"][0], s["source_timestamp_hint"][1]))
            for s in summary_data["sentences"]
        ]
        
        video_duration = max(sc.end for sc in p4_scenes)
        
        # Grouping topology
        greedy_groups = greedy_grouping(sentences, scene_matrix, scene_centers, siglip, sigma=config.get("phase4", {}).get("temporal_sigma", 30.0))
        
        # Text embeddings
        text_cache = {}
        for sent in sentences:
            if sent.text not in text_cache:
                text_cache[sent.text] = siglip.encode(sent.text)
        for g in greedy_groups:
            if g["text"] not in text_cache:
                text_cache[g["text"]] = siglip.encode(g["text"])

        video_data_cache[video_id] = {
            "p4_scenes": p4_scenes,
            "scene_matrix": scene_matrix,
            "scene_centers": scene_centers,
            "sentences": sentences,
            "video_duration": video_duration,
            "greedy_groups": greedy_groups,
            "text_cache": text_cache,
            "original_video_path": original_video_path,
            "audio_manifest_path": audio_manifest_path,
            "keyframes_path": keyframes_path,
            "intermediate_dir": intermediate_dir
        }

    # Unload SigLIP completely to release CUDA VRAM
    vram_manager.unload_current_model()
    siglip.model = None
    siglip.processor = None
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("SigLIP VRAM unloaded successfully. Executing fast CPU sweep...")

    # Perform mathematical sweep
    for video_id in video_ids:
        if video_id not in video_data_cache:
            continue
        vdata = video_data_cache[video_id]
        
        p4_scenes = vdata["p4_scenes"]
        scene_matrix = vdata["scene_matrix"]
        scene_centers = vdata["scene_centers"]
        sentences = vdata["sentences"]
        video_duration = vdata["video_duration"]
        greedy_groups = vdata["greedy_groups"]
        text_cache = vdata["text_cache"]

        for grouping, matching_algo, arch_name in architectures:
            # Build topology
            if grouping:
                arm_groups = greedy_groups
            else:
                arm_groups = []
                for idx, sent in enumerate(sentences):
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

            for normalize in [False, True]:
                # Similarity matrix construction
                num_groups = len(arm_groups)
                num_scenes = len(p4_scenes)
                sim_matrix = np.zeros((num_groups, num_scenes))
                raw_cosine_matrix = np.zeros((num_groups, num_scenes))
                temporal_weight_matrix = np.zeros((num_groups, num_scenes))
                
                for k, group in enumerate(arm_groups):
                    group_text = group["text"]
                    group_emb = text_cache[group_text]
                    raw_cos = _cosine_to_all(group_emb, scene_matrix)
                    hint_center = sum(group["hint_range"]) / 2.0
                    weights = _gaussian_temporal_weights(scene_centers, hint_center, config.get("phase4", {}).get("temporal_sigma", 30.0))
                    
                    raw_cosine_matrix[k] = raw_cos
                    temporal_weight_matrix[k] = weights
                    sim_matrix[k] = raw_cos * weights

                # Normalize similarities if required
                score_matrix = sim_matrix.copy()
                if normalize:
                    for r_idx in range(num_groups):
                        s_min = score_matrix[r_idx].min()
                        s_max = score_matrix[r_idx].max()
                        if s_max - s_min < 1e-6:
                            score_matrix[r_idx] = np.ones_like(score_matrix[r_idx]) * 0.5
                        else:
                            score_matrix[r_idx] = (score_matrix[r_idx] - s_min) / (s_max - s_min)

                # Align sequence
                if matching_algo == "dp":
                    assigned_scenes = dp_sequence_align(score_matrix, scene_centers, video_duration)
                else:
                    assigned_scenes = ccma_align_sequence(
                        score_matrix, scene_centers, video_duration,
                        c_max=config.get("retrieval", {}).get("ccma_c_max", 3),
                        reuse_penalty=config.get("retrieval", {}).get("ccma_reuse_penalty", 0.2)
                    )

                # Match details
                matches_sims = []
                for k in range(num_groups):
                    scene_idx = assigned_scenes[k]
                    matches_sims.append(float(score_matrix[k, scene_idx]))

                matches_sims = np.array(matches_sims)
                mean_sim = float(np.mean(matches_sims))
                std_sim = float(np.std(matches_sims))

                # 1. Sweep Static Thresholds
                for t in static_thresholds:
                    retrieved_count = int(np.sum(matches_sims >= t))
                    generated_count = num_groups - retrieved_count
                    ret_sims = matches_sims[matches_sims >= t]
                    gen_sims = matches_sims[matches_sims < t]
                    avg_ret_sim = float(np.mean(ret_sims)) if len(ret_sims) > 0 else 0.0
                    avg_gen_sim = float(np.mean(gen_sims)) if len(gen_sims) > 0 else 0.0

                    sweep_rows.append({
                        "video_id": video_id,
                        "architecture": arch_name,
                        "normalization": "minmax" if normalize else "raw",
                        "threshold_type": "static",
                        "threshold_param": t,
                        "computed_threshold": t,
                        "total_groups": num_groups,
                        "retrieve_count": retrieved_count,
                        "generate_count": generated_count,
                        "retrieve_percent": float(retrieved_count / num_groups),
                        "mean_sim_all": mean_sim,
                        "std_sim_all": std_sim,
                        "avg_retrieved_sim": avg_ret_sim,
                        "avg_generated_sim": avg_gen_sim
                    })

                # 2. Sweep Dynamic Z-Score Thresholds (Threshold = mean - k * std)
                for k_z in z_multipliers:
                    t_dyn = mean_sim - k_z * std_sim
                    # Guarantee a sane lower/upper bound for threshold
                    t_dyn = max(0.01, min(t_dyn, 0.99))
                    retrieved_count = int(np.sum(matches_sims >= t_dyn))
                    generated_count = num_groups - retrieved_count
                    ret_sims = matches_sims[matches_sims >= t_dyn]
                    gen_sims = matches_sims[matches_sims < t_dyn]
                    avg_ret_sim = float(np.mean(ret_sims)) if len(ret_sims) > 0 else 0.0
                    avg_gen_sim = float(np.mean(gen_sims)) if len(gen_sims) > 0 else 0.0

                    sweep_rows.append({
                        "video_id": video_id,
                        "architecture": arch_name,
                        "normalization": "minmax" if normalize else "raw",
                        "threshold_type": "zscore",
                        "threshold_param": k_z,
                        "computed_threshold": t_dyn,
                        "total_groups": num_groups,
                        "retrieve_count": retrieved_count,
                        "generate_count": generated_count,
                        "retrieve_percent": float(retrieved_count / num_groups),
                        "mean_sim_all": mean_sim,
                        "std_sim_all": std_sim,
                        "avg_retrieved_sim": avg_ret_sim,
                        "avg_generated_sim": avg_gen_sim
                    })

    # Save to CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv_path = results_dir / "gating_threshold_sweep.csv"
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
        writer.writeheader()
        writer.writerows(sweep_rows)

    logger.info(f"Fast mathematical profiling sweep complete! Saved to {sweep_csv_path}")

    # Display Macro statistics
    print("\n### GATING THRESHOLD SWEEP SUMMARY (Macro Average across Videos)")
    print("| Architecture | Normalization | Threshold Type | Param | Computed Threshold (Mean) | Retrieve % | Retrieve Count | Generate Count | Avg Retrieved Sim |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    
    unique_confs = set((r["architecture"], r["normalization"], r["threshold_type"], r["threshold_param"]) for r in sweep_rows)
    sorted_confs = sorted(list(unique_confs), key=lambda x: (x[0], x[1], x[2], x[3]))

    for arch, norm, t_type, param in sorted_confs:
        rows = [r for r in sweep_rows if r["architecture"] == arch and r["normalization"] == norm and r["threshold_type"] == t_type and r["threshold_param"] == param]
        avg_comp = float(np.mean([r["computed_threshold"] for r in rows]))
        avg_pct = float(np.mean([r["retrieve_percent"] for r in rows]))
        avg_ret_cnt = float(np.mean([r["retrieve_count"] for r in rows]))
        avg_gen_cnt = float(np.mean([r["generate_count"] for r in rows]))
        avg_ret_sim = float(np.mean([r["avg_retrieved_sim"] for r in rows]))
        print(f"| {arch} | {norm} | {t_type} | {param} | {avg_comp:.4f} | {avg_pct*100:.1f}% | {avg_ret_cnt:.1f} | {avg_gen_cnt:.1f} | {avg_ret_sim:.4f} |")

    # --- Step 5: Physical video generation and VLM evaluation (if --run-eval is requested) ---
    if args.run_eval:
        logger.info("\n" + "="*80 + "\nINITIATING PHYSICAL EVALUATION SWEEP (--run-eval is active)\n" + "="*80)
        
        eval_confs = [c.strip() for c in args.eval_thresholds.split(",")]
        eval_results = []

        # We will run this on the selected videos
        for video_id in video_ids:
            if video_id not in video_data_cache:
                continue
            vdata = video_data_cache[video_id]
            p4_scenes = vdata["p4_scenes"]
            scene_matrix = vdata["scene_matrix"]
            scene_centers = vdata["scene_centers"]
            sentences = vdata["sentences"]
            video_duration = vdata["video_duration"]
            greedy_groups = vdata["greedy_groups"]
            text_cache = vdata["text_cache"]
            original_video_path = vdata["original_video_path"]
            audio_manifest_path = vdata["audio_manifest_path"]
            keyframes_path = vdata["keyframes_path"]
            intermediate_dir = vdata["intermediate_dir"]

            # We use the user's primary/favored thesis architecture: ccma_grouping (Grouped CCMA Gating)
            grouping = True
            matching_algo = "ccma"
            
            # Recalculate similarities for alignment
            num_groups = len(greedy_groups)
            num_scenes = len(p4_scenes)
            sim_matrix = np.zeros((num_groups, num_scenes))
            raw_cosine_matrix = np.zeros((num_groups, num_scenes))
            temporal_weight_matrix = np.zeros((num_groups, num_scenes))
            
            for k, group in enumerate(greedy_groups):
                group_text = group["text"]
                group_emb = text_cache[group_text]
                raw_cos = _cosine_to_all(group_emb, scene_matrix)
                hint_center = sum(group["hint_range"]) / 2.0
                weights = _gaussian_temporal_weights(scene_centers, hint_center, config.get("phase4", {}).get("temporal_sigma", 30.0))
                
                raw_cosine_matrix[k] = raw_cos
                temporal_weight_matrix[k] = weights
                sim_matrix[k] = raw_cos * weights

            assigned_scenes = ccma_align_sequence(
                sim_matrix, scene_centers, video_duration,
                c_max=config.get("retrieval", {}).get("ccma_c_max", 3),
                reuse_penalty=config.get("retrieval", {}).get("ccma_reuse_penalty", 0.2)
            )

            # Compute similarities of aligned groups
            aligned_similarities = []
            for k in range(num_groups):
                scene_idx = assigned_scenes[k]
                aligned_similarities.append(float(sim_matrix[k, scene_idx]))
            
            aligned_similarities = np.array(aligned_similarities)
            mean_sim = float(np.mean(aligned_similarities))
            std_sim = float(np.std(aligned_similarities))

            for conf in eval_confs:
                # Parse configuration
                if conf.startswith("static_"):
                    t_val = float(conf.split("_")[1])
                    t_name = f"static_{t_val:.2f}"
                elif conf.startswith("z_"):
                    k_z = float(conf.split("_")[1])
                    t_val = mean_sim - k_z * std_sim
                    t_val = max(0.01, min(t_val, 0.99))
                    t_name = f"z_{k_z:.2f}"
                else:
                    logger.warning(f"Unknown threshold format: {conf}. Skipping.")
                    continue

                arm_name = f"investigation_threshold_{t_name}"
                logger.info(f"\nEvaluating: {video_id} | Gating Threshold: {t_name} (computed = {t_val:.4f})")

                # Build assignments
                assignments_list = []
                for k, group in enumerate(greedy_groups):
                    scene_idx = assigned_scenes[k]
                    raw_cos = float(raw_cosine_matrix[k, scene_idx])
                    weight = float(temporal_weight_matrix[k, scene_idx])
                    
                    action = "retrieve" if aligned_similarities[k] >= t_val else "generate"
                    
                    assignments_list.append({
                        "sentence_ids": group["sentence_ids"],
                        "scene_id": p4_scenes[scene_idx].id,
                        "best_similarity": float(sim_matrix[k, scene_idx]),
                        "raw_cosine": raw_cos,
                        "temporal_weight": weight,
                        "action": action,
                        "timestamp_hint_merged": [float(group["hint_range"][0]), float(group["hint_range"][1])],
                        "similarity_trail": group.get("similarity_trail", [float(sim_matrix[k, scene_idx])])
                    })

                # Save JSON
                out_assignments = {
                    "retrieval_method": arm_name,
                    "groups": assignments_list
                }
                assignments_file = intermediate_dir / f"scene_matches_{arm_name}.json"
                with open(assignments_file, "w") as f:
                    json.dump(out_assignments, f, indent=2)

                # Assemble Video Summary
                try:
                    logger.info(f"Assembling video summary for {arm_name}...")
                    p5 = Phase5Assembler(config, vram_manager)
                    p5.run(original_video_path, audio_manifest_path, keyframes_path, assignments_file)
                    
                    # Unload to let subprocess load evaluation model on GPU
                    vram_manager.unload_current_model()
                    
                    # Run subprocess evaluation
                    logger.info(f"Running subprocess unified evaluation for {arm_name}...")
                    cmd = [
                        sys.executable,
                        "-m", "src.eval.unified_evaluation",
                        "--video", video_id,
                        "--arm", arm_name
                    ]
                    subprocess.run(cmd, check=True)
                    
                    # Load metrics
                    csv_path = Path("data/evaluation") / f"unified_eval_{arm_name}.csv"
                    if csv_path.exists():
                        with open(csv_path, "r", encoding="utf-8") as csvf:
                            reader = csv.DictReader(csvf)
                            for r in reader:
                                if r["video_id"] == video_id:
                                    r["threshold_name"] = t_name
                                    r["threshold_value"] = f"{t_val:.4f}"
                                    eval_results.append(r)
                                    break
                except Exception as ex:
                    logger.error(f"Assembly or evaluation failed for {video_id} at {t_name}: {ex}")
                    traceback.print_exc()

        # Output final evaluated results
        eval_csv_path = results_dir / "gating_threshold_evaluation_results.csv"
        if eval_results:
            with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=eval_results[0].keys())
                writer.writeheader()
                writer.writerows(eval_results)
            
            logger.info(f"\n" + "="*80 + f"\nEVALUATION COMPLETE! Metric results saved to: {eval_csv_path}\n" + "="*80)
            
            # Print beautiful evaluation comparison table
            print("\n### GATING THRESHOLD VISUAL EVALUATION RESULTS")
            print("| Video ID | Gating Config | Threshold Value | CLIPScore | BLIPScore | Qwen-VL Judge Quality | Scene Diversity | TempAcc@15s |")
            print("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in eval_results:
                c_score = f"{float(r['clipscore_mean']):.4f}" if r['clipscore_mean'] != 'NaN' else 'NaN'
                b_score = f"{float(r['blipscore_mean']):.4f}" if r['blipscore_mean'] != 'NaN' else 'NaN'
                q_score = f"{float(r['llm_judge_quality']):.4f}" if 'llm_judge_quality' in r and r['llm_judge_quality'] != 'NaN' else 'NaN'
                div = f"{float(r['scene_diversity']):.4f}" if r['scene_diversity'] != 'NaN' else 'NaN'
                temp = f"{float(r['temporal_accuracy_15s']):.4f}" if r['temporal_accuracy_15s'] != 'NaN' else 'NaN'
                print(f"| {r['video_id']} | {r['threshold_name']} | {r['threshold_value']} | {c_score} | {b_score} | {q_score} | {div} | {temp} |")
            print("\n")
        else:
            logger.warning("No evaluation results were recorded.")

if __name__ == "__main__":
    run_investigation()

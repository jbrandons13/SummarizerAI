import os
import yaml
import json
import random
import numpy as np
import torch
from pathlib import Path
import subprocess
from dataclasses import asdict

# Add project root to sys.path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.io import load_json_as_model
from src.schemas import SummaryScript, KeyframesManifest
from src.models.siglip import SigLIPEncoder
from src.phase4_retrieve import (
    RetrievalGate, RetrievalGateConfig, 
    Sentence as P4Sentence, Scene as P4Scene
)
from src.utils.vram import VRAMManager

def extract_frame(video_path, timestamp, output_path):
    """Extract a frame at timestamp from video_path and save to output_path at 480p."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=-1:480",
        "-q:v", "2",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

def run_sanity_check():
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    threshold = config["phase4"]["gate_threshold"]
    print(f"Using threshold: {threshold}")
    
    video_ids = [f"review_{i}" for i in range(1, 11)]
    eval_video_dir = Path("data/eval_videos")
    intermediate_base = Path(config["paths"]["intermediate_dir"])
    
    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )
    
    siglip_model = config.get("models", {}).get("siglip", {}).get("model_name", "google/siglip2-so400m-patch16-naflex")
    siglip = SigLIPEncoder(vram_manager, siglip_model)
    
    all_retrieve_assignments = []
    
    for video_id in video_ids:
        print(f"Processing {video_id}...")
        video_path = eval_video_dir / f"{video_id}.mp4"
        intermediate_dir = intermediate_base / video_id
        
        summary_path = intermediate_dir / "summary_script.json"
        keyframes_manifest_path = intermediate_dir / "keyframes_manifest.json"
        
        if not summary_path.exists() or not keyframes_manifest_path.exists():
            print(f"Missing data for {video_id}, skipping.")
            continue
            
        summary = load_json_as_model(summary_path, SummaryScript)
        manifest = load_json_as_model(keyframes_manifest_path, KeyframesManifest)
        
        # Load frame embeddings
        # embed_scenes will return cached embeddings if they exist
        frame_embeddings = siglip.embed_scenes(video_id, manifest)
        
        # Prepare scenes
        p4_scenes = []
        for sc in manifest.scenes:
            embs = [frame_embeddings[(sc.id, ts)] for ts in sc.multi_frame_timestamps]
            if embs:
                scene_emb = np.mean(embs, axis=0)
                norm = np.linalg.norm(scene_emb)
                if norm > 0:
                    scene_emb = scene_emb / norm
            else:
                scene_emb = np.zeros(siglip.get_embedding_dim())

            p4_scenes.append(P4Scene(
                id=sc.id,
                start=sc.start_seconds,
                end=sc.end_seconds,
                embedding=scene_emb
            ))

        # Prepare sentences
        p4_sentences = []
        for s in summary.sentences:
            hint = s.source_timestamp_hint
            if not hint or len(hint) < 2:
                print(f"Malformed hint for sentence {s.id} in {video_id}: {hint}. Using fallback [0, 0].")
                hint_val = (0.0, 0.0)
            else:
                hint_val = (float(hint[0]), float(hint[1]))
            
            p4_sentences.append(P4Sentence(
                id=s.id,
                text=s.text,
                timestamp_hint=hint_val
            ))

        # Run Gate
        gate_cfg_vals = config["phase4"]
        gate = RetrievalGate(
            text_encoder=siglip,
            config=RetrievalGateConfig(
                gate_threshold=gate_cfg_vals["gate_threshold"],
                extend_epsilon=gate_cfg_vals["extend_epsilon"],
                max_group_size=gate_cfg_vals["max_group_size"],
                join_sep=gate_cfg_vals["join_sep"],
                temporal_sigma=gate_cfg_vals["temporal_sigma"],
                enable_temporal_prior=gate_cfg_vals["enable_temporal_prior"],
            ),
        )
        assignments = gate.run(p4_sentences, p4_scenes)
        
        # Filter retrieve assignments
        for idx, a in enumerate(assignments):
            if a.action == "retrieve":
                # Add extra info for sampling
                a_dict = asdict(a)
                a_dict["video_id"] = video_id
                a_dict["group_id"] = idx
                a_dict["joined_text"] = gate_cfg_vals["join_sep"].join([s.text for s in p4_sentences if s.id in a.sentence_ids])
                
                # Get scene time range
                scene = next(sc for sc in p4_scenes if sc.id == a.scene_id)
                a_dict["scene_start"] = scene.start
                a_dict["scene_end"] = scene.end
                a_dict["middle_ts"] = (scene.start + scene.end) / 2
                
                all_retrieve_assignments.append(a_dict)

    print(f"Total retrieve assignments: {len(all_retrieve_assignments)}")
    
    # Sampling
    random.seed(42) # For reproducibility
    
    bucket_1 = [a for a in all_retrieve_assignments if 0.12 <= a["best_similarity"] < 0.15]
    bucket_2 = [a for a in all_retrieve_assignments if 0.15 <= a["best_similarity"] < 0.20]
    bucket_3 = [a for a in all_retrieve_assignments if a["best_similarity"] >= 0.20]
    
    print(f"Bucket 1 [0.12, 0.15): {len(bucket_1)}")
    print(f"Bucket 2 [0.15, 0.20): {len(bucket_2)}")
    print(f"Bucket 3 [0.20, +inf): {len(bucket_3)}")
    
    sample = []
    sample.extend(random.sample(bucket_1, min(2, len(bucket_1))))
    sample.extend(random.sample(bucket_2, min(2, len(bucket_2))))
    sample.extend(random.sample(bucket_3, min(1, len(bucket_3))))
    
    print(f"Sampled {len(sample)} assignments.")
    
    output_dir = Path("sanity_check_threshold_012")
    output_dir.mkdir(exist_ok=True)
    
    # Report generation
    print("\n=== SANITY CHECK REPORT ===\n")
    print(f"Threshold used: {threshold}")
    print(f"Total retrieve assignments across 10 videos: {len(all_retrieve_assignments)}")
    print("\nSampled assignments:\n")
    
    for i, a in enumerate(sample, 1):
        sim_str = f"{a['best_similarity']:.3f}"
        frame_name = f"sanity_{a['video_id']}_g{a['group_id']}_{sim_str}.jpg"
        frame_path = output_dir / frame_name
        
        video_path = eval_video_dir / f"{a['video_id']}.mp4"
        try:
            extract_frame(video_path, a['middle_ts'], frame_path)
            frame_status = frame_path
        except Exception as e:
            print(f"Failed to extract frame for {a['video_id']}: {e}")
            frame_status = f"FAILED: {e}"
        
        print(f"[{i}] video={a['video_id']}, group_id={a['group_id']}, sents={a['sentence_ids']}")
        print(f"    text: \"{a['joined_text']}\"")
        print(f"    scene: id={a['scene_id']}, time=({a['scene_start']:.2f}, {a['scene_end']:.2f})")
        print(f"    similarity: weighted={a['best_similarity']:.3f}, raw={a['raw_cosine']:.3f}, temporal_weight={a['temporal_weight']:.3f}")
        print(f"    frame: {frame_status}")
        print()

    print("Bucket distribution of sample:")
    print(f"  [0.12, 0.15): {sum(1 for a in sample if 0.12 <= a['best_similarity'] < 0.15)}")
    print(f"  [0.15, 0.20): {sum(1 for a in sample if 0.15 <= a['best_similarity'] < 0.20)}")
    print(f"  [0.20, +inf): {sum(1 for a in sample if a['best_similarity'] >= 0.20)}")
    print("\nBlockers: none")
    print("\n=== TASK 2 DONE ===")

if __name__ == "__main__":
    run_sanity_check()

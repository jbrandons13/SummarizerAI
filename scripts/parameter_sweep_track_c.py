import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib

# Add current directory to path so we can import src
import sys
sys.path.append(os.getcwd())

from src.phase4_retrieve import SigLIP2DirectRetrieval, CaptionCosineRetrieval, KeyframesManifest
from src.schemas import SummaryScript, RetrievalOutput, SceneMatch
from src.eval.metrics import temporal_alignment_score, visual_coherence_score, compute_clipscore_batch
from src.utils.io import load_json_as_model

# Mock VRAMManager for script
class MockVRAM:
    def load_model(self, name, loader):
        # We assume models are either not needed (cached) or we just load them
        return loader()

def get_reuse_rate(matches):
    scene_ids = [m.matched_scene_id for m in matches]
    unique_scenes = len(set(scene_ids))
    num_sentences = len(matches)
    return 1 - (unique_scenes / num_sentences), unique_scenes, scene_ids

def run_sweep():
    config = {
        "retrieval": {
            "dp_jump_penalty": 0.01,
            "dp_reuse_bonus": 0.01,
            "use_temporal_guidance": True,
            "temporal_weight": 0.3,
            "temporal_sigma": 30.0,
            "matching_algorithm": "dp"
        },
        "keyframe_extraction": {
            "top_k": 2,
            "frames_per_scene_caption": 3,
            "frames_per_scene_siglip": 5
        },
        "paths": {
            "intermediate_dir": "data/intermediate"
        }
    }
    
    vram = MockVRAM()
    
    bp_values = [0.05, 0.1, 0.2, 0.3, 0.5] # Added 0.5 as baseline reference
    videos = ["review_7", "review_2"]
    
    results = []
    
    for video_id in videos:
        intermediate_dir = Path(f"data/intermediate/{video_id}")
        if not (intermediate_dir / "keyframes_manifest.json").exists():
            print(f"ERROR: {video_id} manifest missing.")
            continue
            
        manifest = load_json_as_model(intermediate_dir / "keyframes_manifest.json", KeyframesManifest)
        summary = load_json_as_model(intermediate_dir / "summary_script.json", SummaryScript)
        
        # Determine arms to run for this video based on task
        if video_id == "review_7":
            arms = ["caption_temporal_dp"]
        else:
            arms = ["siglip_temporal_dp"]
            
        for arm_name in arms:
            print(f"\n>>> Video: {video_id}, Arm: {arm_name}")
            
            # Setup retriever
            if "siglip" in arm_name:
                retriever = SigLIP2DirectRetrieval(config, vram)
            else:
                retriever = CaptionCosineRetrieval(config, vram)
                
            # Let's modify the config for each BP
            for bp in bp_values:
                config["retrieval"]["dp_backward_penalty"] = bp
                
                # Run retrieval (will use cache for embeddings)
                try:
                    output = retriever.retrieve(summary, manifest, method_name=f"{arm_name}_bp{bp}")
                except Exception as e:
                    print(f"FAILED {arm_name} BP={bp}: {e}")
                    continue
                
                # Calculate metrics
                matches = output.matches
                reuse_rate, unique_scenes, scene_ids = get_reuse_rate(matches)
                
                temp_metrics = temporal_alignment_score(matches, summary, manifest)
                temp_acc = temp_metrics["temporal_accuracy_within_15s"]
                
                # VisCoher
                model_slug = "google_siglip2_so400m_patch16_naflex"
                cache_path = intermediate_dir / f"embeddings_{model_slug}.joblib"
                if cache_path.exists():
                    frame_embs = joblib.load(cache_path)
                    coherence = visual_coherence_score(matches, frame_embs)
                    vis_coher = coherence["visual_coherence_mean"]
                else:
                    vis_coher = 0.0
                
                # CLIPScore
                image_paths = []
                texts = []
                for m in matches:
                    image_paths.append(str(intermediate_dir / m.best_frame_path))
                    texts.append(summary.sentences[m.sentence_id].text)
                
                clip_results = compute_clipscore_batch(image_paths, texts)
                clip_score = clip_results["clipscore_mean"]
                
                print(f"BP={bp:.2f} | Reuse={reuse_rate:.3f} ({unique_scenes}/{len(matches)}) | TempAcc={temp_acc:.3f} | VisCoher={vis_coher:.3f} | CLIP={clip_score:.3f}")
                
                results.append({
                    "video": video_id,
                    "arm": arm_name,
                    "backward_penalty": bp,
                    "assignment": str(scene_ids),
                    "unique_scenes": unique_scenes,
                    "num_sentences": len(matches),
                    "reuse_rate": reuse_rate,
                    "temp_acc_15s": temp_acc,
                    "vis_coher": vis_coher,
                    "clip_score": clip_score
                })

    df = pd.DataFrame(results)
    df.to_csv("notes/parameter_sweep_results.csv", index=False)
    
    # Generate Markdown Table for notes/parameter_sweep_pilot.md
    print("\nGenerating report...")
    
    report = "# Parameter Sweep Pilot Results (Track C)\n\n"
    report += "| Video | Arm | BP | Assignment | Unique | Reuse Rate | TempAcc (15s) | VisCoher | CLIPScore |\n"
    report += "|-------|-----|----|------------|--------|------------|--------------|----------|-----------|\n"
    
    for _, row in df.iterrows():
        report += f"| {row['video']} | {row['arm']} | {row['backward_penalty']} | {row['assignment']} | {row['unique_scenes']} | {row['reuse_rate']:.3f} | {row['temp_acc_15s']:.3f} | {row['vis_coher']:.3f} | {row['clip_score']:.3f} |\n"
    
    # Check criteria
    # BP* is the value to pick
    
    # review_7 Caption DP reuse rate ≤ 1/6 (approx 0.167)
    # TempAcc on review_7 Caption DP does not drop more than 0.2 from baseline (0.667 -> 0.467)
    # review_2 SigLIP DP reuse rate stays ≤ 1/6
    
    baseline_temp_acc = 0.667
    
    success_bp = None
    for bp in bp_values:
        r7_row = df[(df['video'] == 'review_7') & (df['backward_penalty'] == bp)]
        r2_row = df[(df['video'] == 'review_2') & (df['backward_penalty'] == bp)]
        
        if r7_row.empty or r2_row.empty:
            continue
            
        r7_reuse = r7_row['reuse_rate'].values[0]
        r7_temp = r7_row['temp_acc_15s'].values[0]
        r2_reuse = r2_row['reuse_rate'].values[0]
        
        cond1 = r7_reuse <= (1/6 + 0.001)
        cond2 = r7_temp >= (baseline_temp_acc - 0.2)
        cond3 = r2_reuse <= (1/6 + 0.001)
        
        if cond1 and cond2 and cond3:
            success_bp = bp
            break
            
    report += "\n## Verdict\n\n"
    if success_bp is not None:
        report += f"**SUCCESS**: Backward penalty **{success_bp}** satisfies all criteria.\n"
        report += f"- review_7 Caption DP reuse rate: {df[(df['video'] == 'review_7') & (df['backward_penalty'] == success_bp)]['reuse_rate'].values[0]:.3f} (<= 0.167)\n"
        report += f"- review_7 Caption DP TempAcc: {df[(df['video'] == 'review_7') & (df['backward_penalty'] == success_bp)]['temp_acc_15s'].values[0]:.3f} (>= 0.467)\n"
        report += f"- review_2 SigLIP DP reuse rate: {df[(df['video'] == 'review_2') & (df['backward_penalty'] == success_bp)]['reuse_rate'].values[0]:.3f} (<= 0.167)\n"
    else:
        report += "**FAILURE**: No backward penalty value satisfied all three criteria.\n"
        
    with open("notes/parameter_sweep_pilot.md", "w") as f:
        f.write(report)
    
    print(f"Report written to notes/parameter_sweep_pilot.md. Verdict: {'SUCCESS' if success_bp else 'FAILURE'}")

if __name__ == "__main__":
    run_sweep()

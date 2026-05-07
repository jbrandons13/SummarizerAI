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
        return loader()

def get_metrics(matches, summary, manifest, video_id, intermediate_dir):
    scene_ids = [m.matched_scene_id for m in matches]
    unique_scenes = len(set(scene_ids))
    num_sentences = len(matches)
    reuse_rate = 1 - (unique_scenes / num_sentences)
    
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
    
    return {
        "assignment": scene_ids,
        "unique": unique_scenes,
        "reuse_rate": reuse_rate,
        "temp_acc": temp_acc,
        "vis_coher": vis_coher,
        "clip_score": clip_score
    }

def run_diagnostic():
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
    video_id = "review_7"
    intermediate_dir = Path(f"data/intermediate/{video_id}")
    manifest = load_json_as_model(intermediate_dir / "keyframes_manifest.json", KeyframesManifest)
    summary = load_json_as_model(intermediate_dir / "summary_script.json", SummaryScript)
    
    retriever = CaptionCosineRetrieval(config, vram)
    
    diagnostic_results = {}
    
    # Test 1: bp = 0.0
    print("\n--- Running Test 1: backward_penalty = 0.0 ---")
    config["retrieval"]["dp_backward_penalty"] = 0.0
    output1 = retriever.retrieve(summary, manifest, method_name="test1")
    diagnostic_results["test1"] = get_metrics(output1.matches, summary, manifest, video_id, intermediate_dir)
    
    # Test 2: bp = -1.0
    print("\n--- Running Test 2: backward_penalty = -1.0 ---")
    config["retrieval"]["dp_backward_penalty"] = -1.0
    output2 = retriever.retrieve(summary, manifest, method_name="test2")
    diagnostic_results["test2"] = get_metrics(output2.matches, summary, manifest, video_id, intermediate_dir)
    
    # Test 3: Print check (already done by the print in the retrieve call above)
    # We'll just run one more with a distinct value to be sure
    print("\n--- Running Test 3: backward_penalty = 0.8 (Verification) ---")
    config["retrieval"]["dp_backward_penalty"] = 0.8
    retriever.retrieve(summary, manifest, method_name="test3")
    
    # Generate Report
    report = "# DP Sweep Diagnostic Report\n\n"
    
    report += "## Test 1: Extreme Value (backward_penalty = 0.0)\n\n"
    m1 = diagnostic_results["test1"]
    report += f"- **Assignment**: {m1['assignment']}\n"
    report += f"- **Reuse Rate**: {m1['reuse_rate']:.3f}\n"
    report += f"- **TempAcc (15s)**: {m1['temp_acc']:.3f}\n"
    report += f"- **VisCoher**: {m1['vis_coher']:.3f}\n\n"
    
    report += "## Test 2: Negative Value (backward_penalty = -1.0)\n\n"
    m2 = diagnostic_results["test2"]
    report += f"- **Assignment**: {m2['assignment']}\n"
    report += f"- **Reuse Rate**: {m2['reuse_rate']:.3f}\n"
    report += f"- **TempAcc (15s)**: {m2['temp_acc']:.3f}\n"
    report += f"- **VisCoher**: {m2['vis_coher']:.3f}\n\n"
    
    report += "## Test 3: Print Verification\n\n"
    report += "(Check terminal output for 'DIAGNOSTIC: DP called with backward_penalty=...') \n\n"
    
    report += "## Verdict\n\n"
    if m1['assignment'] == m2['assignment']:
        report += "**H2 CONFIRMED**: The assignment is identical even with a reward for backward jumps. The parameter is likely being ignored or overridden by a default value somewhere in the call chain.\n"
    else:
        report += "**H1 CONFIRMED**: The assignment changed when moving to a reward for backward jumps. This means the parameter is wired correctly, and the previous 'identical' results were due to the semantic signal being too strong for the 0.05-0.5 penalty range to overcome.\n"
        
    with open("notes/dp_sweep_diagnostic.md", "w") as f:
        f.write(report)
        
    print("\nDiagnostic complete. Report written to notes/dp_sweep_diagnostic.md")

if __name__ == "__main__":
    run_diagnostic()

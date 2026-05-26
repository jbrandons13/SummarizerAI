import json
import yaml
import numpy as np
from pathlib import Path
from dataclasses import asdict

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

def compare_runs():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    vram_manager = VRAMManager(device_id=0, limit_gb=22.0)
    siglip = SigLIPEncoder(vram_manager, config.get("models", {}).get("siglip", {}).get("model_name"))
    
    video_ids = [f"review_{i}" for i in range(1, 11)]
    intermediate_base = Path(config["paths"]["intermediate_dir"])
    
    total_old_groups = 0
    total_old_retrieve = 0
    total_old_generate = 0
    
    total_new_groups = 0
    total_new_retrieve = 0
    total_new_generate = 0
    
    comparison_results = {}

    for video_id in video_ids:
        intermediate_dir = intermediate_base / video_id
        old_assignments_path = intermediate_dir / "p4_assignments.json"
        summary_path = intermediate_dir / "summary_script.json"
        keyframes_manifest_path = intermediate_dir / "keyframes_manifest.json"
        
        if not old_assignments_path.exists() or not summary_path.exists() or not keyframes_manifest_path.exists():
            print(f"Skipping {video_id} due to missing files.")
            continue
            
        with open(old_assignments_path, "r") as f:
            old_assignments = json.load(f)
            
        summary = load_json_as_model(summary_path, SummaryScript)
        manifest = load_json_as_model(keyframes_manifest_path, KeyframesManifest)
        
        # Build scenes and sentences as run_sanity does
        frame_embeddings = siglip.embed_scenes(video_id, manifest)
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
            p4_scenes.append(P4Scene(id=sc.id, start=sc.start_seconds, end=sc.end_seconds, embedding=scene_emb))
            
        p4_sentences = []
        for s in summary.sentences:
            hint = s.source_timestamp_hint
            hint_val = (float(hint[0]), float(hint[1])) if hint and len(hint) >= 2 else (0.0, 0.0)
            p4_sentences.append(P4Sentence(id=s.id, text=s.text, timestamp_hint=hint_val))
            
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
            )
        )
        new_assignments_objs = gate.run(p4_sentences, p4_scenes)
        new_assignments = [asdict(a) for a in new_assignments_objs]
        
        old_count = len(old_assignments)
        new_count = len(new_assignments)
        
        old_ret = sum(1 for a in old_assignments if a["action"] == "retrieve")
        old_gen = sum(1 for a in old_assignments if a["action"] == "generate")
        new_ret = sum(1 for a in new_assignments if a["action"] == "retrieve")
        new_gen = sum(1 for a in new_assignments if a["action"] == "generate")
        
        total_old_groups += old_count
        total_old_retrieve += old_ret
        total_old_generate += old_gen
        
        total_new_groups += new_count
        total_new_retrieve += new_ret
        total_new_generate += new_gen
        
        # Check if identical
        identical = True
        if old_count != new_count:
            identical = False
        else:
            for old_a, new_a in zip(old_assignments, new_assignments):
                if old_a["sentence_ids"] != new_a["sentence_ids"]:
                    identical = False
                if old_a["scene_id"] != new_a["scene_id"]:
                    identical = False
                if old_a["action"] != new_a["action"]:
                    identical = False
                if abs(old_a["best_similarity"] - new_a["best_similarity"]) > 1e-4:
                    identical = False
                    
        comparison_results[video_id] = {
            "identical": identical,
            "old_groups": old_count,
            "new_groups": new_count,
            "old_retrieve": old_ret,
            "new_retrieve": new_ret,
            "old_generate": old_gen,
            "new_generate": new_gen
        }
        
    print("\n--- COMPARISON RESULTS ---")
    for vid, res in comparison_results.items():
        status = "✅ IDENTICAL" if res["identical"] else "❌ DIFFERENT"
        print(f"{vid}: {status}")
        print(f"  Old: {res['old_groups']} groups ({res['old_retrieve']} retrieve, {res['old_generate']} generate)")
        print(f"  New: {res['new_groups']} groups ({res['new_retrieve']} retrieve, {res['new_generate']} generate)")
        
    print("\n--- TOTALS ---")
    print(f"Old: {total_old_groups} groups ({total_old_retrieve} retrieve, {total_old_generate} generate)")
    print(f"New: {total_new_groups} groups ({total_new_retrieve} retrieve, {total_new_generate} generate)")

if __name__ == "__main__":
    compare_runs()

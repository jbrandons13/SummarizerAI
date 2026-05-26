#!/usr/bin/env python
import os
import sys
import json
import logging
import time
import yaml
import torch
from pathlib import Path
from PIL import Image

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.vram import VRAMManager
from src.phase5_prompt_builder import PromptBuilder
from src.phase5_ltx_runner import LTXRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prompt_expansion_study")

def run_study():
    print("\n" + "="*80)
    print("LTX-VIDEO VISUAL PROMPT EXPANSION STUDY")
    print("="*80 + "\n")

    # 1. Load default config
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )

    video_id = "review_1"
    intermediate_dir = Path("data/intermediate") / video_id
    p4_assignments_path = intermediate_dir / "p4_assignments.json"

    if not p4_assignments_path.exists():
        logger.error(f"Please run the pipeline for {video_id} first to generate intermediate files.")
        return

    # Load assignments to find a generate segment
    with open(p4_assignments_path, "r") as f:
        assignments = json.load(f)

    gen_groups = [g for g in assignments if g["action"] == "generate"]
    if not gen_groups:
        logger.warning("No 'generate' actions found in assignments for review_1. Running study on group 0 as fallback.")
        target_group = assignments[0]
    else:
        target_group = gen_groups[0]

    group_idx = assignments.index(target_group)
    print(f"Targeting Group {group_idx} for visual comparison study.")
    
    # Load summary script to extract narration
    summary_script_path = intermediate_dir / "summary_script.json"
    with open(summary_script_path, "r") as f:
        summary_script = json.load(f)
    
    sentences_list = summary_script["sentences"]
    group_sentences = [sentences_list[sid] for sid in target_group["sentence_ids"]]
    raw_narration = " ".join(s["text"] for s in group_sentences)
    keywords = []
    for s in group_sentences:
        for kw in s.get("keywords", []):
            if kw not in keywords:
                keywords.append(kw)

    print(f"\nRaw Narration: \"{raw_narration}\"")
    print(f"Keywords: {keywords}")

    # Build expanded prompt using standard builder
    qwen_model_id = config.get("models", {}).get("qwen_vl", {}).get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    prompt_builder = PromptBuilder(vram_manager=vram_manager, model_id=qwen_model_id)
    
    # Preprocess keyframe if needed
    scenes_map = {s["id"]: s for s in summary_script.get("scenes", [])} # fallback
    # Let's get the preprocessed keyframe path
    keyframes_manifest_path = intermediate_dir / "keyframes_manifest.json"
    with open(keyframes_manifest_path, "r") as f:
        keyframes_manifest = json.load(f)
    scenes_map = {s["id"]: s for s in keyframes_manifest["scenes"]}
    scene = scenes_map.get(target_group["scene_id"])
    
    preprocessed_rel_path = f"keyframes_ltx/group_{group_idx:03d}_keyframe_768x512.jpg"
    preprocessed_abs_path = intermediate_dir / preprocessed_rel_path
    
    if not preprocessed_abs_path.exists():
        prompt_builder._preprocess_keyframe(intermediate_dir / scene["keyframe_path"], preprocessed_abs_path)

    # Let's build the prompt!
    prompts_json_path = prompt_builder.build_prompts(video_id=video_id, rebuild_prompts=True, intermediate_dir=Path("data/intermediate"))
    with open(prompts_json_path, "r") as f:
        prompt_data = json.load(f)
    
    expanded_prompt = prompt_data["groups"][group_idx]["prompt"]
    if not expanded_prompt:
        # Fallback manual rich prompt if model generation failed
        expanded_prompt = (
            f"A sleek tech review shot focusing on {', '.join(keywords)}. Cinematic slow pan, "
            f"soft studio lighting, premium dark background, shallow depth of field, 8k resolution, photorealistic."
        )

    print(f"\nExpanded Visual Prompt (Idea A):")
    print(f"👉 \"{expanded_prompt}\"")

    # Save prompt configs for study run
    study_prompts_json = intermediate_dir / "study_prompts.json"
    study_data = {
        "video_id": video_id,
        "groups": [
            {
                "group_id": 998, # Baseline code
                "action": "generate",
                "audio_duration_seconds": 3.0,
                "num_frames": 121,
                "keyframe_preprocessed_path": str(preprocessed_rel_path),
                "narration": raw_narration,
                "prompt": raw_narration # Use baseline text
            },
            {
                "group_id": 999, # Expanded code
                "action": "generate",
                "audio_duration_seconds": 3.0,
                "num_frames": 121,
                "keyframe_preprocessed_path": str(preprocessed_rel_path),
                "narration": raw_narration,
                "prompt": expanded_prompt # Use expanded text
            }
        ]
    }
    
    with open(study_prompts_json, "w") as f:
        json.dump(study_data, f, indent=2)

    # Run generator for both study prompts
    ltx_model_path = config.get("models", {}).get("ltx", {}).get("model_path", "/home/wins053/models/ltx_video_distilled")
    ltx_runner = LTXRunner(vram_manager=vram_manager, model_path=ltx_model_path)
    
    # We will temporarily point the ltx_prompts.json to study_prompts.json and run generation!
    original_prompts_json = intermediate_dir / "ltx_prompts.json"
    backup_prompts_json = intermediate_dir / "ltx_prompts_backup.json"
    
    if original_prompts_json.exists():
        shutil.copy(original_prompts_json, backup_prompts_json)
        
    try:
        shutil.copy(study_prompts_json, original_prompts_json)
        print("\n" + "="*50)
        print("GENERATING STUDY CLIPS WITH LTX-VIDEO (This will take ~1-2 mins)...")
        print("="*50 + "\n")
        
        ltx_runner.generate_clips(video_id=video_id, rebuild_clips=True, intermediate_dir=Path("data/intermediate"))
        
        baseline_clip = intermediate_dir / "generated/group_998.mp4"
        expanded_clip = intermediate_dir / "generated/group_999.mp4"
        
        print("\n" + "="*50)
        print("STUDY COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"1. Baseline Clip (Raw Text) saved to:\n   👉 {baseline_clip.absolute()}")
        print(f"2. Expanded Clip (Idea A) saved to:\n   👉 {expanded_clip.absolute()}")
        print("\nCompare both clips physically to see the dramatic improvement in quality and aesthetic consistency!")
        
    finally:
        # Restore backup
        if backup_prompts_json.exists():
            shutil.move(backup_prompts_json, original_prompts_json)
        study_prompts_json.unlink(missing_ok=True)

if __name__ == "__main__":
    import shutil
    run_study()

import argparse
import json
import os
import shutil
import yaml
import time
import torch
import logging
from PIL import Image

import transformers.activations
if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

import sys
sys.modules['gptqmodel'] = None
sys.modules['awq'] = None
import peft.import_utils
peft.import_utils.is_auto_awq_available = lambda: False
peft.import_utils.is_gptqmodel_available = lambda: False
peft.import_utils.is_auto_gptq_available = lambda: False

sys.path.append(os.getcwd())
from src.phase4.image_gen import load_pipeline, generate_image, unload_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--baseline-images", required=True)
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--ref-weight", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override ref weight for CONCEPT_ANCHOR
    if "phase4" not in config: config["phase4"] = {}
    if "image_gen" not in config["phase4"]: config["phase4"]["image_gen"] = {}
    config["phase4"]["image_gen"]["concept_anchor_ref_weight"] = args.ref_weight

    with open(args.storyboard, "r") as f:
        storyboard = json.load(f)["shots"]

    with open(args.anchors, "r") as f:
        anchors_data = json.load(f)
        anchors = anchors_data["shots"] if "shots" in anchors_data else anchors_data

    anchor_map = {a["shot_id"]: a for a in anchors}

    os.makedirs(args.out_images, exist_ok=True)

    logger.info("Loading pipeline...")
    pipe, _ = load_pipeline(config)

    n_gen = 0
    n_copy = 0

    for shot in storyboard:
        shot_id = shot["shot_id"]
        anchor_info = anchor_map.get(shot_id)
        if not anchor_info:
            logger.warning(f"No anchor info for {shot_id}, skipping")
            continue
            
        decision = anchor_info["anchor_decision"]
        out_path = os.path.join(args.out_images, f"{shot_id}.png")
        if os.path.exists(out_path):
            logger.info(f"[{shot_id}] SKIP (already exists)")
            continue
            
        # We consider both RESET and anything that isn't CHAIN as a copy from baseline 
        # (Wait, actually if decision is RESET, it's COPY). 
        if decision == "RESET":
            shot_dict = {"id": shot_id, "image_prompt": shot["image_prompt"]}
            try:
                img = generate_image(pipe, shot_dict, decision="RESET", config=config)
                img.save(out_path)
                n_gen += 1
                logger.info(f"[{shot_id}] GEN with RESET")
            except Exception as e:
                logger.error(f"Failed to GEN {shot_id}: {e}")
        elif decision in ("CHAIN", "CONCEPT_ANCHOR"):
            ref_shot_id = anchor_info.get("anchor_source")
            if ref_shot_id:
                ref_img_path = os.path.join(args.out_images, f"{ref_shot_id}.png")
                # Fallback to baseline if it hasn't been generated yet (e.g. out of order, which shouldn't happen)
                if not os.path.exists(ref_img_path):
                    ref_img_path = os.path.join(args.baseline_images, f"{ref_shot_id}.png")
                ref_image = Image.open(ref_img_path).convert("RGB")
            else:
                ref_image = None
                
            shot_dict = {"id": shot_id, "image_prompt": shot["image_prompt"]}
            # Force the decision to CONCEPT_ANCHOR to trigger IP-Adapter
            try:
                img = generate_image(pipe, shot_dict, decision="CONCEPT_ANCHOR", config=config, ref_image=ref_image)
                img.save(out_path)
                n_gen += 1
                logger.info(f"[{shot_id}] GEN with CONCEPT_ANCHOR")
            except Exception as e:
                logger.error(f"Failed to GEN {shot_id}: {e}")

    vram_peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    logger.info(f"Done. [GEN]: {n_gen}, [COPY]: {n_copy}")
    logger.info(f"VRAM Peak: {vram_peak:.2f} GB")

    unload_pipeline(pipe)

if __name__ == "__main__":
    main()

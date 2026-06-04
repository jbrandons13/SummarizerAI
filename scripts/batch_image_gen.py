import json
import os
import glob
import yaml
import time
import shutil
import logging
import traceback
import torch
from PIL import Image

from src.phase4.image_gen import load_pipeline, generate_image, unload_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    
    # Configuration
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_dir_glob = glob.glob("data/intermediate/lT_QAkL6lj0_*/phase4")
    if not base_dir_glob:
        logger.error("Could not find phase4 directory.")
        return
        
    phase4_dir = base_dir_glob[0]
    
    storyboard_path = os.path.join(phase4_dir, "storyboard.json")
    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)["shots"]
        
    policies = ["always_chain", "never_chain", "fixed_interval", "semantic_triggered"]
    
    base_images_dir = os.path.join(phase4_dir, "_base")
    os.makedirs(base_images_dir, exist_ok=True)
    
    # Dictionary to keep track of failed shots
    failed_shots = []

    skip_existing = False

    logger.info("Loading pipeline...")
    pipe, ip_loaded = load_pipeline(config)

    # Phase B.1: Generate Base Images (Text-to-Image for all 44 shots)
    logger.info("Phase B.1: Generating base images (text-to-image)...")
    base_generated = {}
    for i, shot in enumerate(storyboard):
        shot_id = shot["shot_id"]
        prompt = shot["image_prompt"]
        out_path = os.path.join(base_images_dir, f"{shot_id}.png")
        
        if skip_existing and os.path.exists(out_path):
            logger.info(f"[{i+1}/{len(storyboard)}] Skipping {shot_id}, already exists.")
            base_generated[shot_id] = out_path
            continue
            
        logger.info(f"[{i+1}/{len(storyboard)}] Generating base image for {shot_id}...")
        
        try:
            # For base image, decision is RESET (no IP-Adapter)
            shot_dict = {"id": shot_id, "image_prompt": prompt}
            img = generate_image(pipe, shot_dict, decision="RESET", config=config, ref_image=None)
            img.save(out_path)
            base_generated[shot_id] = out_path
            
            vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
            if vram_peak > 18.0:
                logger.warning(f"VRAM peak exceeded 18GB: {vram_peak:.2f} GB")
                
        except Exception as e:
            logger.error(f"Failed to generate base image for {shot_id}: {str(e)}")
            traceback.print_exc()
            failed_shots.append({"shot_id": shot_id, "phase": "base", "error": str(e)})

    # Phase B.2 & B.3: Populate Policy Folders
    logger.info("Phase B.2 & B.3: Populating policy folders and generating SOFT_CHAINs...")
    
    for policy in policies:
        policy_dir = os.path.join(phase4_dir, policy)
        images_dir = os.path.join(policy_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        anchors_path = os.path.join(policy_dir, "storyboard_with_anchors.json")
        if not os.path.exists(anchors_path):
            logger.warning(f"No anchors found for policy {policy}, skipping.")
            continue
            
        with open(anchors_path, "r") as f:
            anchors_data = json.load(f)
            # handle both list and dict formats just in case
            anchors = anchors_data["shots"] if "shots" in anchors_data else anchors_data
            
        # Map anchors by shot_id for easy lookup
        anchor_map = {a["shot_id"]: a for a in anchors}
        
        for shot in storyboard:
            shot_id = shot["shot_id"]
            anchor_info = anchor_map.get(shot_id)
            
            if not anchor_info:
                logger.warning(f"No anchor info for {shot_id} in {policy}")
                continue
                
            decision = anchor_info["anchor_decision"]
            out_path = os.path.join(images_dir, f"{shot_id}.png")
            
            if skip_existing and os.path.exists(out_path):
                continue
                
            if decision in ["RESET", "CHAIN"]:
                # Phase B.2: Copy from base
                base_path = base_generated.get(shot_id)
                if base_path and os.path.exists(base_path):
                    shutil.copy2(base_path, out_path)
                else:
                    logger.error(f"Cannot copy base image for {shot_id} (missing).")
                    failed_shots.append({"shot_id": shot_id, "policy": policy, "phase": "copy", "error": "Missing base image"})
            
            elif decision == "SOFT_CHAIN":
                # Phase B.3: Generate SOFT_CHAIN
                ref_shot_id = anchor_info["anchor_source"]
                ref_path = base_generated.get(ref_shot_id)
                
                if not ref_path or not os.path.exists(ref_path):
                    logger.error(f"Cannot generate SOFT_CHAIN for {shot_id} in {policy}: Missing reference {ref_shot_id}")
                    failed_shots.append({"shot_id": shot_id, "policy": policy, "phase": "soft_chain", "error": f"Missing reference {ref_shot_id}"})
                    continue
                    
                logger.info(f"Generating SOFT_CHAIN for {shot_id} in {policy} (ref: {ref_shot_id})...")
                try:
                    ref_image = Image.open(ref_path).convert("RGB")
                    shot_dict = {"id": shot_id, "image_prompt": shot["image_prompt"]}
                    
                    img = generate_image(pipe, shot_dict, decision="SOFT_CHAIN", config=config, ref_image=ref_image)
                    img.save(out_path)
                    
                    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
                    if vram_peak > 18.0:
                        logger.warning(f"VRAM peak exceeded 18GB: {vram_peak:.2f} GB")
                        
                except Exception as e:
                    logger.error(f"Failed to generate SOFT_CHAIN for {shot_id} in {policy}: {str(e)}")
                    traceback.print_exc()
                    failed_shots.append({"shot_id": shot_id, "policy": policy, "phase": "soft_chain", "error": str(e)})

    # Unload and clean up
    unload_pipeline(pipe)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Batch generation completed in {total_time:.2f} seconds.")
    if failed_shots:
        logger.error(f"There were {len(failed_shots)} failed shots:")
        for f in failed_shots:
            logger.error(f)
    else:
        logger.info("All shots generated successfully!")
        
if __name__ == "__main__":
    main()

import json
import os
import glob
import yaml
from PIL import Image
from src.phase4.image_gen import load_pipeline, generate_image, unload_pipeline

def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Find the target video directory
    base_dir = glob.glob("data/intermediate/lT_QAkL6lj0_*/phase4/semantic_triggered")[0]
    vid_dir = os.path.dirname(os.path.dirname(base_dir))
    
    # Load storyboard and anchors
    with open(os.path.join(base_dir, "storyboard_with_anchors.json"), "r") as f:
        anchors = json.load(f)["shots"]
        
    with open(os.path.join(vid_dir, "phase4", "storyboard.json"), "r") as f:
        storyboard = json.load(f)["shots"]

    # Map shot id to prompt
    shot_map = {s["shot_id"]: s for s in storyboard}
    
    # Find 1 RESET, 1 CHAIN, 1 SOFT_CHAIN
    reset_shot = next(a for a in anchors if a["anchor_decision"] == "RESET")
    chain_shot = next(a for a in anchors if a["anchor_decision"] == "CHAIN")
    soft_chain_shot = next((a for a in anchors if a["anchor_decision"] == "SOFT_CHAIN"), None)
    
    if not soft_chain_shot:
        print("Warning: No SOFT_CHAIN found. Falling back to second CHAIN.")
        soft_chain_shot = next(a for a in anchors if a["anchor_decision"] == "CHAIN" and a["shot_id"] != chain_shot["shot_id"])
        
    targets = [reset_shot, chain_shot, soft_chain_shot]
    
    # Needs anchor source for SOFT_CHAIN
    anchor_source_shot = None
    if soft_chain_shot["anchor_decision"] == "SOFT_CHAIN":
        anchor_src_id = soft_chain_shot["anchor_source"]
        anchor_source_shot = next(a for a in anchors if a["shot_id"] == anchor_src_id)
        if anchor_source_shot not in targets:
            targets.insert(0, anchor_source_shot)

    print(f"Target shots to generate: {[t['shot_id'] for t in targets]}")
    
    # Output dir
    out_dir = os.path.join(vid_dir, "phase4", "_sample")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load pipeline
    pipe, ip_loaded = load_pipeline(config)
    
    generated_images = {}
    
    for t in targets:
        shot_id = t["shot_id"]
        decision = t["anchor_decision"]
        prompt = shot_map[shot_id]["image_prompt"]
        
        # Prepare shot dict for generate_image
        shot_dict = {"id": shot_id, "image_prompt": prompt}
        
        ref_image = None
        if decision == "SOFT_CHAIN":
            ref_id = t["anchor_source"]
            ref_image = generated_images.get(ref_id)
            if not ref_image:
                print(f"WARNING: Reference image for {ref_id} not found!")
                
        img = generate_image(pipe, shot_dict, decision, config, ref_image=ref_image)
        generated_images[shot_id] = img
        
        out_path = os.path.join(out_dir, f"{shot_id}_{decision}.png")
        img.save(out_path)
        print(f"Saved {out_path}")
        
    unload_pipeline(pipe)
    print("Done")

if __name__ == "__main__":
    main()

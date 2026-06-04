import os
import sys
import yaml
import logging
from PIL import Image

sys.path.append(os.getcwd())
from src.phase4.image_gen import load_pipeline, generate_image, unload_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Update negative prompt
    config["phase4"]["image_gen"]["negative_prompt"] = "text, letters, words, captions, labels, numbers, infographic, panels, watermark, signature, gibberish"
    
    # Define test shots
    test_shots = {
        "shot_002": "A circular landscape scene of the rock cycle: a small erupting volcano at center, rivers and eroding cliffs, layered sedimentary banks and rocky mountains arranged in a ring, curved connecting paths, flat 2D educational vector, vivid colors",
        "shot_004": "A rock split open revealing a colorful crystal geode interior, distinct mineral crystals, detailed, flat 2D educational vector, vivid colors",
        "shot_016": "Cross-section view of the earth: glowing magma chamber underground, lava erupting above cooling into layered igneous rock, cutaway, flat 2D educational vector, vivid colors",
        "shot_019": "A rock deep underground being squeezed and heated, glowing cracks, bending and folding rock layers, converging rocky walls showing pressure, dramatic flat 2D educational vector scene, vivid colors"
    }

    pipe, _ = load_pipeline(config)
    
    out_dir = "/tmp/diagram_refined"
    os.makedirs(out_dir, exist_ok=True)
    
    # D1-richer images are from the previous run
    old_images_dir = "/tmp/diagram_smoke"
    
    for shot_id, prompt in test_shots.items():
        shot_dict = {"id": shot_id, "image_prompt": prompt}
        # decision="RESET" will ensure IP-Adapter scale is set to 0.0
        img = generate_image(pipe, shot_dict, decision="RESET", config=config)
        new_path = os.path.join(out_dir, f"{shot_id}_refined.png")
        img.save(new_path)
        
        # Combine side by side with D1-richer image
        old_path = os.path.join(old_images_dir, f"{shot_id}_new.png")
        if os.path.exists(old_path):
            old_img = Image.open(old_path)
            # Create a side-by-side image
            w, h = img.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(old_img, (0, 0))
            combined.paste(img, (w, 0))
            
            combined_path = os.path.join(out_dir, f"{shot_id}_refined_comparison.png")
            combined.save(combined_path)
            logging.info(f"Saved comparison to {combined_path}")
        else:
            logging.warning(f"Old image not found for {shot_id} at {old_path}")
            
    unload_pipeline(pipe)
    
if __name__ == '__main__':
    main()

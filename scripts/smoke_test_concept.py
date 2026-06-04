import os
import sys
import yaml
import json
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

def main():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Force IP-Adapter reference weight to 0.2
    if "phase4" not in config: config["phase4"] = {}
    if "image_gen" not in config["phase4"]: config["phase4"]["image_gen"] = {}
    config["phase4"]["image_gen"]["concept_anchor_ref_weight"] = 0.2
    
    video_id = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    storyboard_path = f"data/intermediate/{video_id}/phase4/storyboard.json"
    with open(storyboard_path, "r") as f:
        storyboard = json.load(f)["shots"]
        
    # The baseline images are from concept_anchor_canonical_w02/images/ or just the concept anchor base.
    # Wait, the pipeline uses concept-canonical image as ref! Where is it?
    # Usually it's in data/intermediate/.../phase4/_eval/ or phase4/concept_anchor_canonical_w02/images/
    baseline_dir = f"data/intermediate/{video_id}/phase4/concept_anchor_canonical_w02/images"
    
    test_shots = ["shot_002", "shot_004", "shot_016", "shot_019"]
    
    pipe, _ = load_pipeline(config)
    
    out_dir = "/tmp/concept_smoke"
    os.makedirs(out_dir, exist_ok=True)
    
    for shot in storyboard:
        shot_id = shot["shot_id"]
        if shot_id not in test_shots:
            continue
            
        ref_path = os.path.join(baseline_dir, f"{shot_id}.png")
        if not os.path.exists(ref_path):
            logging.warning(f"Ref image not found: {ref_path}")
            ref_img = Image.new("RGB", (832, 480), "white")
        else:
            ref_img = Image.open(ref_path).convert("RGB")
            
        logging.info(f"Generating {shot_id} with prompt: {shot['image_prompt']}")
        shot_dict = {"id": shot_id, "image_prompt": shot["image_prompt"]}
        # Render with CONCEPT_ANCHOR
        img = generate_image(pipe, shot_dict, decision="CONCEPT_ANCHOR", config=config, ref_image=ref_img)
        new_path = os.path.join(out_dir, f"{shot_id}_concept.png")
        img.save(new_path)
        
    unload_pipeline(pipe)
    
if __name__ == '__main__':
    main()

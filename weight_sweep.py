import argparse
import json
import os
import yaml
import logging
from PIL import Image

import transformers.activations
if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

import sys
sys.modules['gptqmodel'] = None
sys.modules['awq'] = None
sys.modules['bitsandbytes'] = None
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
    parser.add_argument("--reference", required=True)
    parser.add_argument("--shots", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.storyboard, "r") as f:
        storyboard = json.load(f)["shots"]

    os.makedirs(args.out, exist_ok=True)
    
    pipe, _ = load_pipeline(config)
    
    shots_to_gen = args.shots.split(",")
    weights = [float(w) for w in args.weights.split(",")]
    
    ref_image = Image.open(args.reference).convert("RGB")
    
    if ref_image.size == (1, 1) and storyboard:
        logger.info("Dummy reference image detected. Generating base reference from first shot (w=0.0).")
        first_shot = storyboard[0].copy()
        first_shot["id"] = first_shot["shot_id"]
        if "phase4" not in config: config["phase4"] = {}
        if "image_gen" not in config["phase4"]: config["phase4"]["image_gen"] = {}
        config["phase4"]["image_gen"]["concept_anchor_ref_weight"] = 0.0
        ref_image = generate_image(pipe, first_shot, "CONCEPT_ANCHOR", config, ref_image=None)
        ref_image.save(args.reference)
        logger.info(f"Saved real reference image to {args.reference}")
    
    manifest = []
    
    for shot in storyboard:
        shot_id = shot["shot_id"]
        shot["id"] = shot_id
        if shot_id not in shots_to_gen:
            continue
            
        for w in weights:
            logger.info(f"Generating {shot_id} with weight {w}")
            
            # Override config weight
            if "phase4" not in config: config["phase4"] = {}
            if "image_gen" not in config["phase4"]: config["phase4"]["image_gen"] = {}
            config["phase4"]["image_gen"]["concept_anchor_ref_weight"] = w
            
            out_name = f"{shot_id}_w{w:.1f}.png"
            out_path = os.path.join(args.out, out_name)
            
            if not os.path.exists(out_path):
                img = generate_image(pipe, shot, "CONCEPT_ANCHOR", config, ref_image=ref_image)
                img.save(out_path)
                
            manifest.append({
                "shot_id": shot_id,
                "weight": w,
                "file": out_name,
                "path": out_path
            })
            
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()

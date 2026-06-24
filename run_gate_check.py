import os
import json
import yaml
from PIL import Image

videos = [
    {"id": "V12", "name": "Eye", "concept": "a colorful cartoon illustration of a human eye, the organ of sight"},
    {"id": "V13", "name": "Hurricane", "concept": "a colorful cartoon illustration of a hurricane, a large swirling storm system"},
    {"id": "V14", "name": "Reef", "concept": "a colorful cartoon illustration of a coral reef, an underwater ecosystem of corals and fish"}
]

os.makedirs("gate_check", exist_ok=True)

import sys
sys.modules['gptqmodel'] = None
sys.modules['awq'] = None
sys.modules['bitsandbytes'] = None
import peft.import_utils
peft.import_utils.is_auto_awq_available = lambda: False
peft.import_utils.is_gptqmodel_available = lambda: False
peft.import_utils.is_auto_gptq_available = lambda: False
sys.path.append(os.getcwd())
from src.phase4.image_gen import load_pipeline, generate_image

print("Loading SDXL Pipeline...")
with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)
pipe, _ = load_pipeline(config)

for v in videos:
    vid = v["id"]
    vname = v["name"]
    concept = v["concept"]
    
    print(f"\n=== GATE CHECK FOR {vid} {vname} ===")
    run_dir = f"data/intermediate/{vid}/phase4"
            
    with open(f"{run_dir}/storyboard.json") as f:
        storyboard = json.load(f)["shots"]
        
    print(f"Generating Reference Image for {vid}...")
    ref_shot = {"id": "reference", "image_prompt": concept}
    config["phase4"] = config.get("phase4", {})
    config["phase4"]["image_gen"] = config["phase4"].get("image_gen", {})
    config["phase4"]["image_gen"]["concept_anchor_ref_weight"] = 0.0
    
    ref_img = generate_image(pipe, ref_shot, "CONCEPT_ANCHOR", config, ref_image=None)
    ref_img.save(f"gate_check/{vid}_{vname}_reference.png")
    
    print(f"Generating 3 w=0 shots for {vid}...")
    shots_to_check = storyboard[:3]
    for i, shot in enumerate(shots_to_check):
        shot["id"] = shot["shot_id"]
        config["phase4"]["image_gen"]["concept_anchor_ref_weight"] = 0.0
        shot_img = generate_image(pipe, shot, "CONCEPT_ANCHOR", config, ref_image=None)
        shot_img.save(f"gate_check/{vid}_{vname}_shot_{i+1}_w0.png")
        
    print(f"Gate check images saved for {vid}.")

print("\nGate check generation complete!")

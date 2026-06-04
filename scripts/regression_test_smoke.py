import json
import time
import hashlib
from pathlib import Path
from src.phase4.comfyui_client import ComfyUIClient
from src.phase4.video_gen import generate_clip

def log_image_load(label, path):
    if not Path(path).exists():
        print(f"❌ {label}: path NOT EXIST: {path}")
        return
    with open(path, 'rb') as f:
        data = f.read()
    md5 = hashlib.md5(data).hexdigest()[:8]
    size = len(data)
    print(f"✅ {label}: {path} (size={size}B, md5={md5})")

def main():
    video_id = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    base_path = Path(f"data/intermediate/{video_id}/phase4")
    
    out_dir = base_path / "_regression_test"
    out_dir.mkdir(exist_ok=True)
    
    # Load workflow template
    with open("scripts/wan_flf2v_workflow.json") as f:
        workflow_template = json.load(f)
        
    client = ComfyUIClient()
    if not client.is_running():
        client.start_server()
        
    try:
        # Task 1: Regression Test (shot_009 -> shot_010 FLF2V)
        shot_010 = {
            "shot_id": "shot_010",
            "visual_description": "The clastic rocks are shown forming from bits of other rocks compacted underground."
        }
        
        first_frame = base_path / "semantic_triggered/images/shot_009.png"
        last_frame = base_path / "semantic_triggered/images/shot_010.png"
        
        # Task 3: Image Load Verification
        print("\n--- Task 3: Image Load Verification ---")
        log_image_load("First Frame (shot_009)", first_frame)
        log_image_load("Last Frame (shot_010)", last_frame)
        
        # Task 2: Dump the JSON
        workflow = json.loads(json.dumps(workflow_template))
        ff_name = f"ff_{first_frame.name}"
        lf_name = f"lf_{last_frame.name}"
        workflow["5"]["inputs"]["image"] = ff_name
        workflow["6"]["inputs"]["image"] = lf_name
        workflow["9"]["inputs"]["text"] = shot_010["visual_description"]
        workflow["12"]["inputs"]["seed"] = 42
        
        with open("/tmp/regression_workflow.json", "w") as f:
            json.dump(workflow, f, indent=2)
            
        print("\n--- Task 1: Regression Test ---")
        out_path = out_dir / "shot_010_FLF2V_regression.webp"
        print(f"Generating {out_path.name}...")
        
        # We use the raw generate_clip to match production exactly
        res = generate_clip(client, shot_010, first_frame, last_frame, out_path, workflow_template, seed=42)
        
        print(f"Status: {res['status']}")
        print(f"Gen Time: {res.get('gen_time_sec', 0):.1f}s")
        print(f"VRAM Peak: {res.get('vram_peak_gb', 0):.1f}GB")
        
        # Dump the actual workflow sent via a small override
        # We will compare the /tmp/regression_workflow.json to what smoke_test sent
    finally:
        # Don't kill server so user can inspect or next script can use it
        client.free_vram()

if __name__ == "__main__":
    main()

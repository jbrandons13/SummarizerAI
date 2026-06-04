import json
import time
from pathlib import Path
from src.phase4.comfyui_client import ComfyUIClient
from src.phase4.video_gen import generate_clip

def extract_frames(webp_path, out_prefix):
    from PIL import Image, ImageSequence
    img = Image.open(webp_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
    for i in [0, 40, 80]:
        if i < len(frames):
            out_file = f"{out_prefix}_frame_{i:03d}.png"
            frames[i].save(out_file)
            print(f"Extracted {out_file}")

def test_1():
    video_id = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    base_path = Path(f"data/intermediate/{video_id}/phase4")
    out_dir = base_path / "_diagnostic"
    out_dir.mkdir(exist_ok=True)
    
    with open("scripts/wan_flf2v_workflow.json") as f:
        workflow_template = json.load(f)
        
    client = ComfyUIClient()
    client.start_server()
    
    try:
        print("\n=== Running Test 1: FLF2V(self, self) ===")
        shot_mock = {
            "visual_description": "static composition, slow ambient camera push, subtle parallax movement"
        }
        first_frame = base_path / "semantic_triggered/images/shot_001.png"
        last_frame = base_path / "semantic_triggered/images/shot_001.png"
        out_path = out_dir / "test_01_flf_self_self.webp"
        
        res = generate_clip(client, shot_mock, first_frame, last_frame, out_path, workflow_template, seed=42)
        
        print(f"Status: {res['status']}")
        print(f"Gen Time: {res.get('gen_time_sec', 0):.1f}s")
        print(f"VRAM Peak: {res.get('vram_peak_gb', 0):.1f}GB")
        
        if res["status"] == "success":
            extract_frames(out_path, str(out_dir / "test_01_flf_self_self"))
            
    finally:
        client.free_vram()
        client.kill_server()

if __name__ == "__main__":
    test_1()

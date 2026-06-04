import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image
from src.phase4.comfyui_client import ComfyUIClient

def extract_last_frame(clip_path: Path) -> Path:
    """
    Extract last frame from .webp clip. Save as PNG in temp dir, return path.
    """
    temp_dir = Path("/tmp/video_summarizer_frames")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    h = hashlib.md5(str(clip_path).encode()).hexdigest()
    frame_path = temp_dir / f"{clip_path.stem}_{h}_last_frame.png"
    
    if frame_path.exists():
        return frame_path
        
    img = Image.open(clip_path)
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(img.n_frames - 1)
    img.convert("RGB").save(frame_path, "PNG")
    return frame_path

def determine_input_paths(
    shot_id: str,
    anchor_decision: dict,
    policy_dir: Path,
    storyboard_dir: Path,
    rendered_clips: Dict[str, Path]
) -> Tuple[Path, Optional[Path], str]:
    decision_type = anchor_decision["anchor_decision"]
    anchor_source = anchor_decision.get("anchor_source")
    
    current_image = policy_dir / "images" / f"{shot_id}.png"
    
    if decision_type in ("RESET", "SOFT_CHAIN"):
        return current_image, None, "I2V"
    elif decision_type == "CHAIN":
        if anchor_source and anchor_source in rendered_clips:
            first_frame = extract_last_frame(rendered_clips[anchor_source])
            return first_frame, current_image, "FLF2V"
        else:
            return current_image, None, "I2V"
    return current_image, None, "I2V"

def generate_clip(
    client: ComfyUIClient,
    shot: dict,
    first_frame_path: Path,
    last_frame_path: Optional[Path],
    output_path: Path,
    workflow_template: dict,
    seed: Optional[int] = None
) -> dict:
    
    workflow = json.loads(json.dumps(workflow_template))
    
    prompt_text = shot["visual_description"]
    
    input_dir = client.comfyui_path / "input"
    input_dir.mkdir(exist_ok=True)
    
    ff_name = f"ff_{first_frame_path.name}"
    shutil.copy(first_frame_path, input_dir / ff_name)
    workflow["5"]["inputs"]["image"] = ff_name
    
    if last_frame_path:
        lf_name = f"lf_{last_frame_path.name}"
        shutil.copy(last_frame_path, input_dir / lf_name)
        workflow["6"]["inputs"]["image"] = lf_name
    else:
        workflow["6"]["inputs"]["image"] = ff_name
        if "end_image" in workflow["11"]["inputs"]:
            del workflow["11"]["inputs"]["end_image"]
        if "clip_vision_end_image" in workflow["11"]["inputs"]:
            del workflow["11"]["inputs"]["clip_vision_end_image"]
            
    workflow["9"]["inputs"]["text"] = prompt_text
    if seed is not None:
        workflow["12"]["inputs"]["seed"] = seed
        
    start_time = time.time()
    
    prompt_id = client.queue_workflow(workflow)
    res = client.wait_for_completion(prompt_id)
    
    gen_time_sec = time.time() - start_time
    
    stats = client.system_stats()
    vram_peak_gb = 0.0
    if "devices" in stats and len(stats["devices"]) > 0:
        vram_peak_gb = stats["devices"][0].get("vram_used", 0) / (1024**3)
    
    if res["status"] != "success":
        return {
            "status": res["status"],
            "output_path": None,
            "gen_time_sec": gen_time_sec,
            "vram_peak_gb": vram_peak_gb,
            "error": res["details"]
        }
    
    output_info = client.get_output(prompt_id, "14")
    if not output_info:
        return {
            "status": "error",
            "output_path": None,
            "gen_time_sec": gen_time_sec,
            "vram_peak_gb": vram_peak_gb,
            "error": "Output info not found in history"
        }
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_output(output_info["filename"], output_info.get("subfolder", ""), output_path)
    
    return {
        "status": "success",
        "output_path": output_path,
        "gen_time_sec": gen_time_sec,
        "vram_peak_gb": vram_peak_gb
    }

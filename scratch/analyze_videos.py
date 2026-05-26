import os
import json
import numpy as np
import imageio
from PIL import Image

results_file = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_outputs/ltx_prompt_refine/results.json"
analysis_file = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_outputs/ltx_prompt_refine/analysis.json"
keyframes_dir = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/phase5_smoke_outputs/ltx_prompt_refine/keyframes"

os.makedirs(keyframes_dir, exist_ok=True)

if not os.path.exists(results_file):
    print(f"Results file not found: {results_file}")
    exit(1)

with open(results_file, "r") as f:
    results = json.load(f)

analysis = {}

for run_key, run in results.items():
    if run.get("status") != "success":
        print(f"Skipping failed run: {run_key}")
        continue
        
    video_path = run["output_path_sub"]
    if not os.path.exists(video_path):
        # Fallback to root path if subpath doesn't exist
        video_path = run["output_path_root"]
        
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        continue
        
    print(f"Analyzing {run_key} from {video_path}...")
    
    try:
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            frames.append(frame)
            
        num_frames = len(frames)
        height, width, channels = frames[0].shape
        
        # Save keyframes
        first_frame = frames[0]
        mid_frame = frames[num_frames // 2]
        last_frame = frames[-1]
        
        Image.fromarray(first_frame).save(os.path.join(keyframes_dir, f"{run_key}_first.png"))
        Image.fromarray(mid_frame).save(os.path.join(keyframes_dir, f"{run_key}_mid.png"))
        Image.fromarray(last_frame).save(os.path.join(keyframes_dir, f"{run_key}_last.png"))
        
        # Compute drift metrics for Input B (static scene)
        drift_mae_max = 0.0
        drift_mae_avg = 0.0
        
        if run["input_id"] == "input_b":
            first_frame_gray = np.dot(first_frame[...,:3], [0.2989, 0.5870, 0.1140])
            maes = []
            for f in frames[1:]:
                f_gray = np.dot(f[...,:3], [0.2989, 0.5870, 0.1140])
                mae = np.mean(np.abs(f_gray - first_frame_gray))
                maes.append(mae)
            drift_mae_max = float(np.max(maes))
            drift_mae_avg = float(np.mean(maes))
            print(f"  Drift MAE (Max): {drift_mae_max:.2f} | (Avg): {drift_mae_avg:.2f}")
            
        analysis[run_key] = {
            "input_id": run["input_id"],
            "variant": run["variant"],
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "drift_mae_max": drift_mae_max,
            "drift_mae_avg": drift_mae_avg,
            "first_frame_path": os.path.join(keyframes_dir, f"{run_key}_first.png"),
            "mid_frame_path": os.path.join(keyframes_dir, f"{run_key}_mid.png"),
            "last_frame_path": os.path.join(keyframes_dir, f"{run_key}_last.png")
        }
    except Exception as e:
        print(f"Error analyzing {run_key}: {e}")

with open(analysis_file, "w") as f:
    json.dump(analysis, f, indent=2)
    
print(f"Analysis completed and saved to {analysis_file}")

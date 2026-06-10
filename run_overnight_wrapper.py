#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import shutil
import csv
from glob import glob

VIDEOS = {
    "geology": {
        "dir": "data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge",
        "url": None,
        "concept": "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones"
    },
    "ecology": {
        "dir": "data/intermediate/2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology",
        "url": None,
        "concept": "a colorful cartoon illustration of a dripping water cave, the water cycle"
    },
    "photosynthesis": {
        "dir": None,
        "url": "https://www.youtube.com/watch?v=sQK3Yr4Sc_k",
        "concept": "a colorful cartoon illustration of a green plant leaf, sunlight and chloroplasts"
    },
    "iphone": {
        "dir": None,
        "url": "https://www.youtube.com/watch?v=MRtg6A1f2Ko",
        "concept": "a cartoon illustration of a smartphone"
    }
}

def log(msg):
    print(f"[run_overnight] {msg}", flush=True)

def run(cmd, env=None):
    log(f"RUN: {cmd}")
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        log(f"ERROR: command failed with code {res.returncode}")

def phase_a():
    log("=== PHASE A ===")
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = "."
    os.makedirs("data/raw_videos", exist_ok=True)
    
    for vid, info in VIDEOS.items():
        run_dir = f"runs/{vid}"
        os.makedirs(f"{run_dir}/audio", exist_ok=True)
        os.makedirs(f"{run_dir}/sweep", exist_ok=True)
        os.makedirs(f"{run_dir}/daca", exist_ok=True)
        
        if info["url"] and not info["dir"]:
            log(f"[{vid}] Running upstream pipeline for URL: {info['url']}")
            # 1. Download
            cmd_dl = f"yt-dlp -o 'data/raw_videos/%(id)s.%(ext)s' -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' --restrict-filenames {info['url']}"
            run(cmd_dl)
            
            # get ID to find file
            id_res = subprocess.run(f"yt-dlp --get-id {info['url']}", shell=True, capture_output=True, text=True)
            if id_res.returncode == 0:
                vid_id = id_res.stdout.strip()
                dl_files = glob(f"data/raw_videos/{vid_id}*.mp4")
                if not dl_files:
                    log(f"Failed to find downloaded video for {info['url']}")
                    continue
                mp4_file = dl_files[0]
                
                # 2. Pipeline
                run(f"python scripts/run_pipeline.py '{mp4_file}' --phases 1,2,3,4", env=my_env)
                
                dirs = glob(f"data/intermediate/{vid_id}*")
                if dirs:
                    info["dir"] = dirs[0]
            else:
                continue
                
        if info["dir"] and os.path.exists(info["dir"]):
            d = info["dir"]
            run(f"cp {d}/phase4/storyboard.json {run_dir}/")
            run(f"cp {d}/summary_script.json {run_dir}/ 2>/dev/null || cp {d}/phase2/summary_script.json {run_dir}/ 2>/dev/null")
            # First try phase4/audio, then audio/
            run(f"cp {d}/phase4/audio/*.wav {run_dir}/audio/ 2>/dev/null || cp {d}/audio/*.wav {run_dir}/audio/ 2>/dev/null")
            
            # Generate dummy audio for any shot missing it
            try:
                sb = json.load(open(f"{run_dir}/storyboard.json"))
                for s in sb["shots"]:
                    shot_id = s["shot_id"]
                    wav_path = f"{run_dir}/audio/{shot_id}.wav"
                    if not os.path.exists(wav_path):
                        import wave, struct
                        with wave.open(wav_path, "w") as w:
                            w.setnchannels(1)
                            w.setsampwidth(2)
                            w.setframerate(24000)
                            # 3 seconds silence
                            w.writeframes(struct.pack("<h", 0) * (24000 * 3))
            except Exception as e:
                log(f"Failed to generate dummy audio for {vid}: {e}")
                
            ref_imgs = glob(f"{d}/phase4/concept_anchor_canonical_w02/images/*.png")
            if ref_imgs:
                run(f"cp {ref_imgs[0]} {run_dir}/reference.png")
            else:
                # If no canonical images, grab the first shot from phase4 stills
                still_imgs = glob(f"{d}/phase4/stills_*/shot_00*.png")
                if still_imgs:
                    run(f"cp {still_imgs[0]} {run_dir}/reference.png")
                else:
                    run(f"touch {run_dir}/reference.png")
                
        log(f"[{vid}] Generating I2V Prompts")
        if not os.path.exists(f"{run_dir}/storyboard.json") or "i2v_prompt" not in open(f"{run_dir}/storyboard.json").read():
            run(f"python generate_i2v_prompts.py --storyboard {run_dir}/storyboard.json --in-place", env=my_env)
        
        log(f"[{vid}] Running weight sweep")
        try:
            sb = json.load(open(f"{run_dir}/storyboard.json"))
            shots = ",".join([s["shot_id"] for s in sb["shots"]])
        except Exception as e:
            log(f"[{vid}] Could not read storyboard: {e}")
            continue
            
        if not os.path.exists(f"{run_dir}/sweep/manifest.json"):
            run(f"python weight_sweep.py --config configs/default.yaml --storyboard {run_dir}/storyboard.json --reference {run_dir}/reference.png --shots {shots} --weights 0.2,0.3,0.4,0.5,0.6,0.8 --out {run_dir}/sweep", env=my_env)
        else:
            log(f"[{vid}] Sweep manifest exists, skipping weight sweep.")
            
        log(f"[{vid}] Running collapse metrics")
        if not os.path.exists(f"{run_dir}/content_kept.csv"):
            run(f"python collapse_metrics.py --manifest {run_dir}/sweep/manifest.json --out {run_dir} --reference {run_dir}/reference.png", env=my_env)
            if os.path.exists(f"{run_dir}/collapse_metrics.csv"):
                run(f"mv {run_dir}/collapse_metrics.csv {run_dir}/content_kept.csv")
            
        log(f"[{vid}] Running DACA")
        if not os.path.exists(f"{run_dir}/daca/adaptive_anchor.csv"):
            run(f"python src/phase4/adaptive_anchor.py --manifest {run_dir}/sweep/manifest.json --metrics-csv {run_dir}/content_kept.csv --tau 0.70 --concept '{info['concept']}' --out {run_dir}/daca --clip-model openai/clip-vit-base-patch32", env=my_env)

        
        log(f"[{vid}] Assembling images")
        daca_csv = f"{run_dir}/daca/adaptive_anchor.csv"
        manifest_path = f"{run_dir}/sweep/manifest.json"
        
        if os.path.exists(daca_csv) and os.path.exists(manifest_path):
            w_star = {}
            with open(daca_csv) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].startswith("shot_"):
                        w_star[row[0].strip('"')] = float(row[1])
            man = json.load(open(manifest_path))
            img_map = {}
            if isinstance(man, list):
                for item in man:
                    img_map[(item.get("shot_id", "shot"), round(float(item["weight"]), 4))] = item["path"]
            else:
                for r in man["rows"]:
                    label = r.get("label", "shot")
                    for c in r["cells"]:
                        img_map[(label, round(float(c["weight"]), 4))] = c["image"]
            
            os.makedirs(f"{run_dir}/images_fixed_w02", exist_ok=True)
            os.makedirs(f"{run_dir}/images_daca", exist_ok=True)
            
            for shot, w in w_star.items():
                fixed_path = img_map.get((shot, 0.2))
                daca_path = img_map.get((shot, w))
                if fixed_path and os.path.exists(fixed_path):
                    shutil.copy(fixed_path, f"{run_dir}/images_fixed_w02/{shot}.png")
                if daca_path and os.path.exists(daca_path):
                    shutil.copy(daca_path, f"{run_dir}/images_daca/{shot}.png")

def phase_b():
    log("=== PHASE B ===")
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = "."
    
    for vid in ["geology", "ecology", "photosynthesis", "iphone"]:
        run_dir = f"runs/{vid}"
        if not os.path.exists(f"{run_dir}/storyboard.json"):
            continue
        log(f"[{vid}] Rendering DACA")
        run(f"python render_summary_video.py --all-i2v --storyboard {run_dir}/storyboard.json --script {run_dir}/summary_script.json --images-dir {run_dir}/images_daca --audio-dir {run_dir}/audio --work {run_dir}/clips_daca --final {run_dir}/video_daca.mp4 --workflow scripts/wan_i2v_workflow.json", env=my_env)
        
    for vid in ["geology", "ecology", "photosynthesis", "iphone"]:
        run_dir = f"runs/{vid}"
        if not os.path.exists(f"{run_dir}/storyboard.json"):
            continue
        log(f"[{vid}] Rendering FIXED")
        run(f"python render_summary_video.py --all-i2v --storyboard {run_dir}/storyboard.json --script {run_dir}/summary_script.json --images-dir {run_dir}/images_fixed_w02 --audio-dir {run_dir}/audio --work {run_dir}/clips_fixed --final {run_dir}/video_fixed_w02.mp4 --workflow scripts/wan_i2v_workflow.json", env=my_env)

if __name__ == "__main__":
    phase_a()
    phase_b()
    log("DONE")

import os
import json
import time
import argparse
from pathlib import Path
import yaml

from src.phase4.comfyui_client import ComfyUIClient
from src.phase4.video_gen import generate_clip, determine_input_paths

def run_batch_for_policy(
    video_id: str,
    policy_name: str,
    config: dict,
    checkpoint_path: Path,
    client: ComfyUIClient,
    start_shot: str = None,
    end_shot: str = None
) -> dict:
    
    base_path = Path(f"data/intermediate/{video_id}/phase4")
    policy_dir = base_path / policy_name
    clips_dir = policy_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    storyboard_path = base_path / "storyboard.json"
    anchors_path = policy_dir / "storyboard_with_anchors.json"
    
    with open(storyboard_path) as f:
        storyboard = json.load(f)
    
    with open(anchors_path) as f:
        anchors_data = json.load(f)
        
    with open("scripts/wan_flf2v_workflow.json") as f:
        workflow_template = json.load(f)
        
    anchor_dict = {a["shot_id"]: a for a in anchors_data["shots"]}
    
    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
            
    shots = storyboard["shots"]
    
    rendered_clips = {}
    for s in shots:
        sid = s["shot_id"]
        cp = clips_dir / f"{sid}.webp"
        if cp.exists() and cp.stat().st_size > 1024:
            rendered_clips[sid] = cp

    stats = {"total": len(shots), "success": 0, "skipped": 0, "failed": 0, "wall_time_sec": 0, "failed_shots": []}
    
    start_wall_time = time.time()
    
    shots_since_health_check = 0
    health_check_n = config.get("phase4", {}).get("video_gen", {}).get("recovery", {}).get("health_check_every_n_shots", 5)
    
    for i, shot in enumerate(shots):
        shot_id = shot["shot_id"]
        
        if start_shot and shot_id < start_shot:
            continue
        if end_shot and shot_id > end_shot:
            continue
            
        target_path = clips_dir / f"{shot_id}.webp"
        if target_path.exists() and target_path.stat().st_size > 1024:
            print(f"Skipping {shot_id}, already exists.")
            rendered_clips[shot_id] = target_path
            stats["skipped"] += 1
            continue
            
        print(f"Processing {shot_id}...")
        
        shots_since_health_check += 1
        if shots_since_health_check >= health_check_n:
            print("Performing health check...")
            sys_stats = client.system_stats()
            vram_gb = 0
            if "devices" in sys_stats and len(sys_stats["devices"]) > 0:
                vram_gb = sys_stats["devices"][0].get("vram_used", 0) / (1024**3)
            
            if not sys_stats or vram_gb > 20.0:
                print(f"Health check failed (VRAM {vram_gb:.2f} GB). Restarting server...")
                client.kill_server()
                time.sleep(5)
                client.start_server()
            shots_since_health_check = 0
            
        anchor_decision = anchor_dict.get(shot_id)
        if not anchor_decision:
            anchor_decision = {"shot_id": shot_id, "anchor_decision": "RESET"}
            
        first_frame, last_frame, mode = determine_input_paths(
            shot_id, anchor_decision, policy_dir, base_path, rendered_clips
        )
        
        seed = 42 + i
        
        retries = config.get("phase4", {}).get("video_gen", {}).get("recovery", {}).get("max_oom_retries", 2)
        success = False
        
        for attempt in range(retries + 1):
            res = generate_clip(client, shot, first_frame, last_frame, target_path, workflow_template, seed)
            if res["status"] == "success":
                rendered_clips[shot_id] = target_path
                stats["success"] += 1
                success = True
                break
            elif res["status"] == "oom":
                print(f"OOM on {shot_id}. Retrying {attempt+1}/{retries}...")
                client.free_vram()
                client.kill_server()
                client.start_server()
            else:
                print(f"Error on {shot_id}: {res.get('error')}")
                break
                
        if config.get("phase4", {}).get("video_gen", {}).get("recovery", {}).get("free_vram_between_shots", True):
            client.free_vram()
            
        if not success:
            stats["failed"] += 1
            stats["failed_shots"].append(shot_id)
        else:
            checkpoint = {
                "policy": policy_name,
                "last_completed_shot_id": shot_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": res
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
                
    stats["wall_time_sec"] = time.time() - start_wall_time
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--policy", default="all", choices=["all", "always_chain", "never_chain", "fixed_interval", "semantic_triggered"])
    parser.add_argument("--start-shot", default=None)
    parser.add_argument("--end-shot", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
        
    policies = ["always_chain", "never_chain", "fixed_interval", "semantic_triggered"]
    if args.policy != "all":
        policies = [args.policy]
        
    client = ComfyUIClient()
    if not args.dry_run:
        client.start_server()
        
    try:
        for p in policies:
            print(f"=== Starting policy: {p} ===")
            ckpt_path = Path(f"data/intermediate/{args.video_id}/phase4/{p}/_checkpoint.json")
            if not args.dry_run:
                stats = run_batch_for_policy(args.video_id, p, config, ckpt_path, client, args.start_shot, args.end_shot)
                print(f"Policy {p} done: {stats}")
    finally:
        client.kill_server()

if __name__ == "__main__":
    main()

#!/usr/bin/env python
import os
import sys
import yaml
import time
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.pipeline import VideoSummarizerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_cascade_scale_10")

def main():
    print("\n" + "="*80)
    print("RUNNING SCALE-10 EVALUATION: SOTA CASCADE ENTITY VERIFICATION GATING")
    print("="*80 + "\n")

    # Load configuration
    config_path = Path("configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Force enable cascade verification and ensure AWQ model is used for fast evaluation
    config["phase4"]["enable_cascade_verification"] = True
    config["models"]["qwen_vl"]["model_name"] = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

    pipeline = VideoSummarizerPipeline(config)

    video_ids = [f"review_{i}" for i in range(1, 11)]
    results = []

    start_total = time.time()

    for idx, video_id in enumerate(video_ids):
        video_path = Path("data/eval_videos") / f"{video_id}.mp4"
        assignments_path = Path("data/intermediate") / video_id / "p4_assignments.json"

        if not video_path.exists():
            logger.warning(f"Video {video_id} not found at {video_path}. Skipping.")
            continue

        # Force rerun Phase 4 by removing the existing p4_assignments.json
        if assignments_path.exists():
            logger.info(f"Removing old assignments for {video_id} to force rerun...")
            assignments_path.unlink()

        logger.info(f"[{idx+1}/10] Running Phase 4 Gating for {video_id}...")
        try:
            start_run = time.time()
            # Run pipeline up to Phase 4 (isolated gating run)
            pipeline.run(video_path, method="grouping_gate", stop_after_phase=4)
            dur = time.time() - start_run

            # Parse fresh p4_assignments.json to compute stats
            if assignments_path.exists():
                with open(assignments_path, "r") as f:
                    assignments = json.load(f)
                
                total_groups = len(assignments)
                retrieve_count = sum(1 for a in assignments if a["action"] == "retrieve")
                generate_count = sum(1 for a in assignments if a["action"] == "generate")
                
                # Check for Qwen-VL overrides:
                # An override occurred if action is generate but best_similarity >= gate_threshold (0.12)
                override_count = 0
                for a in assignments:
                    if a["action"] == "generate" and a["best_similarity"] >= 0.12:
                        override_count += 1

                results.append({
                    "video_id": video_id,
                    "status": "success",
                    "total_groups": total_groups,
                    "retrieve_count": retrieve_count,
                    "generate_count": generate_count,
                    "override_count": override_count,
                    "retrieve_percent": float(retrieve_count / total_groups) if total_groups > 0 else 0.0,
                    "duration_seconds": dur
                })
                logger.info(f"-> {video_id} completed: {retrieve_count} Retrievals, {generate_count} Generations ({override_count} overrides by Qwen-VL).")
            else:
                logger.error(f"-> {video_id} completed but p4_assignments.json was not generated.")
                results.append({"video_id": video_id, "status": "missing_assignments"})

        except Exception as e:
            logger.error(f"-> {video_id} failed with error: {e}", exc_info=True)
            results.append({"video_id": video_id, "status": "failed", "error": str(e)})

    dur_total = time.time() - start_total
    
    # Save final report to results directory
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = out_dir / "cascade_gating_scale_10_results.json"
    with open(report_json_path, "w") as f:
        json.dump({
            "total_duration_seconds": dur_total,
            "results": results
        }, f, indent=2)

    # Print final Markdown Table for user's Thesis
    print("\n" + "="*80)
    print("SOTA CASCADE ENTITY VERIFICATION GATING: SCALE-10 EVALUATION REPORT")
    print("="*80)
    print("| Video ID | Total Segments | Retrievals | Generations | Qwen-VL Rejections (Overrides) | Retrieve % | Status |")
    print("|---|---|---|---|---|---|---|")
    
    total_seg = 0
    total_ret = 0
    total_gen = 0
    total_ovr = 0
    
    for r in results:
        if r.get("status") == "success":
            total_seg += r["total_groups"]
            total_ret += r["retrieve_count"]
            total_gen += r["generate_count"]
            total_ovr += r["override_count"]
            print(f"| {r['video_id']} | {r['total_groups']} | {r['retrieve_count']} | {r['generate_count']} | {r['override_count']} | {r['retrieve_percent']*100:.1f}% | Success |")
        else:
            print(f"| {r['video_id']} | - | - | - | - | - | Failed |")
            
    print("|---|---|---|---|---|---|---|")
    macro_pct = (total_ret / total_seg) * 100 if total_seg > 0 else 0.0
    print(f"| **MACRO AVERAGE** | **{total_seg}** | **{total_ret}** | **{total_gen}** | **{total_ovr}** | **{macro_pct:.1f}%** | - |")
    print("="*80)
    print(f"Grand Total Time: {dur_total:.2f} seconds.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

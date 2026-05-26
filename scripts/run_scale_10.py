import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    start_total = time.time()
    
    videos_p1to3 = list(range(2, 11))
    videos_p4to5 = list(range(1, 11))
    
    success_p1to3 = []
    failed_p1to3 = []
    
    success_p4to5 = []
    failed_p4to5 = []
    
    os.makedirs("logs", exist_ok=True)
    
    print("=== Phase A: Restore Phase 1-3 (9 videos) ===")
    start_phase_A = time.time()
    for i in videos_p1to3:
        video_file = f"data/eval_videos/review_{i}.mp4"
        log_file = f"logs/p1to3_review_{i}.log"
        print(f"Running Phase 1-3 for review_{i}...")
        
        cmd = f"source /home/wins053/miniconda3/bin/activate sumarizer && python scripts/run_pipeline.py {video_file} --phases 1,2,3"
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT)
            
        if result.returncode == 0:
            success_p1to3.append(f"review_{i}")
        else:
            failed_p1to3.append(f"review_{i} (exit {result.returncode})")
            
    dur_A = time.time() - start_phase_A
    print(f"Phase A completed in {dur_A/3600:.2f} hours")
    
    print("=== Phase B: Full Pipeline Phase 4-5 (10 videos) ===")
    start_phase_B = time.time()
    for i in videos_p4to5:
        video_file = f"data/eval_videos/review_{i}.mp4"
        log_file = f"logs/pipeline_review_{i}.log"
        print(f"Running Phase 4-5 for review_{i}...")
        
        cmd = f"source /home/wins053/miniconda3/bin/activate sumarizer && python scripts/run_pipeline.py {video_file} --method grouping_gate"
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT)
            
        if result.returncode == 0:
            success_p4to5.append(f"review_{i}")
        else:
            failed_p4to5.append(f"review_{i} (exit {result.returncode})")
            
    dur_B = time.time() - start_phase_B
    dur_total = time.time() - start_total
    
    print(f"Phase B completed in {dur_B/3600:.2f} hours")
    print(f"Grand total time: {dur_total/3600:.2f} hours")
    
    report = f"""## Phase A: Restore Phase 1-3
- Successfully completed: {len(success_p1to3)}/9 videos
- Failed: {', '.join(failed_p1to3) if failed_p1to3 else 'None'}
- Wallclock: {int(dur_A//3600)} hours {int((dur_A%3600)//60)} minutes

## Phase B: Full pipeline
- Successfully completed: {len(success_p4to5)}/10 videos
- Failed: {', '.join(failed_p4to5) if failed_p4to5 else 'None'}
- Wallclock: {int(dur_B//3600)} hours {int((dur_B%3600)//60)} minutes

## Failures / anomalies
Phase A Fails: {failed_p1to3}
Phase B Fails: {failed_p4to5}

## Total wallclock
- Phase A: {dur_A/3600:.2f} hours
- Phase B: {dur_B/3600:.2f} hours
- Grand total: {dur_total/3600:.2f} hours

## Output files
"""
    for i in range(1, 11):
        out_path = Path(f"data/output/review_{i}/summary_grouping_gate.mp4")
        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024)
            report += f"- review_{i}: {out_path} ({size_mb:.2f} MB)\n"
        else:
            report += f"- review_{i}: Missing\n"
            
    report += "\n## Ready for user visual review\n\"Please review samples: data/output/review_*/summary_grouping_gate.mp4\"\n"
    
    with open("scale_10_report.md", "w") as f:
        f.write(report)
        
    print("Report written to scale_10_report.md")

if __name__ == "__main__":
    main()

import signal
import sys
import subprocess

# Ignore SIGINT
signal.signal(signal.SIGINT, signal.SIG_IGN)

print("Starting pipeline...", flush=True)
subprocess.run(
    ["python", "scripts/run_pipeline.py", "data/raw_videos/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge.mp4"],
    env={"PYTHONPATH": ".", **dict(sys.modules["os"].environ)}
)
print("Pipeline finished.", flush=True)

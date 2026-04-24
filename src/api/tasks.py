import uuid
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import yaml

from src.pipeline import VideoSummarizerPipeline
from src.api.websocket import manager, ProgressCallback
from src.utils.io import load_json_as_model
from src.schemas import Phase5Output

logger = logging.getLogger(__name__)

# In-memory job store
# { job_id: { status, current_phase, progress_pct, ... } }
JOBS = {}

# Single worker executor to avoid concurrent GPU usage
executor = ThreadPoolExecutor(max_workers=1)

def run_pipeline_task(job_id: str, video_path: Path, config_path: str, method: str):
    job = JOBS[job_id]
    job["status"] = "processing"
    job["start_time"] = time.time()
    
    # Progress monitoring thread
    stop_monitor = threading.Event()
    def monitor_progress():
        while not stop_monitor.is_set():
            if job["status"] == "processing":
                job["elapsed_seconds"] = int(time.time() - job["start_time"])
                # Broadcast heartbeat every 5 seconds
                import asyncio
                message = {
                    "phase": job.get("current_phase"),
                    "name": job.get("phase_name"),
                    "progress_pct": job.get("progress_pct"),
                    "detail": job.get("phase_details"),
                    "elapsed": job["elapsed_seconds"]
                }
                try:
                    # This is tricky from a separate thread, but we'll try
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(manager.broadcast_to_job(job_id, message))
                except Exception:
                    pass
            time.sleep(5)

    monitor_thread = threading.Thread(target=monitor_progress)
    monitor_thread.start()

    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        callback = ProgressCallback(job_id, manager, JOBS)
        pipeline = VideoSummarizerPipeline(config)
        
        # Run pipeline
        output = pipeline.run(video_path, method=method, progress_callback=callback)
        
        # Success
        job["status"] = "completed"
        job["output_path"] = output.output_path
        
        # Broadcast final
        import asyncio
        final_message = {
            "status": "completed",
            "output_path": output.output_path,
            "metadata": output.model_dump()
        }
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.broadcast_to_job(job_id, final_message))
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job["status"] = "failed"
        job["error"] = str(e)
        
        # Broadcast error
        import asyncio
        error_message = {
            "status": "failed",
            "error": str(e)
        }
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.broadcast_to_job(job_id, error_message))
        except Exception:
            pass

    finally:
        stop_monitor.set()
        monitor_thread.join()
        # Cleanup raw video
        if video_path.exists():
            try:
                video_path.unlink()
                logger.info(f"Deleted raw upload: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {video_path}: {e}")

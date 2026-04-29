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
from src.exceptions import JobCancelledError

logger = logging.getLogger(__name__)

# In-memory job store
# { job_id: { status, current_phase, progress_pct, ... } }
JOBS = {}

# Single worker executor to avoid concurrent GPU usage
executor = ThreadPoolExecutor(max_workers=1)

def run_pipeline_task(job_id: str, video_path: Path, config_path: str, method: str, 
                      tts_backend: str = "kokoro", llm_backend: str = "groq", 
                      style: str = "informative", target_duration: int = 90,
                      original_filename: str = None):
    job = JOBS[job_id]
    job["status"] = "processing"
    job["start_time"] = time.time()
    
    logger.info(f"Starting Job {job_id} | Method: {method} | LLM: {llm_backend} | TTS: {tts_backend} | Target: {target_duration}s")
    
    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override with user choices
        if "llm" not in config: config["llm"] = {}
        config["llm"]["backend"] = llm_backend
        
        if "tts" not in config: config["tts"] = {}
        config["tts"]["backend"] = tts_backend
        
        if "summarization" not in config: config["summarization"] = {}
        config["summarization"]["max_output_duration_seconds"] = target_duration
        
        callback = ProgressCallback(job_id, manager, JOBS)
        pipeline = VideoSummarizerPipeline(config)
        
        # Run pipeline
        if method == "all":
            from src.eval.run_ablation import AblationRunner
            runner = AblationRunner(config)
            _, out_paths_global = runner.run([video_path], ["random", "caption_cosine", "siglip_direct"], progress_callback=callback, original_filename=original_filename)
            out_paths = out_paths_global[video_path.stem]
            job["outputs"] = out_paths
            output_path = out_paths.get("siglip_direct", list(out_paths.values())[0])
            metadata = {}
        else:
            output = pipeline.run(video_path, method=method, progress_callback=callback, original_filename=original_filename)
            job["outputs"] = {method: output.output_path}
            output_path = output.output_path
            metadata = output.model_dump()
            
        # Success
        job["status"] = "completed"
        job["output_path"] = str(output_path)
        
        # Broadcast final - simplified
        import asyncio
        final_message = {
            "status": "completed",
            "output_path": str(output_path),
            "metadata": metadata,
            "job_id": job_id
        }
        # Update JOBS state for polling fallback
        job.update(final_message)

    except JobCancelledError:
        logger.info(f"Job {job_id} was cancelled by user.")
        job["status"] = "cancelled"
        
        # Broadcast cancellation
        import asyncio
        cancel_message = {
            "status": "cancelled",
            "message": "Job cancelled by user"
        }
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.broadcast_to_job(job_id, cancel_message))
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
        # Cleanup
        pass
        # Cleanup raw video
        if video_path.exists():
            try:
                video_path.unlink()
                logger.info(f"Deleted raw upload: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {video_path}: {e}")

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import uuid
import shutil
import os
import json
import pandas as pd
import glob
from typing import List, Optional

import logging
# Suppress noisy transformers logs
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from src.api.models import (
    SummarizeRequest, JobStatusResponse, JobResultResponse, 
    EvalDashboardResponse, ResultOutput, JobPhaseInfo, RecentJob
)
from src.api.tasks import JOBS, executor, run_pipeline_task
from src.api.websocket import manager
from src.utils.io import load_json_as_model
from src.schemas import Phase5Output, SummaryScript

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/summarize")
async def post_summarize(
    file: UploadFile = File(...),
    retrieval_method: str = Form("siglip_direct"),
    style: str = Form("informative"),
    subtitles: str = Form("none"),
    tts_backend: str = Form("kokoro"),
    llm_backend: str = Form("groq"),
    force: bool = Form(False)
):
    # 1. Validate
    if not file.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    job_id = str(uuid.uuid4())
    original_stem = Path(file.filename).stem
    # Sanitize stem: remove non-alphanumeric except - and _
    import re
    original_stem = re.sub(r'[^\w\-]', '_', original_stem)
    
    raw_path = Path("data/raw") / f"{job_id}.mp4"
    
    # 2. Save file
    try:
        import anyio
        def save_file():
            with raw_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        await anyio.to_thread.run_sync(save_file)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    
    # 3. Initialize job state
    JOBS[job_id] = {
        "job_id": job_id,
        "method": retrieval_method,
        "status": "pending",
        "current_phase": 0,
        "phase_name": "Initializing",
        "progress_pct": 0,
        "phase_details": "Queued for processing",
        "elapsed_seconds": 0,
        "phases_completed": [],
        "error": None,
        "start_time": None,
        "config": {
            "retrieval_method": retrieval_method,
            "style": style,
            "subtitles": subtitles,
            "tts_backend": tts_backend,
            "llm_backend": llm_backend,
            "original_filename": original_stem
        }
    }
    
    # 4. Submit to background executor
    executor.submit(
        run_pipeline_task, 
        job_id, 
        raw_path, 
        "configs/default.yaml", 
        retrieval_method,
        tts_backend,
        llm_backend,
        style,
        90, # default target_duration
        original_stem,
        force
    )
    
    return {"job_id": job_id}

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JOBS[job_id]

@router.websocket("/ws/progress/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    # Accept connection immediately to avoid handshake timeout
    await manager.connect(websocket, job_id)
    logger.info(f"WebSocket connected for job {job_id}")
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    JOBS[job_id]["status"] = "cancelling"
    logger.info(f"Cancellation requested for job {job_id}")
    return {"status": "cancelling"}

@router.get("/result/{job_id}", response_model=JobResultResponse)
async def get_result(job_id: str):
    # Allow background recovery if JOBS is lost (server reload)
    job = JOBS.get(job_id)
    
    intermediate_dir = Path("data/intermediate") / job_id
    if not job and not intermediate_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found and no data on disk")
    
    if job and job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # If job is missing from memory but data exists, synthesize a partial job state
    if not job:
        job = {
            "job_id": job_id,
            "method": "siglip_direct", # Default
            "outputs": {}
        }
        # Try to find output videos in data/output
        output_dir = Path("data/output")
        # Match both {job_id}_summary_*.json and {job_id}/{orig}_summary_*.json (recursive)
        metadata_files = list(output_dir.glob(f"**/{job_id}*_summary_*_metadata.json"))
        
        for metadata_file in metadata_files:
            try:
                # Extract method: it's between '_summary_' and '_metadata.json'
                filename = metadata_file.name
                if "_summary_" in filename:
                    method = filename.split("_summary_")[1].replace("_metadata.json", "")
                    
                    # Find corresponding video (might have original_filename in it)
                    video_pattern = f"{job_id}*_summary_{method}.mp4"
                    video_matches = list(output_dir.glob(video_pattern))
                    if video_matches:
                        job["outputs"][method] = str(video_matches[0])
            except Exception as e:
                logger.warning(f"Failed to recover arm from {metadata_file.name}: {e}")
        
        if len(job["outputs"]) > 1:
            job["method"] = "all"
        elif len(job["outputs"]) == 1:
            job["method"] = list(job["outputs"].keys())[0]
    
    # Load data
    video_id = job_id
    
    intermediate_dir = Path("data/intermediate") / video_id
    summary_path = intermediate_dir / "summary_script.json"
    transcript_path = intermediate_dir / "transcript.json"
    
    if not summary_path.exists() or not transcript_path.exists():
        # Fallback: maybe video_id is different?
        # But based on current logic, job_id is the stem.
        raise HTTPException(status_code=404, detail="Result files not found on disk")

    try:
        summary_obj = load_json_as_model(summary_path, SummaryScript)
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
            
        full_transcript = " ".join([s["text"] for s in transcript_data["segments"]])
        transcript_excerpt = full_transcript[:500] + "..."
        
        orig_duration = transcript_data.get("duration_seconds", 100.0)
        summary_dur = sum([s.estimated_duration_seconds for s in summary_obj.sentences])
        comp_ratio = summary_dur / orig_duration if orig_duration > 0 else 0.5
        
        # Build outputs dict
        outputs = {}
        # methods_dict should be a dict of {method: path}
        methods_dict = job.get("outputs") or {job.get("method", "siglip_direct"): None}
        
        for m, m_path in methods_dict.items():
            clipscore = 0.0
            eval_path = intermediate_dir / f"eval_results_{m}.json"
            if eval_path.exists():
                try:
                    with open(eval_path, "r") as ef:
                        eval_data = json.load(ef)
                        clipscore = eval_data.get("clipscore_mean", eval_data.get("clipscore", 0.0))
                except Exception: pass
                    
            outputs[m] = ResultOutput(
                video_url=f"/api/result/{job_id}/video/{m}",
                metadata={},
                clipscore=clipscore
            )
        
        # Enrich script with similarity and timestamps
        display_arm = "siglip_direct" if "siglip_direct" in outputs else (list(outputs.keys())[0] if outputs else "siglip_direct")
        matches_path = intermediate_dir / f"scene_matches_{display_arm}.json"
        matches_dict = {}
        if matches_path.exists():
            try:
                with open(matches_path, "r") as mf:
                    m_data = json.load(mf)
                    for match in m_data.get("matches", []):
                        matches_dict[int(match["sentence_id"])] = match
            except Exception: pass

        enriched_script = []
        for s in summary_obj.sentences:
            s_dict = s.model_dump()
            match = matches_dict.get(int(s.id), {})
            # Defensive check for None/null score
            raw_score = match.get("score")
            s_dict["similarity"] = float(raw_score) if raw_score is not None else 0.0
            
            # Map source_timestamp_hint
            if s.source_timestamp_hint and len(s.source_timestamp_hint) >= 2:
                s_dict["source_start"] = float(s.source_timestamp_hint[0] or 0)
                s_dict["source_end"] = float(s.source_timestamp_hint[1] or 0)
            else:
                s_dict["source_start"] = 0.0
                s_dict["source_end"] = 0.0
            enriched_script.append(s_dict)
        
        return JobResultResponse(
            job_id=job_id,
            method=job.get("method", "siglip_direct"),
            outputs=outputs,
            summary_script=enriched_script,
            transcript_excerpt=transcript_excerpt,
            compression_ratio=float(comp_ratio),
            original_duration=float(orig_duration),
            summary_duration=float(summary_dur),
            config=job.get("config")
        )
    except Exception as e:
        import traceback
        err_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error loading results for job {job_id}: {err_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load results: {err_msg}")

@router.get("/result/{job_id}/video/{method}")
async def get_video(job_id: str, method: str):
    output_dir = Path("data/output")
    video_file = output_dir / f"{job_id}_summary_{method}.mp4" # Default candidate

    if not video_file.exists():
        # Look for it more loosely: {job_id}_{original_stem}_summary_{method}.mp4
        pattern = f"{job_id}_*_summary_{method}.mp4"
        matches = list(output_dir.glob(pattern))
        if matches:
            video_file = matches[0]

    if not video_file.exists():
        # Try alternate naming (subfolder search)
        job_dir = output_dir / job_id
        if job_dir.exists():
            # Try simple name first
            video_file = job_dir / f"summary_{method}.mp4"
            if not video_file.exists():
                # Search for original_filename pattern inside the folder
                pattern = f"*summary_{method}.mp4"
                matches = list(job_dir.glob(pattern))
                if matches:
                    video_file = matches[0]

    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
        
    download_name = video_file.name
    # If we have the job in memory, try to reconstruct a nicer name for user
    if job_id in JOBS:
        orig = JOBS[job_id]["config"].get("original_filename")
        if orig:
            download_name = f"{orig}_summary_{method}.mp4"

    return FileResponse(video_file, media_type="video/mp4", filename=download_name)

# In-memory cache for dashboard data
_DASHBOARD_CACHE = {
    "data": None,
    "last_mtime": 0
}

@router.get("/eval/dashboard", response_model=EvalDashboardResponse)
async def get_eval_dashboard():
    import anyio
    logger.info("Dashboard request received")
    
    # 1. Gather statistical results from results/
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        
    csv_files = list(results_dir.glob("**/ablation_results.csv"))
    
    # Check if we can use cache
    current_mtime = 0
    if csv_files:
        current_mtime = max(os.path.getmtime(f) for f in csv_files)
    
    output_dir = Path("data/output")
    out_mtime = os.path.getmtime(output_dir) if output_dir.exists() else 0

    if _DASHBOARD_CACHE["data"] and _DASHBOARD_CACHE["last_mtime"] == current_mtime:
        if _DASHBOARD_CACHE.get("last_out_mtime") == out_mtime:
            logger.info("Serving dashboard from cache")
            return _DASHBOARD_CACHE["data"]

    logger.info(f"Aggregating dashboard data from {len(csv_files)} files...")
    
    def gather_data():
        all_dfs = []
        for f in csv_files:
            try:
                temp_df = pd.read_csv(f)
                if not temp_df.empty:
                    all_dfs.append(temp_df)
            except Exception as e:
                logger.warning(f"Skipping corrupt CSV {f}: {e}")
                
        df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        if not df.empty:
            if "video_id" in df.columns: df["video_id"] = df["video_id"].astype(str)
            if "job_id" in df.columns: df["job_id"] = df["job_id"].astype(str)
        return df

    df = await anyio.to_thread.run_sync(gather_data)
    logger.info(f"Loaded {len(df)} rows from CSVs. Computing metrics...")
    
    arms_stats = {}
    if not df.empty and "arm" in df.columns:
        for arm in df["arm"].unique():
            arm_df = df[df["arm"] == arm]
            # Use .get() or check columns to avoid KeyErrors, and handle NaN
            stats = {
                "clipscore_mean": float(arm_df["clipscore_mean"].mean() if "clipscore_mean" in arm_df.columns else 0),
                "clipscore_std": float(arm_df["clipscore_mean"].std() if "clipscore_mean" in arm_df.columns else 0),
                "rouge_l_mean": float(arm_df["rouge_l"].mean() if "rouge_l" in arm_df.columns else 0),
                "bertscore_mean": float(arm_df["bertscore"].mean() if "bertscore" in arm_df.columns else 0),
                "processing_time": float(arm_df["total_time_sec"].mean() if "total_time_sec" in arm_df.columns else 0),
                "vram_peak": float(arm_df["peak_vram_gb"].mean() * 1024 if "peak_vram_gb" in arm_df.columns else 0),
            }
            # Clean up NaNs
            for k, v in stats.items():
                if pd.isna(v): stats[k] = 0.0
            arms_stats[str(arm)] = stats
    
    # 2. Gather master list of jobs from Statistics
    all_job_ids = set()
    if not df.empty:
        all_job_ids = set(df["video_id"].unique())
        
    # Sanitize per_video DataFrame for JSON (NaN -> None)
    per_video_data = []
    if not df.empty:
        # Replace all NaN with None
        df_clean = df.where(pd.notnull(df), None)
        per_video_data = df_clean.to_dict(orient="records")
        
    # Store combined info
    master_jobs = {} # job_id -> {timestamp, ...}
    
    # 3. Gather output jobs from data/output (might include non-evaluated ones)
    output_dir = Path("data/output")
    if not output_dir.exists(): output_dir.mkdir(parents=True, exist_ok=True)
    # Search recursively for metadata files
    job_metadata_files = sorted(list(output_dir.glob("**/*_metadata.json")), key=os.path.getmtime, reverse=True)
    
    # Add from CSVs (already loaded in df)
    if not df.empty and "video_id" in df.columns:
        # We don't have timestamps in the CSV, but we can assume they are older or use current
        for vid in df["video_id"].unique():
            vid_s = str(vid)
            master_jobs[vid_s] = {"job_id": vid_s, "timestamp": 0, "video_id": vid_s}
            
    # Add/Update from data/output (prefer these timestamps as they are more accurate for recent work)
    for meta_path in job_metadata_files:
        try:
            # Format: {job_id}_{original_name}_summary_*.json or {job_id}_summary_*.json
            # Extract the UUID (which is the first part separated by _)
            job_id_candidate = meta_path.name.split("_")[0]
            if len(job_id_candidate) == 36 and "-" in job_id_candidate:
                job_id = job_id_candidate
            else:
                # Fallback
                job_id = meta_path.name.split("_summary_")[0]
                
            ts = os.path.getmtime(meta_path)
            if job_id not in master_jobs or ts > master_jobs[job_id]["timestamp"]:
                master_jobs[job_id] = {"job_id": job_id, "timestamp": ts, "video_id": job_id}
        except Exception: continue
        
    # Sort and pick top
    recent_jobs = sorted(master_jobs.values(), key=lambda x: x["timestamp"], reverse=True)[:12]
    
    response = EvalDashboardResponse(
        videos_tested=len(all_job_ids),
        arms=arms_stats,
        per_video=per_video_data,
        recent_jobs=recent_jobs
    )
    
    # Update cache
    _DASHBOARD_CACHE["data"] = response
    _DASHBOARD_CACHE["last_mtime"] = current_mtime
    _DASHBOARD_CACHE["last_out_mtime"] = os.path.getmtime(Path("data/output")) if Path("data/output").exists() else 0
    
    return response

@router.delete("/result/{job_id}")
async def delete_job(job_id: str):
    # 1. Remove from memory
    if job_id in JOBS:
        del JOBS[job_id]
        
    # 2. Remove intermediate files
    intermediate_dir = Path("data/intermediate") / job_id
    if intermediate_dir.exists():
        shutil.rmtree(intermediate_dir)
        
    # 3. Remove output files
    output_dir = Path("data/output")
    
    # Remove job-specific output folder if exists
    job_output_dir = output_dir / job_id
    if job_output_dir.exists():
        try:
            shutil.rmtree(job_output_dir)
        except Exception as e:
            logger.warning(f"Failed to delete job output dir {job_output_dir}: {e}")

    # Pattern must match old '{job_id}_summary_*' in top level
    pattern = f"{job_id}*_summary_*"
    for matched_file in output_dir.glob(pattern):
        try:
            if matched_file.is_file():
                matched_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete legacy file {matched_file}: {e}")

    # 4. Remove from ablation results (History)
    results_dir = Path("results")
    for csv_path in results_dir.glob("**/ablation_results.csv"):
        try:
            df = pd.read_csv(csv_path)
            # Both video_id and job_id might be used in columns
            cols = df.columns
            id_col = "video_id" if "video_id" in cols else ("job_id" if "job_id" in cols else None)
            
            if id_col and (df[id_col].astype(str) == str(job_id)).any():
                df_filtered = df[df[id_col].astype(str) != str(job_id)]
                if df_filtered.empty:
                    # Delete the whole folder if no more records
                    shutil.rmtree(csv_path.parent)
                else:
                    df_filtered.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to clean up CSV {csv_path}: {e}")
            
    return {"status": "ok", "message": f"Job {job_id} deleted successfully from files and history"}

@router.get("/eval/export")
async def export_eval():
    results_dir = Path("results")
    csv_files = sorted(list(results_dir.glob("**/ablation_results.csv")), key=os.path.getmtime)
    
    if not csv_files:
        raise HTTPException(status_code=404, detail="No evaluation results found")
        
    return FileResponse(csv_files[-1], media_type="text/csv", filename="ablation_results.csv")

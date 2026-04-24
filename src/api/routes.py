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

from src.api.models import (
    SummarizeRequest, JobStatusResponse, JobResultResponse, 
    EvalDashboardResponse, ResultOutput, JobPhaseInfo
)
from src.api.tasks import JOBS, executor, run_pipeline_task
from src.api.websocket import manager
from src.utils.io import load_json_as_model
from src.schemas import Phase5Output, SummaryScript

router = APIRouter()

@router.post("/summarize")
async def post_summarize(
    file: UploadFile = File(...),
    target_duration: int = Form(90),
    retrieval_method: str = Form("siglip_direct"),
    style: str = Form("informative"),
    subtitles: str = Form("none"),
    tts_backend: str = Form("kokoro"),
    llm_backend: str = Form("groq")
):
    # 1. Validate
    if not file.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    job_id = str(uuid.uuid4())
    raw_path = Path("data/raw") / f"{job_id}.mp4"
    
    # 2. Save file
    try:
        with raw_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    
    # 3. Initialize job state
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "current_phase": 0,
        "phase_name": "Initializing",
        "progress_pct": 0,
        "phase_details": "Queued for processing",
        "elapsed_seconds": 0,
        "phases_completed": [],
        "error": None,
        "start_time": None
    }
    
    # 4. Submit to background executor
    executor.submit(
        run_pipeline_task, 
        job_id, 
        raw_path, 
        "configs/default.yaml", 
        retrieval_method
    )
    
    return {"job_id": job_id}

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JOBS[job_id]

@router.websocket("/ws/progress/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

@router.get("/result/{job_id}", response_model=JobResultResponse)
async def get_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Load data
    video_id = Path(job["output_path"]).parent.name # This might be job_id or video_stem
    # Actually our pipeline uses video_stem, but raw_path was {job_id}.mp4, so video_id is job_id
    video_id = job_id
    
    intermediate_dir = Path("data/intermediate") / video_id
    summary_path = intermediate_dir / "summary_script.json"
    transcript_path = intermediate_dir / "transcript.json"
    
    try:
        summary_obj = load_json_as_model(summary_path, SummaryScript)
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
            
        full_transcript = " ".join([s["text"] for s in transcript_data["segments"]])
        transcript_excerpt = full_transcript[:500] + "..."
        
        # We need the Phase5Output to get metadata
        # In a real app we'd save this or keep in JOBS. 
        # Here we re-load if possible or get from JOBS if we stored it.
        # Let's assume we want to support multiple arms in "all" mode eventually
        
        outputs = {}
        # Simple implementation: just the one that was run
        method = job.get("method", "siglip_direct") # We should store this in task
        
        # In run_pipeline_task, we should store more info
        # For now, mock-filling some values
        outputs[method] = ResultOutput(
            video_url=f"/static/{video_id}/{Path(job['output_path']).name}",
            metadata={},
            clipscore=0.0 # Placeholder
        )
        
        return JobResultResponse(
            job_id=job_id,
            outputs=outputs,
            summary_script=[s.model_dump() for s in summary_obj.sentences],
            transcript_excerpt=transcript_excerpt,
            compression_ratio=0.5, # Placeholder
            original_duration=100.0, # Placeholder
            summary_duration=summary_obj.sentences[-1].id if summary_obj.sentences else 0.0 # Placeholder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {e}")

@router.get("/result/{job_id}/video/{method}")
async def get_video(job_id: str, method: str):
    # This serves the actual .mp4 file
    # We find it in data/output/ or intermediate
    # According to pipeline, it's usually in data/output/
    output_dir = Path("data/output")
    video_file = list(output_dir.glob(f"*/summary_{method}.mp4")) # Adjust based on actual naming
    
    if not video_file:
        raise HTTPException(status_code=404, detail="Video not found")
        
    return FileResponse(video_file[0], media_type="video/mp4")

@router.get("/eval/dashboard", response_model=EvalDashboardResponse)
async def get_eval_dashboard():
    # Gather from results/
    results_dir = Path("results")
    csv_files = list(results_dir.glob("**/ablation_results.csv"))
    
    if not csv_files:
        return EvalDashboardResponse(videos_tested=0, arms={}, per_video=[])
    
    # Load latest or all
    all_dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(all_dfs, ignore_index=True)
    
    arms_stats = {}
    for arm in df["arm"].unique():
        arm_df = df[df["arm"] == arm]
        arms_stats[arm] = {
            "clipscore_mean": arm_df["clipscore_mean"].mean() if "clipscore_mean" in arm_df else 0,
            "clipscore_std": arm_df["clipscore_mean"].std() if "clipscore_mean" in arm_df else 0,
            "rouge_l_mean": arm_df["rouge_l"].mean() if "rouge_l" in arm_df else 0,
            "bertscore_mean": arm_df["bertscore"].mean() if "bertscore" in arm_df else 0,
        }
        
    return EvalDashboardResponse(
        videos_tested=len(df["video_id"].unique()),
        arms=arms_stats,
        per_video=df.to_dict(orient="records")
    )

@router.get("/eval/export")
async def export_eval():
    results_dir = Path("results")
    csv_files = sorted(list(results_dir.glob("**/ablation_results.csv")), key=os.path.getmtime)
    
    if not csv_files:
        raise HTTPException(status_code=404, detail="No evaluation results found")
        
    return FileResponse(csv_files[-1], media_type="text/csv", filename="ablation_results.csv")

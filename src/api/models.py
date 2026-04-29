from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

class SummarizeRequest(BaseModel):
    retrieval_method: str = "siglip_direct" # "random" | "caption_cosine" | "siglip_direct" | "all"
    style: str = "informative" # "informative" | "hook-driven" | "educational"
    subtitles: str = "none" # "burn_in" | "srt_only" | "none"
    tts_backend: str = "kokoro" # "kokoro" | "f5tts"
    llm_backend: str = "groq" # "groq" | "local"

class JobPhaseInfo(BaseModel):
    phase: int
    name: str
    duration_seconds: Optional[float] = None
    vram_peak_gb: Optional[float] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str # "pending" | "processing" | "completed" | "failed"
    current_phase: Optional[int] = None
    phase_name: Optional[str] = None
    progress_pct: int = 0
    phase_details: Optional[str] = None
    elapsed_seconds: int = 0
    phases_completed: List[JobPhaseInfo] = []
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class ResultOutput(BaseModel):
    video_url: str
    metadata: Dict[str, Any]
    clipscore: Optional[float] = None

class JobResultResponse(BaseModel):
    job_id: str
    method: str
    outputs: Dict[str, ResultOutput]
    summary_script: List[Dict[str, Any]]
    transcript_excerpt: str
    compression_ratio: float
    original_duration: float
    summary_duration: float
    config: Optional[Dict[str, Any]] = None

class EvalArmStats(BaseModel):
    clipscore_mean: float
    clipscore_std: float
    rouge_l_mean: float
    bertscore_mean: float
    processing_time: float = 0.0
    vram_peak: float = 0.0

class RecentJob(BaseModel):
    job_id: str
    timestamp: float
    video_id: str

class EvalDashboardResponse(BaseModel):
    videos_tested: int
    arms: Dict[str, EvalArmStats]
    per_video: List[Dict[str, Any]]
    recent_jobs: List[RecentJob] = []

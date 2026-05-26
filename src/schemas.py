from pydantic import BaseModel
from typing import List, Optional

class Word(BaseModel):
    word: str
    start: float
    end: float
    score: float

class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None

class TranscriptSchema(BaseModel):
    video_id: str
    duration_seconds: float
    language: str
    segments: List[Segment]
    
class SummarySentence(BaseModel):
    id: int
    text: str
    estimated_duration_seconds: float
    source_timestamp_hint: List[float] # [start, end]
    keywords: List[str]

class SummaryScript(BaseModel):
    video_id: str
    target_duration: int
    style: str
    backend_used: str
    sentences: List[SummarySentence]

class AudioSentence(BaseModel):
    id: int
    text: str
    audio_path: str
    duration_seconds: float
    rms_db: float

class AudioManifest(BaseModel):
    video_id: str
    sample_rate: int
    tts_backend: str
    sentences: List[AudioSentence]
    total_duration_seconds: float

class KeyframeScene(BaseModel):
    id: int
    start_seconds: float
    end_seconds: float
    keyframe_path: str                       # midpoint frame (backward compat)
    keyframe_timestamp: float                # midpoint timestamp (backward compat)
    multi_frame_paths: List[str] = []        # NEW: paths to all sampled frames
    multi_frame_timestamps: List[float] = [] # NEW: timestamps of all sampled frames

class KeyframesManifest(BaseModel):
    video_id: str
    scenes: List[KeyframeScene]

class AlternativeMatch(BaseModel):
    scene_id: int
    score: float

class SceneMatch(BaseModel):
    sentence_id: int
    matched_scene_id: int
    score: float
    best_frame_path: str = ""           # NEW: frame within scene that scored best
    best_frame_timestamp: float = 0.0   # NEW: timestamp of winning frame
    alternatives: List[AlternativeMatch] = []

class RetrievalOutput(BaseModel):
    video_id: str
    retrieval_method: str
    matches: List[SceneMatch]

class Phase5SegmentMetadata(BaseModel):
    sentence_id: int
    text: str
    source_scene_id: int
    best_frame_timestamp: float = 0.0
    source_time_range: List[float] # [start, end]
    audio_path: str
    group_audio_duration: float = 0.0
    similarity_score: float

class Phase5Output(BaseModel):
    video_id: str
    output_path: str
    method: str
    total_duration_seconds: float
    segments: List[Phase5SegmentMetadata]
    total_processing_time_seconds: float
    peak_vram_gb: float

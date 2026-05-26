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


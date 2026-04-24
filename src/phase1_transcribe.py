import whisperx
import torch
import logging
from pathlib import Path
from typing import Dict, Any
import json

from src.schemas import TranscriptSchema, Segment, Word
from src.utils.vram import VRAMManager
from src.utils.ffmpeg_ops import extract_audio
from src.utils.io import save_model_as_json
from src.exceptions import FFmpegError, VideoSummarizerError, NoAudioError

logger = logging.getLogger(__name__)

class TranscriptionPhase:
    def __init__(self, vram_manager: VRAMManager, config: Dict[str, Any]):
        """
        Initialize the transcription phase.
        
        Args:
            vram_manager: Initialized VRAMManager instance.
            config: Configuration dictionary (can contain model_name, etc.)
        """
        self.vram = vram_manager
        self.config = config
        self.device = "cuda"
        self.compute_type = "float16"
        self.batch_size = config.get("batch_size", 16)

    def run(self, video_path: Path, progress_callback: Any = None) -> Path:
        """
        Execute Phase 1: Audio extraction, transcription, and alignment.
        
        Args:
            video_path: Path to the input video.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Path to the generated transcript.json.
        """
        video_id = Path(video_path).stem
        video_path = Path(video_path)
        
        # 1. Setup paths
        intermediate_dir = Path(self.config.get("intermediate_dir", "data/intermediate")) / video_id
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = intermediate_dir / "audio.wav"
        transcript_path = intermediate_dir / "transcript.json"

        # 2. Extract Audio
        logger.info(f"Extracting audio from {video_path}")
        if progress_callback:
            progress_callback.update(1, "Transcription", 10, "Extracting audio via FFmpeg")
            
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            extract_audio(video_path, audio_path)
        except FFmpegError as e:
            # Check if it's a 'no audio' issue
            if "Output file does not contain any stream" in str(e) or (audio_path.exists() and audio_path.stat().st_size == 0):
                raise NoAudioError(f"Video has no audio track: {video_path}")
            raise e

        # 3. Transcribe
        def load_whisper():
            return whisperx.load_model(
                self.config.get("model_name", "large-v3"), 
                self.device, 
                compute_type=self.compute_type
            )

        if progress_callback:
            progress_callback.update(1, "Transcription", 20, "Loading WhisperX model")
            
        model = self.vram.load_model("WhisperX Transcribe", load_whisper)
        
        if progress_callback:
            progress_callback.update(1, "Transcription", 40, "Running transcription")
            
        audio = whisperx.load_audio(str(audio_path))
        language = self.config.get("language", "en")
        result = model.transcribe(audio, batch_size=self.batch_size, language=language)
        
        language = result.get("language", "en")
        logger.info(f"Detected language: {language}")

        # 4. Align
        if progress_callback:
            progress_callback.update(1, "Transcription", 70, "Aligning word timestamps")
            
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, 
                device=self.device
            )
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            logger.info("Alignment successful.")
        except Exception as e:
            logger.warning(f"Alignment failed: {e}. Falling back to segment-level timestamps.")

        # 5. Peak VRAM & Unload
        if progress_callback:
            progress_callback.update(1, "Transcription", 90, "Cleaning up VRAM")
            
        self.vram.log_peak_usage("Phase 1 - Transcription")
        self.vram.load_model("None (Cleanup)", lambda: None) # Trigger unload (from vram.py logic)

        # 6. Map to Schema
        segments = []
        for i, seg in enumerate(result["segments"]):
            words = []
            if "words" in seg:
                for w in seg["words"]:
                    words.append(Word(
                        word=w["word"],
                        start=w.get("start", seg["start"]),
                        end=w.get("end", seg["end"]),
                        score=w.get("score", 0.0)
                     ))
            
            segments.append(Segment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=words if words else None
            ))

        transcript = TranscriptSchema(
            video_id=video_id,
            duration_seconds=float(len(audio) / 16000.0),
            language=language,
            segments=segments
        )

        # 7. Save
        save_model_as_json(transcript, transcript_path)
        logger.info(f"Transcript saved to {transcript_path}")

        if progress_callback:
            progress_callback.update(1, "Transcription", 100, "Phase 1 complete")

        return transcript_path

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.phase1_transcribe import TranscriptionPhase
from src.phase2_summarize import Phase2Summarizer
from src.phase3_voiceover import Phase3Voiceover
from src.phase4_retrieve import Phase4Retrieval
from src.phase5_assemble import Phase5Assembler
from src.utils.vram import VRAMManager
from src.models.llm_wrapper import GroqBackend, LocalBackend
from src.models.tts_wrapper import KokoroBackend, F5TTSBackend
from src.schemas import Phase5Output
from src.exceptions import VideoSummarizerError

logger = logging.getLogger(__name__)

class VideoSummarizerPipeline:
    """Main orchestrator for the entire Video Summarization Pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vram_manager = VRAMManager(
            device_id=config.get("vram", {}).get("device_id", 0),
            limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
        )
        
        # Initialize LLM Backend
        llm_config = config.get("llm", {})
        if llm_config.get("backend") == "groq":
            self.llm_backend = GroqBackend(
                api_key=os.getenv("GROQ_API_KEY", ""),
                model_name=llm_config.get("groq", {}).get("model_name", "llama-3.1-70b-versatile")
            )
        else:
            self.llm_backend = LocalBackend(
                vram_manager=self.vram_manager,
                model_name=llm_config.get("local", {}).get("model_name", "Qwen/Qwen2.5-14B-Instruct-AWQ")
            )
            
        # Initialize TTS Backend
        tts_config = config.get("tts", {})
        if tts_config.get("backend") == "f5tts":
            self.tts_backend = F5TTSBackend(self.vram_manager, tts_config)
        else:
            self.tts_backend = KokoroBackend(self.vram_manager, tts_config)

    def run(self, video_path: Path, method: str = "siglip_direct") -> Phase5Output:
        """
        Run the full pipeline from raw video to final summary.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        video_id = video_path.stem
        start_time = time.time()
        
        try:
            # Phase 1: Transcription
            logger.info("--- Starting Phase 1: Transcription ---")
            p1 = TranscriptionPhase(self.vram_manager, self.config.get("models", {}).get("whisper", {}))
            transcript_path = p1.run(video_path)
            
            # Phase 2: Summarization
            logger.info("--- Starting Phase 2: Summarization ---")
            p2 = Phase2Summarizer(self.llm_backend, self.config.get("summarization", {}))
            summary_path = p2.run(transcript_path, target_duration=self.config.get("summarization", {}).get("max_output_duration_seconds", 60))
            
            # Phase 3: Voiceover
            logger.info("--- Starting Phase 3: Voiceover ---")
            p3 = Phase3Voiceover(self.tts_backend, self.config.get("tts", {}))
            audio_manifest_path = p3.run(summary_path)
            
            # Phase 4: Retrieval
            logger.info("--- Starting Phase 4: Retrieval ---")
            # Phase4Retrieval.run returns a dict of results for all methods, but also saves them.
            # We need the summary object for p4.
            from src.utils.io import load_json_as_model
            from src.schemas import SummaryScript
            summary = load_json_as_model(summary_path, SummaryScript)
            
            p4 = Phase4Retrieval(self.vram_manager)
            p4_results = p4.run(video_path, summary)
            
            retrieval_output_path = Path("data/intermediate") / video_id / f"scene_matches_{method}.json"
            keyframes_manifest_path = Path("data/intermediate") / video_id / "keyframes_manifest.json"
            
            # Phase 5: Assembly
            logger.info("--- Starting Phase 5: Assembly ---")
            p5 = Phase5Assembler(self.config, self.vram_manager)
            output = p5.run(
                video_path, 
                audio_manifest_path, 
                keyframes_manifest_path, 
                retrieval_output_path
            )
            
            logger.info(f"Pipeline complete in {time.time() - start_time:.2f} seconds.")
            return output
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise VideoSummarizerError(f"Pipeline execution failed: {e}")

import os

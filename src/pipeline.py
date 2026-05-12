import logging
import time
import os
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
from src.exceptions import VideoSummarizerError, JobCancelledError

logger = logging.getLogger(__name__)

class VideoSummarizerPipeline:
    """Main orchestrator for the entire Video Summarization Pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Global Seed Initialization
        import random
        import numpy as np
        import torch
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
            
        # TTS Backend selection is now handled within Phase3Voiceover
        pass

    def run(self, video_path: Path, method: str = "siglip_direct", force: bool = False, progress_callback: Any = None, original_filename: str = None) -> Phase5Output:
        """
        Run the full pipeline from raw video to final summary.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        video_id = video_path.stem
        
        # Handle force restart by clearing intermediate data
        intermediate_dir = Path(self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")) / video_id
        if force and intermediate_dir.exists():
            import shutil
            logger.info(f"Force flag detected. Clearing existing intermediate data for {video_id}...")
            shutil.rmtree(intermediate_dir)
            
        start_time = time.time()
        
        try:
            # Phase 1: Transcription
            whisper_model = self.config.get("models", {}).get("whisper", {}).get("model_size", "large-v3")
            logger.info(f"--- Phase 1: Transcription (Model: WhisperX {whisper_model}) ---")
            if progress_callback:
                progress_callback.update(1, "Transcription", 0, "Starting audio extraction")
            
            transcript_path = Path(self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")) / video_id / "transcript.json"
            if transcript_path.exists():
                logger.info(f"Skipping Phase 1, using existing transcript: {transcript_path}")
                if progress_callback:
                    progress_callback.update(1, "Transcription", 100, "Loaded existing transcript")
            else:
                p1 = TranscriptionPhase(self.vram_manager, self.config.get("models", {}).get("whisper", {}))
                transcript_path = p1.run(video_path, progress_callback=progress_callback)
            self.vram_manager.log_peak_usage("Phase 1: Transcription")
            
            # Phase 2: Summarization
            llm_name = getattr(self.llm_backend, "model_name", "Unknown")
            logger.info(f"--- Phase 2: Summarization (Model: {llm_name}) ---")
            if progress_callback:
                progress_callback.update(2, "Summarization", 0, "Calling LLM for script generation")
            
            summary_path = Path(self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")) / video_id / "summary_script.json"
            if summary_path.exists():
                logger.info(f"Skipping Phase 2, using existing summary: {summary_path}")
                if progress_callback:
                    progress_callback.update(2, "Summarization", 100, "Loaded existing summary")
            else:
                p2 = Phase2Summarizer(self.llm_backend, self.config.get("summarization", {}))
                summary_path = p2.run(transcript_path, target_duration=self.config.get("summarization", {}).get("max_output_duration_seconds", 60), progress_callback=progress_callback)
            self.vram_manager.log_peak_usage("Phase 2: Summarization")
            
            # Phase 3: Voiceover
            tts_backend_name = self.config.get("tts", {}).get("backend", "Unknown")
            logger.info(f"--- Phase 3: Voiceover (Backend: {tts_backend_name}) ---")
            if progress_callback:
                progress_callback.update(3, "Voiceover", 0, "Generating TTS audio segments")
            
            audio_manifest_path = Path(self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")) / video_id / "audio_manifest.json"
            if audio_manifest_path.exists():
                logger.info(f"Skipping Phase 3, using existing audio manifest: {audio_manifest_path}")
                if progress_callback:
                    progress_callback.update(3, "Voiceover", 100, "Loaded existing voiceover")
            else:
                p3 = Phase3Voiceover(self.config, self.vram_manager)
                audio_manifest_path = p3.run(summary_path, progress_callback=progress_callback)
            self.vram_manager.log_peak_usage("Phase 3: Voiceover")
            
            # Phase 4: Retrieval
            retrieval_models = {
                "random": "None (Deterministic Random)",
                "siglip_direct": "SigLIP 2 (Semantic Only)",
                "siglip_temporal": "SigLIP 2 + Temporal Guidance",
                "caption_cosine": "Qwen2.5-VL + Cosine (Semantic Only)",
                "caption_temporal": "Qwen2.5-VL + Cosine + Temporal Guidance"
            }
            active_retrieval = retrieval_models.get(method, method)
            logger.info(f"--- Phase 4: Retrieval (Model: {active_retrieval}) ---")
            if progress_callback:
                progress_callback.update(4, "Visual Retrieval", 0, "Extracting and matching keyframes")
            keyframes_manifest_path = Path(self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")) / video_id / "keyframes_manifest.json"
            retrieval_output_path = Path(self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")) / video_id / f"scene_matches_{method}.json"
            
            if retrieval_output_path.exists() and keyframes_manifest_path.exists():
                logger.info(f"Skipping Phase 4, using existing retrieval results: {retrieval_output_path}")
                if progress_callback:
                    progress_callback.update(4, "Visual Retrieval", 100, "Loaded existing retrieval results")
            else:
                from src.utils.io import load_json_as_model
                from src.schemas import SummaryScript
                summary = load_json_as_model(summary_path, SummaryScript)
                p4 = Phase4Retrieval(self.config, self.vram_manager)
                p4.run(video_path, summary, method=method, progress_callback=progress_callback)
            self.vram_manager.log_peak_usage(f"Phase 4: Retrieval ({method})")
            
            # Phase 5: Assembly
            if progress_callback:
                progress_callback.update(5, "Assembly", 0, "Cutting and muxing final video")
            
            logger.info("--- Phase 5: Assembly ---")
            p5 = Phase5Assembler(self.config, self.vram_manager)
            output = p5.run(
                video_path, 
                audio_manifest_path, 
                keyframes_manifest_path, 
                retrieval_output_path,
                progress_callback=progress_callback,
                original_filename=original_filename
            )
            self.vram_manager.log_peak_usage("Phase 5: Assembly")
            
            duration = time.time() - start_time
            logger.info(f"Pipeline complete in {duration:.2f} seconds.")
            
            if progress_callback:
                progress_callback.update(5, "Completed", 100, f"Total time: {duration:.2f}s")
                
            return output
            
        except JobCancelledError:
            # Re-raise cancellation directly to be caught by tasks.py correctly
            raise
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise VideoSummarizerError(f"Pipeline execution failed: {e}")


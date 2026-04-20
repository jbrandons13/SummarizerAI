import os
import logging
from pathlib import Path
from typing import Dict, Any, List
import soundfile as sf
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from src.schemas import SummaryScript, AudioManifest, AudioSentence
from src.models.tts_wrapper import TTSBackend, KokoroBackend, F5TTSBackend
from src.utils.io import load_json_as_model, save_model_as_json
from src.utils.vram import VRAMManager

logger = logging.getLogger(__name__)

class Phase3Voiceover:
    def __init__(self, tts_backend: TTSBackend, config: Dict[str, Any]):
        self.backend = tts_backend
        self.config = config
        self.sample_rate = config.get("sample_rate", 24000)
        self.padding_ms = config.get("padding_ms", 200)
        self.target_lufs = config.get("target_lufs", -18.0)

    def run(self, script_path: Path) -> Path:
        """
        Execute Phase 3: Text-to-Speech generation and manifest creation.
        
        Args:
            script_path: Path to the summary_script.json.
            
        Returns:
            Path to the generated audio_manifest.json.
        """
        script = load_json_as_model(script_path, SummaryScript)
        video_id = script.video_id
        
        # Setup output directory
        output_dir = script_path.parent / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = script_path.parent / "audio_manifest.json"
        
        audio_sentences = []
        total_duration = 0.0
        
        logger.info(f"Starting TTS generation for video: {video_id}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Generating Voiceover...", total=len(script.sentences))
            
            for i, sentence in enumerate(script.sentences):
                sentence_id = f"{i:03d}"
                audio_path = output_dir / f"sentence_{sentence_id}.wav"
                
                progress.update(task, description=f"[cyan]Generating: {sentence.text[:30]}...")
                
                try:
                    # Generate audio
                    # Note: duration from generate() is just an estimate, we'll re-measure
                    self.backend.generate(sentence.text, audio_path)
                    
                    # Measure actual duration and RMS
                    if not audio_path.exists() or audio_path.stat().st_size == 0:
                        logger.error(f"Failed to generate audio for sentence {i}")
                        continue
                        
                    info = sf.info(audio_path)
                    duration = info.duration
                    
                    # Calculate approximate RMS for loudness check (simple estimate)
                    # Real RMS would require reading the data
                    data, _ = sf.read(audio_path)
                    rms = float(np.sqrt(np.mean(data**2)))
                    rms_db = 20 * np.log10(rms) if rms > 0 else -100
                    
                    audio_sentences.append(AudioSentence(
                        id=i,
                        text=sentence.text,
                        audio_path=str(audio_path.relative_to(script_path.parent)),
                        duration_seconds=duration,
                        rms_db=rms_db
                    ))
                    total_duration += duration
                    
                except Exception as e:
                    logger.error(f"Error generating TTS for sentence {i}: {e}")
                    # Log and skip as per requirements
                
                progress.advance(task)

        # Create and save manifest
        manifest = AudioManifest(
            video_id=video_id,
            sample_rate=self.sample_rate,
            tts_backend=self.config.get("backend", "unknown"),
            sentences=audio_sentences,
            total_duration_seconds=total_duration
        )
        
        save_model_as_json(manifest, manifest_path)
        logger.info(f"Phase 3 complete. Manifest saved to {manifest_path}")
        logger.info(f"Total voiceover duration: {total_duration:.2f} seconds")
        
        return manifest_path

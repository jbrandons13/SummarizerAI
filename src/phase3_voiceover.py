import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import soundfile as sf
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from src.schemas import SummaryScript, AudioManifest, AudioSentence
from src.models.tts_wrapper import TTSBackend, KokoroBackend, F5TTSBackend
from src.utils.io import load_json_as_model, save_model_as_json
from src.utils.vram import VRAMManager
from src.exceptions import TTSError

def clean_for_tts(text):
    import re
    text = text.replace("—", ". ")
    text = text.replace("–", ". ")
    text = text.replace(";", ".")
    text = text.replace("...", ".")
    text = text.replace("(", ". ")
    text = text.replace(")", ". ")
    text = re.sub(r'\.{2,}', '.', text)
    replacements = {
        "%": " percent", "&": " and ",
        "e.g.": "for example", "i.e.": "that is",
        "etc.": "and so on", "vs.": "versus",
        "Dr.": "Doctor", "Mr.": "Mister",
    }
    for abbr, exp in replacements.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', exp, text)
    text = " ".join(text.split())
    if text and text[-1] not in '.!?':
        text += '.'
    return text

logger = logging.getLogger(__name__)

class Phase3Voiceover:
    def __init__(self, config: Dict[str, Any], vram_manager: Optional[VRAMManager] = None):
        self.config = config
        self.vram_manager = vram_manager
        self.backend = self._get_backend(config)
        self.sample_rate = config.get("tts", {}).get("sample_rate", 24000)
        self.padding_ms = config.get("tts", {}).get("silence_padding_ms", 200)
        self.target_lufs = config.get("tts", {}).get("loudness_lufs", -18.0)

    def _get_backend(self, config: Dict[str, Any]) -> TTSBackend:
        backend_name = config.get("tts", {}).get("backend", "kokoro")
        if backend_name == "kokoro":
            return KokoroBackend(config)
        elif backend_name == "f5-tts":
            return F5TTSBackend()
        else:
            raise TTSError(f"Unknown TTS backend: {backend_name}")

    def run(self, script_path: Path, progress_callback: Any = None) -> Path:
        """
        Execute Phase 3: Text-to-Speech generation and manifest creation.
        
        Args:
            script_path: Path to the summary_script.json.
            progress_callback: Optional callback for progress updates.
            
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
        
        total_sentences = len(script.sentences)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Generating Voiceover...", total=total_sentences)
            
            for i, sentence in enumerate(script.sentences):
                sentence_id = f"{i:03d}"
                audio_path = output_dir / f"sentence_{sentence_id}.wav"
                
                progress.update(task, description=f"[cyan]Generating: {sentence.text[:30]}...")
                
                pct = int((i / total_sentences) * 100)
                if progress_callback:
                    progress_callback.update(3, "Voiceover", pct, f"Sentence {i+1}/{total_sentences}")
                
                try:
                    # Clean text
                    cleaned_text = clean_for_tts(sentence.text)
                    
                    # Generate audio
                    kwargs = {}
                    if isinstance(self.backend, F5TTSBackend):
                        f5_config = self.config.get("tts", {}).get("f5", {})
                        kwargs["ref_audio"] = f5_config.get("ref_audio")
                        kwargs["ref_text"] = f5_config.get("ref_text", "")
                    
                    self.backend.generate(cleaned_text, audio_path, **kwargs)
                    
                    # Measure actual duration and RMS
                    if not audio_path.exists() or audio_path.stat().st_size == 0:
                        logger.error(f"Failed to generate audio for sentence {i}")
                        continue
                        
                    info = sf.info(audio_path)
                    duration = info.duration
                    
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
                
                progress.advance(task)

        # Create and save manifest
        manifest = AudioManifest(
            video_id=video_id,
            sample_rate=self.sample_rate,
            tts_backend=self.config.get("backend", "unknown"),
            sentences=audio_sentences,
            total_duration_seconds=total_duration
        )
        
        # Unload model after all sentences are done
        self.backend.unload()

        save_model_as_json(manifest, manifest_path)
        logger.info(f"Phase 3 complete. Manifest saved to {manifest_path}")
        
        if progress_callback:
            progress_callback.update(3, "Voiceover", 100, f"Total duration: {total_duration:.2f}s")
            
        return manifest_path

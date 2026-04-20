from abc import ABC, abstractmethod
from pathlib import Path
import os
import logging
import numpy as np
import soundfile as sf
import torch
import pyloudnorm as pyln
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TTSBackend(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: Path) -> float:
        """
        Generate audio from text and save to output_path.
        Returns the duration of the generated audio in seconds.
        """
        pass

    def normalize_audio(self, audio: np.ndarray, sample_rate: int, target_lufs: float = -18.0) -> np.ndarray:
        """Normalize audio loudness to target LUFS."""
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
        return normalized_audio

    def add_padding(self, audio: np.ndarray, sample_rate: int, padding_ms: int = 200) -> np.ndarray:
        """Add silence padding to the end of the audio."""
        padding_samples = int(sample_rate * (padding_ms / 1000.0))
        silence = np.zeros(padding_samples)
        return np.concatenate([audio, silence])

class KokoroBackend(TTSBackend):
    def __init__(self, model_path: Optional[str] = None, voices_path: Optional[str] = None):
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise ImportError("kokoro-onnx not installed. Run 'pip install kokoro-onnx onnxruntime-gpu'")
        
        # Default paths relative to project root
        self.model_path = model_path or "models/kokoro/kokoro-v0_19.onnx"
        self.voices_path = voices_path or "models/kokoro/voices.json"
        self.kokoro = None
        self.sample_rate = 24000

    def _load_model(self):
        from kokoro_onnx import Kokoro
        if not Path(self.model_path).exists():
            msg = f"Kokoro model not found at {self.model_path}. Please download it from HuggingFace (e.g., https://huggingface.co/hexgrad/Kokoro-82M/blob/main/kokoro-v0_19.onnx)"
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        if not Path(self.voices_path).exists():
            msg = f"Kokoro voices file not found at {self.voices_path}."
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        self.kokoro = Kokoro(self.model_path, self.voices_path)
        logger.info("Kokoro model loaded successfully.")

    def generate(self, text: str, output_path: Path, voice: str = "af_heart") -> float:
        """Generate English TTS using Kokoro-ONNX."""
        if self.kokoro is None:
            self._load_model()
        
        # Limit text length for ONNX performance if needed, but Kokoro handles sentences well
        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0)
        
        # Ensure it's a numpy array for processing
        samples = np.array(samples).astype(np.float32)
        
        # Normalize and pad
        samples = self.normalize_audio(samples, sample_rate)
        samples = self.add_padding(samples, sample_rate)
        
        sf.write(output_path, samples, sample_rate)
        return len(samples) / sample_rate

class F5TTSBackend(TTSBackend):
    def __init__(self, model_type: str = "F5-TTS", ckpt_path: Optional[str] = None):
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None
        self.vocoder = None
        self.sample_rate = 24000

    def _load_model(self):
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
        except ImportError:
            raise ImportError("f5-tts not installed. Run 'pip install f5-tts'")

        logger.info(f"Loading {self.model_type} model...")
        # F5-TTS will download weights to ~/.cache/huggingface if ckpt_path is None
        self.model = load_model(self.model_type, ckpt_path=self.ckpt_path)
        self.vocoder = load_vocoder()
        logger.info(f"{self.model_type} model loaded successfully.")

    def generate(self, text: str, output_path: Path, ref_audio: Optional[str] = None, ref_text: str = "") -> float:
        """Generate TTS using F5-TTS (Zero-shot cloning)."""
        if self.model is None:
            self._load_model()
        
        from f5_tts.infer.utils_infer import infer_process
        
        # F5-TTS typically requires a reference audio (at least 3-10s)
        if ref_audio is None:
            # We'll use a placeholder/error for now since it's a 'cloning' model
            msg = "F5-TTS requires a reference audio file for generation. Please provide 'ref_audio' in config or method call."
            logger.error(msg)
            raise ValueError(msg)
            
        samples, sample_rate, _ = infer_process(
            ref_audio, ref_text, text, 
            self.model, self.vocoder, 
            model_type=self.model_type
        )
        
        # Ensure numpy float32
        samples = np.array(samples).astype(np.float32)
        
        # Normalize and pad
        samples = self.normalize_audio(samples, sample_rate)
        samples = self.add_padding(samples, sample_rate)
        
        sf.write(output_path, samples, sample_rate)
        return len(samples) / sample_rate

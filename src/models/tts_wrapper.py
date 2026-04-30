from abc import ABC, abstractmethod
from pathlib import Path
import os
import logging
import numpy as np
import soundfile as sf
import torch
import pyloudnorm as pyln
from typing import Dict, Any, Optional
from src.exceptions import TTSError
from src.utils.vram import VRAMManager

logger = logging.getLogger(__name__)

class TTSBackend(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: Path) -> float:
        """
        Generate audio from text and save to output_path.
        Returns the duration of the generated audio in seconds.
        """
        pass

    def unload(self):
        """Unload model from VRAM."""
        pass

    def normalize_audio(self, audio: np.ndarray, sample_rate: int, target_lufs: float = -18.0) -> np.ndarray:
        """Normalize audio loudness to target LUFS."""
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
        return normalized_audio

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        logger.info(f"Resampling audio from {orig_sr}Hz to {target_sr}Hz")
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback to torchaudio
            import torchaudio
            import torch
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            resampled_tensor = resampler(audio_tensor)
            return resampled_tensor.squeeze(0).numpy()

    def add_padding(self, audio: np.ndarray, sample_rate: int, padding_ms: int = 200) -> np.ndarray:
        """Add silence padding to the end of the audio."""
        padding_samples = int(sample_rate * (padding_ms / 1000.0))
        silence = np.zeros(padding_samples)
        return np.concatenate([audio, silence])

class DiaBackend(TTSBackend):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.model_name = self.config.get("tts", {}).get("dia_model", "nari-labs/Dia-1.6B")
        self.target_sr = self.config.get("tts", {}).get("sample_rate", 24000)
        self.model = None

    def _load_model(self):
        try:
            from dia.model import Dia
        except ImportError:
            raise ImportError("dia-tts (nari-tts) not installed. Run 'pip install -e libs/dia'")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Dia 1.6B model ({self.model_name}) on {device}...")
        
        # Note: Dia-1.6B is ~1.6B parameters, ~10GB VRAM
        # from_pretrained handles device placement
        self.model = Dia.from_pretrained(self.model_name, device=torch.device(device))
        
        logger.info("Dia model loaded successfully.")

    def generate(self, text: str, output_path: Path) -> float:
        """Generate high-fidelity TTS using Dia 1.6B."""
        if self.model is None:
            self._load_model()
        
        # Dia is a dialogue model. Prefixing with [S1] ensures it stays in Speaker 1 mode.
        # This prevents the "changing voices" issue.
        styled_text = f"[S1] {text}"
        
        # Use a consistent audio prompt for speaker stability if available
        # You can place a 'speaker_ref.wav' in configs/ to use a custom voice.
        audio_prompt = None
        ref_path = Path("configs/dia_speaker_ref.wav")
        if ref_path.exists():
            audio_prompt = str(ref_path)
        elif Path("libs/dia/example_prompt.mp3").exists():
            # Use the library's example prompt as a stable baseline
            audio_prompt = "libs/dia/example_prompt.mp3"

        logger.debug(f"Dia generating (Speaker 1 mode): {text[:50]}...")
        
        # Use stable parameters to prevent quality degradation and voice drift
        audio = self.model.generate(
            styled_text,
            audio_prompt=audio_prompt,
            temperature=1.0,  # Lower for stability
            top_p=0.95,
            cfg_scale=3.0,
            use_torch_compile=False # Already handled or skipped for speed
        )
        
        # Audio is usually a numpy array. Ensure it's float32.
        audio = np.array(audio).astype(np.float32)
        
        # Normalize and resample
        audio = self.resample_audio(audio, 44100, self.target_sr)
        audio = self.normalize_audio(audio, self.target_sr)
        audio = self.add_padding(audio, self.target_sr)
        
        import soundfile as sf
        sf.write(output_path, audio, self.target_sr)
        return len(audio) / self.target_sr

    def unload(self):
        if self.model is not None:
            logger.info("Unloading Dia model from VRAM...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

class KokoroBackend(TTSBackend):
    def __init__(self, config: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None, voices_path: Optional[str] = None):
        super().__init__()
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise ImportError("kokoro-onnx not installed. Run 'pip install kokoro-onnx onnxruntime-gpu'")
        
        self.config = config or {}
        # Default paths relative to project root
        self.model_path = model_path or self.config.get("tts", {}).get("kokoro", {}).get("model_path") or "models/kokoro/kokoro-v1.0.onnx"
        self.voices_path = voices_path or self.config.get("tts", {}).get("kokoro", {}).get("voices_path") or "models/kokoro/voices-v1.0.bin"
        self.speed = self.config.get("tts", {}).get("kokoro_speed", 1.0)
        self.voice = self.config.get("tts", {}).get("kokoro_voice", "af_heart")
        self.target_sr = self.config.get("tts", {}).get("sample_rate", 24000)
        self.kokoro = None

    def _load_model(self):
        from kokoro_onnx import Kokoro
        if not Path(self.model_path).exists():
            msg = f"Kokoro model not found at {self.model_path}."
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        if not Path(self.voices_path).exists():
            msg = f"Kokoro voices file not found at {self.voices_path}."
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        # Initialize Kokoro with GPU support if available
        import onnxruntime as ort
        providers = ort.get_available_providers()
        selected_providers = []
        if "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")
        
        logger.info(f"Loading Kokoro model with providers: {selected_providers}")
        session = ort.InferenceSession(self.model_path, providers=selected_providers)
        self.kokoro = Kokoro.from_session(session, self.voices_path)
        logger.info("Kokoro model loaded successfully.")

    def generate(self, text: str, output_path: Path) -> float:
        """Generate English TTS using Kokoro-ONNX."""
        if self.kokoro is None:
            self._load_model()
        
        # Kokoro handles sentences well
        samples, sr = self.kokoro.create(text, voice=self.voice, speed=self.speed)
        
        # Ensure it's a numpy array for processing
        samples = np.array(samples).astype(np.float32)
        
        # Resample if needed (though Kokoro is usually 24000)
        samples = self.resample_audio(samples, sr, self.target_sr)
        
        # Normalize and pad
        samples = self.normalize_audio(samples, self.target_sr)
        samples = self.add_padding(samples, self.target_sr)
        
        sf.write(output_path, samples, self.target_sr)
        return len(samples) / self.target_sr

class F5TTSBackend(TTSBackend):
    def __init__(self, model_type: str = "F5-TTS", ckpt_path: Optional[str] = None):
        super().__init__()
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.model = None
        self.vocoder = None
        self.target_sr = 24000

    def _load_model(self):
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model import DiT
            from omegaconf import OmegaConf
            from importlib.resources import files
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(f"f5-tts or its dependencies (omegaconf, huggingface_hub) not installed: {e}")

        # Ensure ckpt_path is set (automatic download if missing)
        if self.ckpt_path is None:
            logger.info("No checkpoint specified for F5-TTS. Downloading default from HuggingFace...")
            try:
                self.ckpt_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_v1_Base/model_1250000.safetensors")
            except Exception as e:
                logger.error(f"Failed to download F5-TTS checkpoint: {e}")
                raise TTSError(f"F5-TTS requires a checkpoint. Automatic download failed: {e}")

        # Load official config for architecture
        config_path = str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
        if not os.path.exists(config_path):
             # Fallback to standard Base architecture if file missing
             logger.warning(f"Config file not found at {config_path}. Using fallback architecture.")
             model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        else:
            full_cfg = OmegaConf.load(config_path)
            model_cfg = dict(full_cfg.model.arch)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.model_type} model on {device}...")
        self.model = load_model(DiT, model_cfg, ckpt_path=self.ckpt_path, device=device)
        self.vocoder = load_vocoder(device=device)
        logger.info(f"{self.model_type} model loaded successfully.")

    def generate(self, text: str, output_path: Path, ref_audio: Optional[str] = None, ref_text: str = "") -> float:
        """Generate TTS using F5-TTS (Zero-shot cloning)."""
        if self.model is None:
            self._load_model()
        
        from f5_tts.infer.utils_infer import infer_process
        from huggingface_hub import hf_hub_download
        
        if ref_audio is None:
            logger.info("No reference audio provided for F5-TTS. Downloading default reference...")
            # Use a known clean reference audio from a public datasets/models
            try:
                # Using a 24kHz sample from F5-TTS examples or similar
                # Attempting to download from a known stable repo
                ref_audio = hf_hub_download(repo_id="SWivid/F5-TTS", filename="tests/test.wav", repo_type="space")
            except Exception as e:
                logger.warning(f"Failed to download default reference audio: {e}. Attempting local exhaustive search.")
                
                # Logical search for any usable wav in the workspace
                project_root = Path(__file__).parent.parent.parent
                search_paths = [
                    project_root / "test_bella.wav",
                    project_root / "video-summarizer/test_bella.wav",
                    project_root / "tests/fixtures/ref_audio.wav"
                ]
                
                for p in search_paths:
                    if p.exists():
                        ref_audio = str(p)
                        logger.info(f"Found fallback reference audio at {ref_audio}")
                        break
                
                if not ref_audio or not isinstance(ref_audio, str):
                    # Final attempt: find *any* wav in data or results
                    any_wavs = list(project_root.glob("**/*.wav"))
                    if any_wavs:
                        ref_audio = str(any_wavs[0])
                        logger.info(f"Using random fallback audio found at {ref_audio}")
                    else:
                        msg = "F5-TTS requires a reference audio file (.wav) for generation. Please place a file named 'test_bella.wav' in the project root."
                        logger.error(msg)
                        raise ValueError(msg)
            
        samples, sr, _ = infer_process(
            ref_audio, ref_text, text, 
            self.model, self.vocoder, 
            model_type=self.model_type
        )
        
        samples = np.array(samples).astype(np.float32)
        samples = self.resample_audio(samples, sr, self.target_sr)
        samples = self.normalize_audio(samples, self.target_sr)
        samples = self.add_padding(samples, self.target_sr)
        
        sf.write(output_path, samples, self.target_sr)
        return len(samples) / self.target_sr


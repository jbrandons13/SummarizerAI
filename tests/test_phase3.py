import pytest
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from src.phase3_voiceover import Phase3Voiceover
from src.models.tts_wrapper import TTSBackend
from src.schemas import AudioManifest

class MockTTSBackend(TTSBackend):
    def generate(self, text: str, output_path: Path) -> float:
        # Create a 1-second dummy audio
        sample_rate = 24000
        duration = 1.0
        samples = np.random.uniform(-0.1, 0.1, int(sample_rate * duration)).astype(np.float32)
        
        # Mock normalization and padding in the backend (as implemented in base class)
        samples = self.normalize_audio(samples, sample_rate)
        samples = self.add_padding(samples, sample_rate)
        
        sf.write(output_path, samples, sample_rate)
        return len(samples) / sample_rate

@pytest.fixture
def mock_script_path(tmp_path):
    script_data = {
        "video_id": "test_video",
        "target_duration": 90,
        "style": "informative",
        "backend_used": "groq",
        "sentences": [
            {
                "id": 0,
                "text": "This is a test sentence for the voiceover pipeline.",
                "estimated_duration_seconds": 3.5,
                "source_timestamp_hint": [0.0, 5.0],
                "keywords": ["test", "voiceover"]
            },
            {
                "id": 1,
                "text": "Second sentence to verify multi-file generation.",
                "estimated_duration_seconds": 4.0,
                "source_timestamp_hint": [5.0, 10.0],
                "keywords": ["verify", "multi-file"]
            }
        ]
    }
    path = tmp_path / "summary_script.json"
    with open(path, "w") as f:
        json.dump(script_data, f)
    return path

def test_voiceover_orchestration(mock_script_path):
    config = {
        "sample_rate": 24000,
        "padding_ms": 200,
        "target_lufs": -18.0,
        "backend": "mock"
    }
    backend = MockTTSBackend()
    orchestrator = Phase3Voiceover(backend, config)
    
    manifest_path = orchestrator.run(mock_script_path)
    
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        data = json.load(f)
        AudioManifest.model_validate(data)
        assert data["video_id"] == "test_video"
        assert len(data["sentences"]) == 2
        
    # Check if wav files exist
    audio_dir = mock_script_path.parent / "audio"
    assert (audio_dir / "sentence_000.wav").exists()
    assert (audio_dir / "sentence_001.wav").exists()
    
    # Verify duration > 1.2s (1s dummy + 0.2s padding)
    info = sf.info(audio_dir / "sentence_000.wav")
    assert info.duration >= 1.2
    
    # Total duration match
    assert data["total_duration_seconds"] >= 2.4

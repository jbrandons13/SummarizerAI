import pytest
from pathlib import Path
from src.phase1_transcribe import TranscriptionPhase
from src.utils.vram import VRAMManager
from src.schemas import TranscriptSchema
import json
import logging

# Setup basic logging for tests
logging.basicConfig(level=logging.INFO)

def test_transcription_pipeline():
    video_path = Path("tests/fixtures/tiny_video.mp4")
    assert video_path.exists(), "Test video not found. Please ensure tiny_video.mp4 exists in tests/fixtures/"

    # Use default config or passed from user request
    vram = VRAMManager(device_id=0)
    config = {
        "model_name": "large-v3",
        "intermediate_dir": "data/intermediate",
        "batch_size": 16
    }
    
    phase = TranscriptionPhase(vram, config)
    transcript_path = phase.run(video_path)
    
    assert transcript_path.exists()
    
    # Validate with Pydantic
    with open(transcript_path, 'r') as f:
        data = json.load(f)
        TranscriptSchema.model_validate(data)
    
    print(f"\nTranscript path: {transcript_path}")
    print("First 50 lines of transcript:")
    with open(transcript_path, 'r') as f:
        lines = f.readlines()
        print("".join(lines[:50]))
    
if __name__ == "__main__":
    test_transcription_pipeline()
    print("\nVerification complete.")

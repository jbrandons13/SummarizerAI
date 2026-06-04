import pytest
from src.phase4.segmenter import run_segmenter
import json
import os
import soundfile as sf
import numpy as np

def test_segmenter(tmp_path):
    video_id = "test_vid"
    video_dir = tmp_path / video_id
    video_dir.mkdir(parents=True)
    
    # 1.0 + 1.0 (merged -> 2.0)
    # 10.0 (split -> multiple < 6.0)
    script = {
        "sentences": [
            {"id": "seg_1", "text": "This is a short sentence.", "start_time": 0.0, "end_time": 1.0},
            {"id": "seg_2", "text": "This is another one.", "start_time": 1.0, "end_time": 2.0},
            {"id": "seg_3", "text": "And a very long segment. It has multiple sentences. It should definitely be split up eventually if it is too long but maybe not. Let's see what happens.", "start_time": 2.0, "end_time": 12.0}
        ]
    }
    with open(video_dir / "summary_script.json", "w") as f:
        json.dump(script, f)
        
    # Create dummy audio files
    sr = 24000
    for i, dur in enumerate([1.0, 1.0, 10.0]):
        # Generate dummy sine wave
        t = np.linspace(0, dur, int(sr * dur), False)
        tone = np.sin(440 * 2 * np.pi * t)
        sf.write(str(video_dir / f"seg_{i+1}.wav"), tone, sr)
        
    audio = {
        "sentences": [
            {"id": "seg_1", "duration_seconds": 1.0, "audio_path": "seg_1.wav"},
            {"id": "seg_2", "duration_seconds": 1.0, "audio_path": "seg_2.wav"},
            {"id": "seg_3", "duration_seconds": 10.0, "audio_path": "seg_3.wav"}
        ]
    }
    with open(video_dir / "audio_manifest.json", "w") as f:
        json.dump(audio, f)
        
    out_file = run_segmenter(video_id, 2.5, 6.0, str(tmp_path))
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        data = json.load(f)
        assert len(data["shots"]) > 0
        
        # We know seg_3 was 10.0 seconds. It should be split into multiple shots.
        seg3_shots = [s for s in data["shots"] if "seg_3" in s["source_segment_ids"]]
        seg3_total_dur = sum(s["duration_sec"] for s in seg3_shots)
        
        # Check duration assertion (sum of splits == original)
        assert abs(seg3_total_dur - 10.0) < 1e-5
        
        for s in data["shots"]:
            # Check duration constraint
            assert s["duration_sec"] <= 6.0 * 1.1
            
            # Check audio file exists
            audio_path = video_dir / s["audio_path"]
            assert audio_path.exists()
            d, r = sf.read(str(audio_path))
            # Verify duration approximately matches generated audio length
            assert abs((len(d) / r) - s["duration_sec"]) < 0.1

if __name__ == "__main__":
    pytest.main(["-s", __file__])

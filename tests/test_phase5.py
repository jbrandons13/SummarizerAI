import pytest
import json
import os
import shutil
from pathlib import Path
from src.phase5_assemble import Phase5Assembler
from src.schemas import AudioManifest, AudioSentence, KeyframesManifest, KeyframeScene, RetrievalOutput, SceneMatch
from src.utils.io import save_model_as_json

@pytest.fixture
def mock_data(tmp_path):
    video_id = "tiny_video"
    intermediate_dir = tmp_path / "intermediate" / video_id
    intermediate_dir.mkdir(parents=True)
    
    # 1. Mock Keyframes Manifest
    keyframes = KeyframesManifest(
        video_id=video_id,
        scenes=[
            KeyframeScene(id=0, start_seconds=0.0, end_seconds=2.0, keyframe_path="keyframes/scene_000.jpg", keyframe_timestamp=1.0),
            KeyframeScene(id=1, start_seconds=2.0, end_seconds=4.0, keyframe_path="keyframes/scene_001.jpg", keyframe_timestamp=3.0),
            KeyframeScene(id=2, start_seconds=4.0, end_seconds=6.0, keyframe_path="keyframes/scene_002.jpg", keyframe_timestamp=5.0)
        ]
    )
    kf_path = intermediate_dir / "keyframes_manifest.json"
    save_model_as_json(keyframes, kf_path)
    
    # 2. Mock Audio Manifest
    # Create fake audio files
    audio_dir = intermediate_dir / "audio"
    audio_dir.mkdir()
    audio_sentences = []
    for i in range(3):
        p = audio_dir / f"sentence_{i:03d}.wav"
        # Generate 1s silence using ffmpeg for valid wav
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "1.0", str(p)], capture_output=True)
        
        audio_sentences.append(AudioSentence(
            id=i,
            text=f"This is segment {i}",
            audio_path=f"audio/sentence_{i:03d}.wav",
            duration_seconds=1.0,
            rms_db=-20.0
        ))
        
    audio_manifest = AudioManifest(
        video_id=video_id,
        sample_rate=48000,
        tts_backend="mock",
        sentences=audio_sentences,
        total_duration_seconds=3.0
    )
    am_path = intermediate_dir / "audio_manifest.json"
    save_model_as_json(audio_manifest, am_path)
    
    # 3. Mock Retrieval Output
    retrieval = RetrievalOutput(
        video_id=video_id,
        retrieval_method="mock_method",
        matches=[
            SceneMatch(sentence_id=0, matched_scene_id=0, score=0.9, alternatives=[]),
            SceneMatch(sentence_id=1, matched_scene_id=1, score=0.8, alternatives=[]),
            SceneMatch(sentence_id=2, matched_scene_id=2, score=0.7, alternatives=[])
        ]
    )
    ro_path = intermediate_dir / "scene_matches_mock_method.json"
    save_model_as_json(retrieval, ro_path)
    
    return {
        "video_path": Path("tests/fixtures/tiny_video.mp4"),
        "kf_path": kf_path,
        "am_path": am_path,
        "ro_path": ro_path,
        "output_dir": tmp_path / "output"
    }

def test_phase5_assembler(mock_data):
    config = {
        "output_dir": str(mock_data["output_dir"]),
        "cleanup_temp": True,
        "tts": {"padding_ms": 200},
        "subtitle": {"enabled": True}
    }
    
    assembler = Phase5Assembler(config)
    output = assembler.run(
        mock_data["video_path"],
        mock_data["am_path"],
        mock_data["kf_path"],
        mock_data["ro_path"]
    )
    
    assert Path(output.output_path).exists()
    assert output.video_id == "tiny_video"
    assert len(output.segments) == 3
    
    # Verify metadata file exists
    metadata_json = mock_data["output_dir"] / "tiny_video_summary_mock_method_metadata.json"
    assert metadata_json.exists()
    
    # Verify video is playable (basic check with ffprobe)
    import subprocess
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", output.output_path], capture_output=True, text=True)
    duration = float(result.stdout.strip())
    # 3 segments * 1.0s + 2 * 0.2s padding = 3.4s
    assert abs(duration - 3.4) < 0.5 

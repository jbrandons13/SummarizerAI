import sys
import logging
from pathlib import Path
from src.phase5_assemble import Phase5Assembler
from src.schemas import AudioManifest, AudioSentence, KeyframesManifest, KeyframeScene, RetrievalOutput, SceneMatch
from src.utils.io import save_model_as_json
import subprocess

logging.basicConfig(level=logging.INFO)

def debug_assembly():
    tmp_path = Path("./debug_p5")
    if tmp_path.exists():
        import shutil
        shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    
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
    audio_dir = intermediate_dir / "audio"
    audio_dir.mkdir()
    audio_sentences = []
    for i in range(3):
        p = audio_dir / f"sentence_{i:03d}.wav"
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "1.0", str(p)], capture_output=True)
        audio_sentences.append(AudioSentence(
            id=i, text=f"Seg {i}", audio_path=f"audio/sentence_{i:03d}.wav", duration_seconds=1.0, rms_db=-20.0
        ))
    audio_manifest = AudioManifest(video_id=video_id, sample_rate=48000, tts_backend="mock", sentences=audio_sentences, total_duration_seconds=3.0)
    am_path = intermediate_dir / "audio_manifest.json"
    save_model_as_json(audio_manifest, am_path)
    
    # 3. Mock Retrieval Output
    retrieval = RetrievalOutput(video_id=video_id, retrieval_method="mock", matches=[
        SceneMatch(sentence_id=0, matched_scene_id=0, score=0.9, alternatives=[]),
        SceneMatch(sentence_id=1, matched_scene_id=1, score=0.8, alternatives=[]),
        SceneMatch(sentence_id=2, matched_scene_id=2, score=0.7, alternatives=[])
    ])
    ro_path = intermediate_dir / "scene_matches_mock.json"
    save_model_as_json(retrieval, ro_path)
    
    config = {
        "output_dir": str(tmp_path / "output"),
        "cleanup_temp": False, # Keep for inspection
        "tts": {"padding_ms": 200},
        "subtitle": {"enabled": True},
        "temp_root": str(tmp_path / "temp")
    }
    
    assembler = Phase5Assembler(config)
    output = assembler.run(Path("tests/fixtures/tiny_video.mp4"), am_path, kf_path, ro_path)
    
    print(f"Output path: {output.output_path}")
    
    # Inspect video duration
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", output.output_path], capture_output=True, text=True)
    print(f"Final duration: {result.stdout.strip()}")
    
    # Check segment files
    temp_dir = Path(config["temp_root"]) / f"{video_id}_mock"
    print(f"Checking segments in {temp_dir / 'video_segments'}")
    segments = list((temp_dir / "video_segments").glob("*.mp4"))
    print(f"Number of video segments: {len(segments)}")
    for s in segments:
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(s)], capture_output=True, text=True).stdout.strip()
        print(f"  {s.name} duration: {dur}")
        
    # Check concat files
    for f in ["concat_video_silent.mp4", "concat_audio.wav"]:
        p = temp_dir / f
        if p.exists():
            dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(p)], capture_output=True, text=True).stdout.strip()
            print(f"{f} duration: {dur}")
        else:
            print(f"{f} MISSING")

if __name__ == "__main__":
    debug_assembly()

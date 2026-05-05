import pytest
from src.eval.metrics import temporal_alignment_score
from src.schemas import SceneMatch, KeyframeScene, SummarySentence

def test_perfect_alignment_zero_error():
    """Match jatuh persis di tengah hint window → error 0."""
    matches = [SceneMatch(
        sentence_id=0, matched_scene_id=0, score=1.0,
        best_frame_path="x", best_frame_timestamp=15.0,
    )]
    summary = type("S", (), {"sentences": [SummarySentence(
        id=0, text="x", source_timestamp_hint=[10.0, 20.0],
        estimated_duration_seconds=5.0, keywords=[]
    )]})()
    manifest = type("M", (), {"scenes": [KeyframeScene(
        id=0, start_seconds=10, end_seconds=20,
        keyframe_path="x", keyframe_timestamp=15.0,
    )]})()

    result = temporal_alignment_score(matches, summary, manifest)
    print(f"Result: {result}")
    assert result["mean_temporal_error_seconds"] == 0.0
    assert result["temporal_accuracy_within_5s"] == 1.0


def test_outside_window_correct_error():
    """Match di luar window → error = jarak ke edge terdekat."""
    matches = [SceneMatch(
        sentence_id=0, matched_scene_id=0, score=1.0,
        best_frame_path="x", best_frame_timestamp=30.0,  # window [10,20], match di 30
    )]
    summary = type("S", (), {"sentences": [SummarySentence(
        id=0, text="x", source_timestamp_hint=[10.0, 20.0],
        estimated_duration_seconds=5.0, keywords=[]
    )]})()
    manifest = type("M", (), {"scenes": [KeyframeScene(
        id=0, start_seconds=25, end_seconds=35,
        keyframe_path="x", keyframe_timestamp=30.0,
    )]})()

    result = temporal_alignment_score(matches, summary, manifest)
    print(f"Result: {result}")
    assert result["mean_temporal_error_seconds"] == 10.0  # 30 - 20
    assert result["temporal_accuracy_within_5s"] == 0.0
    assert result["temporal_accuracy_within_15s"] == 1.0


def test_no_hint_skipped():
    """Sentence tanpa hint harus di-skip, bukan crash."""
    matches = [SceneMatch(
        sentence_id=0, matched_scene_id=0, score=1.0,
        best_frame_path="x", best_frame_timestamp=15.0,
    )]
    summary = type("S", (), {"sentences": [SummarySentence(
        id=0, text="x", source_timestamp_hint=[],
        estimated_duration_seconds=5.0, keywords=[]
    )]})()
    manifest = type("M", (), {"scenes": [KeyframeScene(
        id=0, start_seconds=10, end_seconds=20,
        keyframe_path="x", keyframe_timestamp=15.0,
    )]})()

    result = temporal_alignment_score(matches, summary, manifest)
    print(f"Result: {result}")
    # n_evaluated=0, semua metric -1 atau 0
    assert result.get("n_evaluated", 0) == 0

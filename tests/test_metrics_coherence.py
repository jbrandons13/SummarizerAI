import numpy as np
from src.eval.metrics import visual_coherence_score
from src.schemas import SceneMatch

def test_identical_frames_high_coherence():
    """2 match dengan embedding sama → coherence ≈ 1.0"""
    same_emb = np.random.randn(128).astype(np.float32)
    matches = [
        SceneMatch(sentence_id=0, matched_scene_id=0, score=1.0,
                   best_frame_path="a", best_frame_timestamp=5.0),
        SceneMatch(sentence_id=1, matched_scene_id=1, score=1.0,
                   best_frame_path="b", best_frame_timestamp=15.0),
    ]
    embeddings = {
        (0, 5.0): same_emb,
        (1, 15.0): same_emb,
    }

    result = visual_coherence_score(matches, embeddings)
    print(f"Result: {result}")
    assert result["visual_coherence_mean"] > 0.99


def test_orthogonal_frames_zero_coherence():
    """2 match dengan embedding orthogonal → coherence ≈ 0"""
    matches = [
        SceneMatch(sentence_id=0, matched_scene_id=0, score=1.0,
                   best_frame_path="a", best_frame_timestamp=5.0),
        SceneMatch(sentence_id=1, matched_scene_id=1, score=1.0,
                   best_frame_path="b", best_frame_timestamp=15.0),
    ]
    embeddings = {
        (0, 5.0): np.array([1.0, 0.0, 0.0], dtype=np.float32),
        (1, 15.0): np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    result = visual_coherence_score(matches, embeddings)
    print(f"Result: {result}")
    assert abs(result["visual_coherence_mean"]) < 0.01


def test_missing_embedding_keys_handled():
    """Kalau embedding key tidak ada di dict, function harus skip pasangan
    itu, bukan return 0 untuk semua."""
    matches = [
        SceneMatch(sentence_id=0, matched_scene_id=0, score=1.0,
                   best_frame_path="a", best_frame_timestamp=5.0),
        SceneMatch(sentence_id=1, matched_scene_id=1, score=1.0,
                   best_frame_path="b", best_frame_timestamp=15.0),
        SceneMatch(sentence_id=2, matched_scene_id=2, score=1.0,
                   best_frame_path="c", best_frame_timestamp=25.0),
    ]
    same_emb = np.random.randn(128).astype(np.float32)
    embeddings = {
        (0, 5.0): same_emb,
        (1, 15.0): same_emb,
        # (2, 25.0) sengaja missing
    }

    result = visual_coherence_score(matches, embeddings)
    print(f"Result: {result}")
    # Pasangan (0,1) jalan; pasangan (1,2) skip karena missing
    # n_pairs harus 1, bukan 0
    assert result.get("n_pairs", 0) >= 1

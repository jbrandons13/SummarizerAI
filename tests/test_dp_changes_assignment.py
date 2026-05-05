import pytest
import numpy as np

def test_dp_differs_from_greedy_when_jumps_costly():
    """
    Kasus dirancang sehingga greedy akan loncat tapi DP tidak.

    sim_matrix (3 sentences x 4 scenes), times = [0, 30, 60, 90]:
        Sentence 0: prefer scene 0 (0.9)
        Sentence 1: prefer scene 3 (0.8) — jauh dari sentence 0
        Sentence 2: prefer scene 1 (0.7) — kembali mundur

    Greedy: [0, 3, 1] — total skor 0.9+0.8+0.7 = 2.4, banyak loncat
    DP dengan jump penalty: should pick path lebih monoton meskipun skor
    individual lebih rendah.
    """
    sim = np.array([
        [0.9, 0.5, 0.4, 0.3],
        [0.2, 0.4, 0.6, 0.8],
        [0.3, 0.7, 0.5, 0.4],
    ])
    scenes = [
        type("S", (), {"keyframe_timestamp": 0,  "end_seconds": 10})(),
        type("S", (), {"keyframe_timestamp": 30, "end_seconds": 40})(),
        type("S", (), {"keyframe_timestamp": 60, "end_seconds": 70})(),
        type("S", (), {"keyframe_timestamp": 90, "end_seconds": 100})(),
    ]

    # Let's import directly from class method or subclass
    from src.phase4_retrieve import SigLIP2DirectRetrieval

    # Initialize just enough to access base class methods
    backend = SigLIP2DirectRetrieval(config={})
    greedy_a = backend.greedy_assign(sim)
    dp_a = backend.dp_sequence_align(sim, scenes, video_duration=100,
                            jump_penalty=1.0, reuse_bonus=0.0,
                            backward_penalty=0.5)

    print(f"Greedy: {greedy_a}")
    print(f"DP:     {dp_a}")
    assert greedy_a != dp_a, "DP should differ from greedy when jumps are penalized"


def test_dp_equals_greedy_when_penalties_zero():
    """jump_penalty=0, reuse_bonus=0, backward_penalty=0 → DP = argmax."""
    sim = np.random.rand(5, 8)
    scenes = [type("S", (), {"keyframe_timestamp": i*10, "end_seconds": i*10+10})()
             for i in range(8)]

    from src.phase4_retrieve import SigLIP2DirectRetrieval

    backend = SigLIP2DirectRetrieval(config={})
    dp_a = backend.dp_sequence_align(sim, scenes, video_duration=80,
                            jump_penalty=0.0, reuse_bonus=0.0,
                            backward_penalty=0.0)
    expected = [int(np.argmax(sim[i])) for i in range(5)]
    assert dp_a == expected, f"With zero penalties, DP must reduce to argmax. Got {dp_a}, expected {expected}"

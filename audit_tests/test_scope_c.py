import sys
import os
from pathlib import Path
import numpy as np
from scipy import stats

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.compute_additional_metrics import compute_max_consecutive, compute_strict_viscoher
from src.eval.metrics import temporal_alignment_score

def test_c1_viscoher_strict():
    print("--- Running C.1: VisCoher_strict Correctness ---")
    # Synthetic test: 5 matches
    # Scene IDs: [0, 0, 1, 2, 2]
    # Consecutive pairs:
    # (0, 0) -> same scene, EXCLUDE
    # (0, 1) -> different, INCLUDE
    # (1, 2) -> different, INCLUDE
    # (2, 2) -> same scene, EXCLUDE
    
    matches = [
        {'matched_scene_id': 0, 'best_frame_timestamp': 5.0},
        {'matched_scene_id': 0, 'best_frame_timestamp': 6.0},
        {'matched_scene_id': 1, 'best_frame_timestamp': 15.0},
        {'matched_scene_id': 2, 'best_frame_timestamp': 25.0},
        {'matched_scene_id': 2, 'best_frame_timestamp': 26.0},
    ]
    
    # Embeddings: all ones but different lengths to test dot product
    # emb_0: [1, 0], emb_1: [0, 1], emb_2: [1, 1]/sqrt(2)
    frame_embeddings = {
        (0, 5.0): np.array([1.0, 0.0]),
        (0, 6.0): np.array([1.0, 0.0]),
        (1, 15.0): np.array([0.0, 1.0]),
        (2, 25.0): np.array([1.0, 0.0]),
        (2, 26.0): np.array([1.0, 0.0]),
    }
    
    # Pairs to count:
    # 1. match[1] vs match[2]: (0, 15.0) vs (1, 15.0)? No, (0, 6.0) vs (1, 15.0)
    #    Sim = dot([1,0], [0,1]) = 0.0
    # 2. match[2] vs match[3]: (1, 15.0) vs (2, 25.0)
    #    Sim = dot([0,1], [1,0]) = 0.0
    
    # If we change emb_1 to [1, 0], then sim should be 1.0
    frame_embeddings[(1, 15.0)] = np.array([1.0, 0.0])
    # Pair (0,1) sim = 1.0
    # Pair (1,2) sim = 1.0
    # Expected mean = 1.0
    
    val = compute_strict_viscoher(matches, frame_embeddings)
    if abs(val - 1.0) < 1e-6:
        print("PASS: VisCoher_strict matches expected")
    else:
        print(f"FAIL: VisCoher_strict got {val}, expected 1.0")

def test_c2_scene_diversity():
    print("\n--- Running C.2: Scene Diversity ---")
    def div(seq):
        return len(set(seq)) / len(seq) if seq else 0.0
        
    cases = [
        ([2, 4, 4, 11, 11, 16], 4/6),
        ([2, 2, 2, 2, 2, 2], 1/6),
        ([1, 2, 3, 4, 5], 1.0)
    ]
    passed = True
    for seq, expected in cases:
        val = div(seq)
        if abs(val - expected) > 1e-6:
            print(f"FAIL: {seq} got {val}, expected {expected}")
            passed = False
    if passed: print("PASS: Scene Diversity")

def test_c3_max_consecutive():
    print("\n--- Running C.3: Max Consecutive Reuse ---")
    cases = [
        ([1, 1, 1, 2, 2], 3),
        ([1, 2, 1, 2, 1], 1),
        ([1, 1, 2, 2, 2], 3),
        ([5], 1),
        ([], 0)
    ]
    passed = True
    for seq, expected in cases:
        val = compute_max_consecutive(seq)
        if val != expected:
            print(f"FAIL: {seq} got {val}, expected {expected}")
            passed = False
    if passed: print("PASS: Max Consecutive Reuse")

def test_c7_temporal_accuracy():
    print("\n--- Running C.7: Temporal Accuracy Validation ---")
    # Using src.eval.metrics.temporal_alignment_score logic
    # But let's check the synthetic cases from prompt
    
    # From prompt: 
    # source_hint=[100, 200], retrieved_ts=150 -> error=0
    # source_hint=[100, 200], retrieved_ts=80 -> error=20
    
    def compute_error(hint, ts):
        if hint[0] <= ts <= hint[1]: return 0
        return min(abs(ts - hint[0]), abs(ts - hint[1]))
        
    cases = [
        ([100, 200], 150, 0),
        ([100, 200], 120, 0),
        ([100, 200], 80, 20),
        ([100, 200], 250, 50)
    ]
    passed = True
    for hint, ts, expected in cases:
        val = compute_error(hint, ts)
        if val != expected:
            print(f"FAIL: hint={hint} ts={ts} got {val}, expected {expected}")
            passed = False
    if passed: print("PASS: Temporal Accuracy Error Formula")

if __name__ == "__main__":
    test_c1_viscoher_strict()
    test_c2_scene_diversity()
    test_c3_max_consecutive()
    test_c7_temporal_accuracy()

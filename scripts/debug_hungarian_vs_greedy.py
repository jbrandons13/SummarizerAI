"""
Compare assignment from Greedy vs Hungarian for 1 video.
Analyze sentence/scene ratio and scene reuse.
"""
import json
import sys
from pathlib import Path
import numpy as np

def load_assignments(video_id: str, arm_name: str):
    """Load matched scene IDs from intermediate data."""
    matches_path = Path(f"data/intermediate/{video_id}/scene_matches_{arm_name}.json")
    if not matches_path.exists():
        print(f"❌ Error: Matches file not found: {matches_path}")
        return None
        
    with open(matches_path) as f:
        data = json.load(f)
    
    matches = data.get("matches", data) if isinstance(data, dict) else data
    return [(m["sentence_id"], m["matched_scene_id"], m.get("score", 0))
            for m in sorted(matches, key=lambda x: x["sentence_id"])]

def analyze(video_id: str):
    video_dir = Path(f"data/intermediate/{video_id}")
    if not video_dir.exists():
        print(f"❌ Error: Video directory not found: {video_dir}")
        return

    # --- Step 1.2a: Sentence/Scene Ratio ---
    summary_path = video_dir / "summary_script.json"
    manifest_path = video_dir / "keyframes_manifest.json"

    with open(summary_path) as f:
        summary = json.load(f)
    num_sentences = len(summary.get("sentences", []))

    with open(manifest_path) as f:
        manifest = json.load(f)
    num_scenes = len(manifest.get("scenes", []))

    print(f"\n=== PROJECT DISCOVERY: {video_id} ===")
    print(f"Number of sentences: {num_sentences}")
    print(f"Number of scenes: {num_scenes}")
    print(f"Ratio scenes/sentences: {num_scenes / num_sentences:.2f}")

    if num_scenes >= 3 * num_sentences:
        print("⚠️  Diagnosis B Candidate: scenes >> sentences. Hungarian may degenerate to Greedy.")
    elif num_scenes < 2 * num_sentences:
        print("✓ Resource constrained: Hungarian SHOULD differ from Greedy if reuse occurs.")

    # --- Step 1.1: Compare Assignments ---
    greedy = load_assignments(video_id, "siglip_temporal")
    hungarian = load_assignments(video_id, "siglip_temporal_hungarian")

    if not greedy or not hungarian:
        return

    print(f"\n{'sent':<6}{'Greedy scene':<15}{'Hungarian scene':<18}{'Match?':<10}{'G score':<10}{'H score':<10}")
    print("-" * 75)

    diff_count = 0
    for (s_id, g_scene, g_score), (_, h_scene, h_score) in zip(greedy, hungarian):
        match = "SAME" if g_scene == h_scene else "DIFF"
        if g_scene != h_scene:
            diff_count += 1
        print(f"{s_id:<6}{g_scene:<15}{h_scene:<18}{match:<10}{g_score:<10.4f}{h_score:<10.4f}")

    print(f"\nTotal differences: {diff_count} / {len(greedy)}")
    print(f"Identical assignments: {len(greedy) - diff_count} / {len(greedy)}")

    # --- Step 1.2b: Greedy Reuse Analysis ---
    greedy_scenes = [g[1] for g in greedy]
    unique_used = len(set(greedy_scenes))
    total = len(greedy_scenes)

    print(f"\n=== Greedy reuse analysis ===")
    print(f"Total assignments: {total}")
    print(f"Unique scenes used: {unique_used}")
    print(f"Reused scenes: {total - unique_used}")

    if unique_used == total:
        print("⚠️  Greedy did NOT reuse any scenes. Hungarian has no reason to differ.")
        print("   This is mathematically expected when scenes are abundant.")
    else:
        print("✓ Greedy reused scenes. Hungarian SHOULD produce different assignments.")
        print("   If Hungarian still identical, there's a bug in tile construction.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_hungarian_vs_greedy.py <video_id>")
    else:
        analyze(sys.argv[1])

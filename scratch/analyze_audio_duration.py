import sys
import os
import json
import csv
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    intermediate_dir = project_root / "data/intermediate"
    review_dirs = sorted(list(intermediate_dir.glob("review_*")))

    groups_data = []

    for review_dir in review_dirs:
        video_id = review_dir.name
        assignments_path = review_dir / "p4_assignments.json"
        manifest_path = review_dir / "audio_manifest.json"

        if not assignments_path.exists():
            print(f"[{video_id}] p4_assignments.json not found, skipping.")
            continue
        if not manifest_path.exists():
            print(f"[{video_id}] audio_manifest.json not found, skipping.")
            continue

        # Load assignments
        with open(assignments_path, "r") as f:
            assignments = json.load(f)

        # Load manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Map sentence ID to audio sentence dict
        sentences_map = {s["id"]: s for s in manifest["sentences"]}

        for group_idx, group in enumerate(assignments):
            sentence_ids = group.get("sentence_ids", [])
            action = group.get("action", "generate")

            group_duration = 0.0
            n_sentences = len(sentence_ids)
            missing_files = []

            for s_id in sentence_ids:
                if s_id not in sentences_map:
                    print(f"[{video_id}] Warning: sentence ID {s_id} not found in manifest.")
                    continue
                
                audio_sentence = sentences_map[s_id]
                rel_path = audio_sentence.get("audio_path")
                wav_path = review_dir / rel_path

                if not wav_path.exists():
                    # Fallback to absolute path check or manifest check
                    print(f"[{video_id}] Warning: WAV file not found at {wav_path}")
                    missing_files.append(str(wav_path))
                    continue

                try:
                    info = sf.info(wav_path)
                    group_duration += info.duration
                except Exception as e:
                    print(f"[{video_id}] Error reading {wav_path}: {e}")
                    # fallback to manifest duration if reading fails
                    group_duration += audio_sentence.get("duration_seconds", 0.0)

            if missing_files:
                print(f"[{video_id}] Group {group_idx} has missing WAV files: {missing_files}")
                continue

            groups_data.append({
                "video_id": video_id,
                "group_id": group_idx,
                "n_sentences": n_sentences,
                "audio_duration_s": group_duration,
                "action": action
            })

    # Save to CSV
    csv_path = project_root / "audio_duration_per_group.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "group_id", "n_sentences", "audio_duration_s", "action"])
        writer.writeheader()
        writer.writerows(groups_data)

    print(f"Successfully wrote {len(groups_data)} group records to {csv_path}")

    # Compute Statistics
    for action_type in ["generate", "retrieve"]:
        action_groups = [g for g in groups_data if g["action"] == action_type]
        n_groups = len(action_groups)

        if n_groups == 0:
            print(f"\nNo groups found for action={action_type}")
            continue

        durations = [g["audio_duration_s"] for g in action_groups]

        mean = np.mean(durations)
        median = np.median(durations)
        std = np.std(durations)
        min_val = np.min(durations)
        max_val = np.max(durations)
        p25 = np.percentile(durations, 25)
        p75 = np.percentile(durations, 75)
        p90 = np.percentile(durations, 90)
        p95 = np.percentile(durations, 95)

        # Histogram buckets
        b1 = sum(1 for d in durations if d <= 2.56)
        b2 = sum(1 for d in durations if 2.56 < d <= 5.12)
        b3 = sum(1 for d in durations if 5.12 < d <= 7.68)
        b4 = sum(1 for d in durations if 7.68 < d <= 10.24)
        b5 = sum(1 for d in durations if d > 10.24)

        fit_single = (b1 / n_groups) * 100
        fit_double = ((b1 + b2) / n_groups) * 100
        need_triple = ((b3 + b4 + b5) / n_groups) * 100

        print(f"\n==================================================")
        print(f"Distribution (action={action_type}, n={n_groups})")
        print(f"==================================================")
        print(f"Mean: {mean:.2f}s")
        print(f"Median: {median:.2f}s")
        print(f"Std: {std:.2f}s")
        print(f"Min: {min_val:.2f}s")
        print(f"Max: {max_val:.2f}s")
        print(f"p25: {p25:.2f}s")
        print(f"p75: {p75:.2f}s")
        print(f"p90: {p90:.2f}s")
        print(f"p95: {p95:.2f}s")
        print(f"\nHistogram:")
        print(f"0-2.56s (1 clip): {b1} ({b1/n_groups*100:.1f}%)")
        print(f"2.56-5.12s (2 clips): {b2} ({b2/n_groups*100:.1f}%)")
        print(f"5.12-7.68s (3 clips): {b3} ({b3/n_groups*100:.1f}%)")
        print(f"7.68-10.24s (4 clips): {b4} ({b4/n_groups*100:.1f}%)")
        print(f">10.24s (5+ clips): {b5} ({b5/n_groups*100:.1f}%)")
        print(f"\nImplications:")
        print(f"- Fit single clip (<=2.56s): {fit_single:.1f}%")
        print(f"- Fit 2 clips concat (<=5.12s): {fit_double:.1f}%")
        print(f"- Need 3+ clips (>5.12s): {need_triple:.1f}%")

if __name__ == "__main__":
    main()

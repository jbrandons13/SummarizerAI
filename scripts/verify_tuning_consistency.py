"""
Compare DP results before vs after tuning across all 3 videos.
"""
import json
from pathlib import Path

VIDEOS = ["review_1", "review_2", "review_3"]
ARMS = ["random", "caption_temporal", "siglip_direct", "siglip_temporal",
        "siglip_temporal_hungarian", "siglip_temporal_dp"]

# Adjusted keys based on discovery
KEY_CLIP = "clipscore_mean"
KEY_TEMP = "temporal_acc_30s"
KEY_VIS = "visual_coherence_mean"

def load_metrics(video_id, arm):
    path = Path(f"data/intermediate/{video_id}/eval_results_{arm}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def main():
    print(f"\n{'Video':<12}{'Arm':<30}{'CLIPScore':<12}{'TempAcc':<10}{'VisCoher':<10}")
    print("-" * 74)

    for video in VIDEOS:
        for arm in ARMS:
            m = load_metrics(video, arm)
            if m is None:
                print(f"{video:<12}{arm:<30}MISSING")
                continue
            cs = m.get(KEY_CLIP, 0)
            ta = m.get(KEY_TEMP, 0)
            vc = m.get(KEY_VIS, 0)
            print(f"{video:<12}{arm:<30}{cs:<12.4f}{ta:<10.4f}{vc:<10.4f}")
        print()

    # DP-specific analysis
    print("\n=== DP vs Greedy Analysis ===")
    for video in VIDEOS:
        greedy = load_metrics(video, "siglip_temporal")
        dp = load_metrics(video, "siglip_temporal_dp")
        if greedy and dp:
            print(f"\n{video}:")
            print(f"  CLIPScore: Greedy={greedy[KEY_CLIP]:.4f}, DP={dp[KEY_CLIP]:.4f}, Δ={dp[KEY_CLIP]-greedy[KEY_CLIP]:+.4f}")
            print(f"  TempAcc:   Greedy={greedy[KEY_TEMP]:.4f}, DP={dp[KEY_TEMP]:.4f}, Δ={dp[KEY_TEMP]-greedy[KEY_TEMP]:+.4f}")
            print(f"  VisCoher:  Greedy={greedy[KEY_VIS]:.4f}, DP={dp[KEY_VIS]:.4f}, Δ={dp[KEY_VIS]-greedy[KEY_VIS]:+.4f}")

    # Hungarian degeneracy check
    print("\n=== Hungarian vs Greedy (should be ~identical) ===")
    for video in VIDEOS:
        greedy = load_metrics(video, "siglip_temporal")
        hungarian = load_metrics(video, "siglip_temporal_hungarian")
        if greedy and hungarian:
            # Check if metrics are near-identical
            diffs = [
                abs(greedy[KEY_CLIP] - hungarian[KEY_CLIP]),
                abs(greedy[KEY_TEMP] - hungarian[KEY_TEMP]),
                abs(greedy[KEY_VIS] - hungarian[KEY_VIS])
            ]
            identical = all(d < 0.001 for d in diffs)
            print(f"  {video}: {'IDENTICAL' if identical else 'DIFFERENT'}")
            if not identical:
                print(f"    Diffs: CS={diffs[0]:.4f}, TA={diffs[1]:.4f}, VC={diffs[2]:.4f}")

if __name__ == "__main__":
    main()

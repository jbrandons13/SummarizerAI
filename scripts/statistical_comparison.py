import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("results/final_ablation_results_v2.csv")

comparisons = [
    ("caption_temporal_dp", "caption_temporal_ccma"),
    ("siglip_temporal_dp", "siglip_temporal_ccma"),
]

metrics = [
    "clipscore_mean", 
    "temporal_acc_15s", 
    "visual_coherence_mean", 
    "viscoher_strict", 
    "scene_diversity",
    "max_consecutive_reuse"
]

print(f"{'Metric':<25} | {'Arm 1 (DP)':<15} | {'Arm 2 (CCMA)':<15} | {'p-value':<10} | {'Sig':<5}")
print("-" * 80)

for arm1, arm2 in comparisons:
    print(f"\n=== {arm1} vs {arm2} ===")
    for metric in metrics:
        # Sort by video_id to ensure paired samples match
        v1 = df[df["arm"] == arm1].sort_values("video_id")[metric].values
        v2 = df[df["arm"] == arm2].sort_values("video_id")[metric].values
        
        if len(v1) == len(v2) and len(v1) > 1:
            # Paired t-test
            t, p = stats.ttest_rel(v1, v2)
            sig = "**SIG**" if p < 0.05 else "ns"
            print(f"{metric:<25} | {v1.mean():<15.4f} | {v2.mean():<15.4f} | {p:<10.4f} | {sig}")
        else:
            print(f"{metric:<25} | Insufficient data (v1:{len(v1)}, v2:{len(v2)})")

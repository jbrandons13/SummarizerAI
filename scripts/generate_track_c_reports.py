import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def main():
    csv_path = "results/cleanrun_aggregated_20260505.csv/ablation_results.csv"
    df = pd.read_csv(csv_path)
    
    # 1. Scene Diversity Report
    diversity_report = "# Scene Diversity Results\n\n"
    
    # Aggregate per arm
    agg_diversity = df.groupby("arm")["scene_diversity"].mean().reset_index()
    agg_diversity.columns = ["Arm", "Mean Scene Diversity"]
    diversity_report += "## Per-Arm Aggregate Scene Diversity\n\n"
    diversity_report += agg_diversity.to_markdown(index=False) + "\n\n"
    
    # Looping cases (consecutive_reuse >= 3)
    looping_cases = df[df["max_consecutive_reuse"] >= 3][["video_id", "arm", "max_consecutive_reuse", "scene_diversity"]]
    diversity_report += "## Looping Cases (Max Consecutive Reuse >= 3)\n\n"
    if not looping_cases.empty:
        diversity_report += looping_cases.to_markdown(index=False) + "\n\n"
    else:
        diversity_report += "No severe looping cases detected.\n\n"
        
    # Per-video breakdown
    diversity_report += "## Per-Video Breakdown\n\n"
    pivot_diversity = df.pivot(index="video_id", columns="arm", values="scene_diversity")
    diversity_report += pivot_diversity.to_markdown() + "\n\n"
    
    with open("notes/scene_diversity_results.md", "w") as f:
        f.write(diversity_report)
        
    # 2. VisCoher Strict Report
    viscoher_report = "# Visual Coherence Strict Results\n\n"
    
    # Aggregate comparison
    agg_vis = df.groupby("arm")[["visual_coherence_mean", "viscoher_strict"]].mean().reset_index()
    agg_vis["gap"] = agg_vis["visual_coherence_mean"] - agg_vis["viscoher_strict"]
    viscoher_report += "## VisCoher vs VisCoher_Strict Comparison\n\n"
    viscoher_report += agg_vis.to_markdown(index=False) + "\n\n"
    
    viscoher_report += "### Analysis\n"
    largest_gap_arm = agg_vis.loc[agg_vis["gap"].idxmax()]["arm"]
    viscoher_report += f"The largest gap is observed in the **{largest_gap_arm}** arm, indicating it is most affected by scene reuse artifacting.\n\n"
    
    # Significance tests for VisCoher_strict
    viscoher_report += "## Statistical Significance (Paired T-test for VisCoher_Strict)\n\n"
    
    key_comparisons = [
        ("caption_temporal", "caption_temporal_dp"),
        ("siglip_temporal", "siglip_temporal_dp"),
        ("caption_temporal_dp", "siglip_temporal_dp")
    ]
    
    for arm1, arm2 in key_comparisons:
        scores1 = df[df["arm"] == arm1]["viscoher_strict"].values
        scores2 = df[df["arm"] == arm2]["viscoher_strict"].values
        
        if len(scores1) == len(scores2) and len(scores1) > 1:
            t_stat, p_val = stats.ttest_rel(scores1, scores2)
            viscoher_report += f"- **{arm1} vs {arm2}**: p-value = {p_val:.4f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}\n"
        else:
            viscoher_report += f"- **{arm1} vs {arm2}**: Insufficient data\n"
            
    with open("notes/viscoher_strict_results.md", "w") as f:
        f.write(viscoher_report)
        
    print("Reports generated: notes/scene_diversity_results.md, notes/viscoher_strict_results.md")

if __name__ == "__main__":
    main()

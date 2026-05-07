import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def main():
    csv_path = "results/aggregated_20260507_083942/ablation_results.csv"
    df = pd.read_csv(csv_path)
    
    comparisons = [
        ("caption_temporal_dp", "caption_temporal_cvalign", "CV-Align vs Vanilla DP (Caption)"),
        ("siglip_temporal_dp", "siglip_temporal_cvalign", "CV-Align vs Vanilla DP (SigLIP)"),
        ("caption_temporal", "caption_temporal_cvalign", "CV-Align vs Greedy (Caption)"),
        ("siglip_temporal", "siglip_temporal_cvalign", "CV-Align vs Greedy (SigLIP)")
    ]
    
    metrics = ["clipscore_mean", "temporal_acc_15s", "visual_coherence_mean", "viscoher_strict", "scene_diversity"]
    
    report = "# Significance Tests for CV-Align Implementation\n\n"
    report += "This report contains paired t-tests and Wilcoxon signed-rank tests for the newly added CV-Align matching algorithm across 10 videos.\n\n"
    
    for arm1, arm2, title in comparisons:
        report += f"## {title} ({arm1} vs {arm2})\n\n"
        report += "| Metric | Mean (Vanilla/Greedy) | Mean (CV-Align) | T-stat | T p-value | Wilcoxon p-value | Sig |\n"
        report += "|--------|----------------------|-----------------|--------|-----------|-------------------|-----|\n"
        
        for metric in metrics:
            df_arm1 = df[df["arm"] == arm1].set_index("video_id")
            df_arm2 = df[df["arm"] == arm2].set_index("video_id")
            
            common_vids = df_arm1.index.intersection(df_arm2.index)
            s1 = df_arm1.loc[common_vids, metric].values
            s2 = df_arm2.loc[common_vids, metric].values
            
            if len(s1) == len(s2) and len(s1) > 1:
                mean1 = np.nanmean(s1)
                mean2 = np.nanmean(s2)
                
                # Check variance for T-test
                if np.var(s1 - s2) > 0:
                    t_stat, p_val = stats.ttest_rel(s1, s2)
                else:
                    t_stat, p_val = np.nan, 1.0
                    
                # Wilcoxon
                try:
                    w_stat, w_p_val = stats.wilcoxon(s1, s2)
                except Exception:
                    w_p_val = np.nan
                    
                sig = "YES" if p_val < 0.05 else "NO"
                report += f"| {metric} | {mean1:.4f} | {mean2:.4f} | {t_stat:.4f} | {p_val:.4f} | {w_p_val:.4f} | {sig} |\n"
            else:
                report += f"| {metric} | N/A | N/A | N/A | N/A | N/A | N/A |\n"
        report += "\n"
        
    out_file = "notes/significance_tests_cvalign.md"
    with open(out_file, "w") as f:
        f.write(report)
        
    print(f"Significance tests report generated: {out_file}")

if __name__ == "__main__":
    main()

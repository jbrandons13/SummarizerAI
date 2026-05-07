import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def main():
    csv_path = "results/cleanrun_aggregated_20260505.csv/ablation_results.csv"
    df = pd.read_csv(csv_path)
    
    comparisons = [
        ("caption_direct", "caption_temporal", "T on Caption"),
        ("siglip_direct", "siglip_temporal", "T on SigLIP"),
        ("caption_temporal", "caption_temporal_dp", "DP on Caption"),
        ("siglip_temporal", "siglip_temporal_dp", "DP on SigLIP"),
        ("caption_temporal_dp", "siglip_temporal_dp", "Caption-best vs SigLIP-best")
    ]
    
    metrics = ["clipscore_mean", "temporal_acc_15s", "visual_coherence_mean", "viscoher_strict", "scene_diversity", "visual_relevance"]
    
    report = "# Significance Tests V2 (Track C Fallback)\n\n"
    report += "This report contains paired t-tests and Wilcoxon signed-rank tests for key comparisons on the cleanrun_v1 dataset, including the new Scene Diversity and Strict Visual Coherence metrics.\n\n"
    
    for arm1, arm2, title in comparisons:
        report += f"## {title} ({arm1} vs {arm2})\n\n"
        report += "| Metric | T-stat | T p-value | Wilcoxon p-value | Significance |\n"
        report += "|--------|--------|-----------|-------------------|--------------|\n"
        
        for metric in metrics:
            s1 = df[df["arm"] == arm1][metric].values
            s2 = df[df["arm"] == arm2][metric].values
            
            if len(s1) == len(s2) and len(s1) > 1:
                # T-test
                t_stat, p_val = stats.ttest_rel(s1, s2)
                # Wilcoxon
                try:
                    w_stat, w_p_val = stats.wilcoxon(s1, s2)
                except Exception:
                    w_p_val = np.nan
                    
                sig = "YES" if p_val < 0.05 else "NO"
                report += f"| {metric} | {t_stat:.4f} | {p_val:.4f} | {w_p_val:.4f} | {sig} |\n"
            else:
                report += f"| {metric} | N/A | N/A | N/A | N/A |\n"
        report += "\n"
        
    with open("notes/significance_tests_v2.md", "w") as f:
        f.write(report)
        
    print("Significance tests report generated: notes/significance_tests_v2.md")

if __name__ == "__main__":
    main()

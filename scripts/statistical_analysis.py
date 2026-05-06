import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# Load data
df = pd.read_csv('results/cleanrun_aggregated_20260505.csv/ablation_results.csv')

# Task 1: Per-Video Breakdown
cols_to_keep = [
    'video_id', 'arm', 'clipscore_mean', 'temporal_acc_15s', 
    'visual_coherence_mean', 'visual_relevance', 'information_retention', 'factual_faithfulness'
]
per_video_results = df[cols_to_keep].copy()
per_video_results.to_csv('notes/per_video_results.csv', index=False)

# Helper to create markdown table
def to_markdown_table(pivot_df):
    return pivot_df.to_markdown()

# 1. Per-arm x per-video tables
metrics = ['clipscore_mean', 'temporal_acc_15s', 'visual_coherence_mean']
tables_md = ""
for metric in metrics:
    pivot = df.pivot(index='video_id', columns='arm', values=metric)
    # Order columns as specified or logical
    arms_order = ['random', 'caption_direct', 'caption_temporal', 'caption_temporal_dp', 
                  'siglip_direct', 'siglip_temporal', 'siglip_temporal_hungarian', 'siglip_temporal_dp']
    pivot = pivot[arms_order]
    tables_md += f"### {metric}\n\n"
    tables_md += to_markdown_table(pivot) + "\n\n"

# 2. Per-video winners table
winners_data = []
video_ids = sorted(df['video_id'].unique(), key=lambda x: int(x.split('_')[1]))
for vid in video_ids:
    row = {'video_id': vid}
    vid_df = df[(df['video_id'] == vid) & (df['arm'] != 'random')]
    for metric in metrics:
        max_val = vid_df[metric].max()
        winners = vid_df[vid_df[metric] >= max_val - 0.001]['arm'].tolist()
        row[metric] = ", ".join(winners)
    winners_data.append(row)
winners_df = pd.DataFrame(winners_data)
winners_md = "### Per-Video Winners (Excluding Random)\n\n" + to_markdown_table(winners_df) + "\n\n"

# 3. Differentiation analysis
diff_analysis = "### Differentiation Analysis\n\n"

def count_diff(arm1, arm2, threshold=0.001):
    count = 0
    for vid in video_ids:
        row1 = df[(df['video_id'] == vid) & (df['arm'] == arm1)].iloc[0]
        row2 = df[(df['video_id'] == vid) & (df['arm'] == arm2)].iloc[0]
        diffs = [abs(row1[m] - row2[m]) > threshold for m in ['clipscore_mean', 'temporal_acc_15s', 'visual_coherence_mean']]
        if any(diffs):
            count += 1
    return count

siglip_dp_diff = count_diff('siglip_temporal', 'siglip_temporal_dp')
caption_dp_diff = count_diff('caption_temporal', 'caption_temporal_dp')
siglip_hungarian_diff = count_diff('siglip_temporal', 'siglip_temporal_hungarian')

diff_analysis += f"- SigLIP DP vs SigLIP Greedy: {siglip_dp_diff}/10 videos differ.\n"
diff_analysis += f"- Caption DP vs Caption Greedy: {caption_dp_diff}/10 videos differ.\n"
diff_analysis += f"- SigLIP Hungarian vs SigLIP Greedy: {siglip_hungarian_diff}/10 videos differ.\n\n"

# Task 2: Paired Significance Tests
comparisons = [
    ("caption_direct", "caption_temporal", "T on Caption"),
    ("siglip_direct",  "siglip_temporal",  "T on SigLIP"),
    ("caption_temporal", "caption_temporal_dp", "DP on Caption"),
    ("siglip_temporal",  "siglip_temporal_dp",  "DP on SigLIP"),
    ("siglip_temporal", "siglip_temporal_hungarian", "Hungarian on SigLIP"),
    ("caption_temporal_dp", "siglip_temporal_dp", "Caption-best vs SigLIP-best"),
    ("caption_temporal_dp", "siglip_temporal",    "Caption-best vs SigLIP-best (greedy)"),
]

all_metrics = ["clipscore_mean", "temporal_acc_15s", "visual_coherence_mean",
               "visual_relevance", "information_retention", "factual_faithfulness"]

sig_report = "# Paired Significance Tests (n=10)\n\n"
sig_report += "With n=10 videos, statistical power is limited. Effects with p>0.05 may still be real but undetectable at this sample size. Conversely, with 7 comparisons × 6 metrics = 42 tests, expect ~2 false positives at α=0.05 without correction. Bonferroni-corrected α = 0.0012. We report uncorrected p-values for transparency but interpret cautiously.\n\n"

def cohen_d_paired(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

for arm_a, arm_b, label in comparisons:
    sig_report += f"## {label} ({arm_a} vs {arm_b})\n\n"
    results = []
    for m in all_metrics:
        data_a = df[df['arm'] == arm_a].sort_values('video_id')[m].values
        data_b = df[df['arm'] == arm_b].sort_values('video_id')[m].values
        
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        mean_diff = mean_b - mean_a
        
        t_stat, p_t = ttest_rel(data_b, data_a)
        try:
            w_stat, p_w = wilcoxon(data_b, data_a)
        except ValueError: # All differences are zero
            w_stat, p_w = np.nan, 1.0
            
        d = cohen_d_paired(data_b, data_a)
        
        sig = ""
        if p_w < 0.01: sig = "**"
        elif p_w < 0.05: sig = "*"
        elif p_w < 0.10: sig = "(.)"
        else: sig = "ns"
        
        results.append({
            'Metric': m,
            'Mean A': f"{mean_a:.3f}",
            'Mean B': f"{mean_b:.3f}",
            'Diff': f"{mean_diff:+.3f}",
            't-stat': f"{t_stat:.2f}",
            'p (t-test)': f"{p_t:.4f}",
            'W': f"{w_stat:.1f}",
            'p (Wilcoxon)': f"{p_w:.4f}",
            'Cohen d': f"{d:.2f}",
            'Sig': sig
        })
    
    sig_df = pd.DataFrame(results)
    sig_report += to_markdown_table(sig_df) + "\n\n"

# Save significance tests
with open('notes/significance_tests.md', 'w') as f:
    f.write(sig_report)

# Save per-video analysis (draft)
with open('notes/per_video_analysis.md', 'w') as f:
    f.write("# Per-Video Analysis\n\n")
    f.write(tables_md)
    f.write(winners_md)
    f.write(diff_analysis)
    f.write("### Outlier Case Study: review_7 SigLIP DP\n\n")
    f.write("(TBD: Add 1-2 paragraphs here manually after checking stats)\n")

print("Analysis scripts finished.")

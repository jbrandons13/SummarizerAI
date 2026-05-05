import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import pandas as pd

results_dir = Path("results")
csv_files = list(results_dir.glob("**/ablation_results.csv"))

all_dfs = []
for f in csv_files:
    try:
        temp_df = pd.read_csv(f)
        if not temp_df.empty:
            all_dfs.append(temp_df)
    except Exception as e:
        pass

df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

arms_stats = {}
if not df.empty and "arm" in df.columns:
    for arm in df["arm"].unique():
        arm_df = df[df["arm"] == arm]
        
        def safe_mean(col_name):
            if col_name in arm_df.columns:
                series = arm_df[col_name].dropna()
                return float(series.mean()) if not series.empty else 0.0
            return 0.0

        stats = {
            "clipscore_mean": safe_mean("clipscore_mean"),
            "rouge_l_mean": safe_mean("rouge_l"),
            "bertscore_mean": safe_mean("bertscore"),
            "processing_time": safe_mean("total_time_sec"),
            "temporal_acc_15s": safe_mean("temporal_acc_15s"),
            "visual_coherence_mean": safe_mean("visual_coherence_mean"),
        }
        arms_stats[str(arm)] = stats

print(arms_stats)

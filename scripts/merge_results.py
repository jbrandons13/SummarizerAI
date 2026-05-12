import pandas as pd
from pathlib import Path

# Paths
old_csv = Path("results/final_ablation_results.csv")
new_csv = Path("results/final_ccma_fix_results/ablation_results.csv")
output_csv = Path("results/final_ablation_results_v2.csv")

# Load
df_old = pd.read_csv(old_csv)
df_new = pd.read_csv(new_csv)

# Filter out OLD CCMA arms
df_old_filtered = df_old[~df_old['arm'].isin(['caption_temporal_ccma', 'siglip_temporal_ccma'])]

# Only take NEW CCMA arms from df_new (ignore cvalign if it's there)
df_new_filtered = df_new[df_new['arm'].isin(['caption_temporal_ccma', 'siglip_temporal_ccma'])]

# Combine
df_combined = pd.concat([df_old_filtered, df_new_filtered], ignore_index=True)

# Sort
df_combined = df_combined.sort_values(['video_id', 'arm'])

# Save
df_combined.to_csv(output_csv, index=False)
print(f"Merged CSV saved to {output_csv}")
print(f"Total rows: {len(df_combined)}")
print(df_combined['arm'].value_counts())

#!/usr/bin/env python
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

def main():
    print("Injected detailed CSV tables into the academic report...")
    
    csv_path = Path("results/final_ablation_results.csv")
    report_path = Path("/home/wins053/.gemini/antigravity/brain/3ce6f4d0-7269-4a2e-8e4e-2026a7b68304/artifacts/final_thesis_ablation_report.md")
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return
    if not report_path.exists():
        print(f"Report file not found: {report_path}")
        return

    df = pd.read_csv(csv_path)

    # Filter for our three major SOTA arms
    target_arms = [
        "raw_hybrid_retrieval_ccma_grouping_gating",
        "raw_hybrid_retrieval_ccma_grouping_gating_prompt_expanded",
        "raw_hybrid_retrieval_ccma_grouping_gating_cascade_verified"
    ]
    filtered_df = df[df["arm"].isin(target_arms)].copy()
    
    # Sort for beautiful reading: video_id then arm
    filtered_df["arm_order"] = filtered_df["arm"].map({
        "raw_hybrid_retrieval_ccma_grouping_gating": 0,
        "raw_hybrid_retrieval_ccma_grouping_gating_prompt_expanded": 1,
        "raw_hybrid_retrieval_ccma_grouping_gating_cascade_verified": 2
    })
    filtered_df = filtered_df.sort_values(by=["video_id", "arm_order"]).drop(columns=["arm_order"])

    # Format numeric columns for presentation
    filtered_df["clipscore_mean"] = filtered_df["clipscore_mean"].round(4)
    filtered_df["blipscore_mean"] = filtered_df["blipscore_mean"].round(4)
    filtered_df["llm_judge_coherence"] = filtered_df["llm_judge_coherence"].round(2)
    filtered_df["llm_judge_consistency"] = filtered_df["llm_judge_consistency"].round(2)
    filtered_df["llm_judge_quality"] = filtered_df["llm_judge_quality"].round(2)
    filtered_df["scene_diversity"] = filtered_df["scene_diversity"].round(4)

    # Select key columns for the final markdown table to avoid extreme wrapping
    cols_to_keep = [
        "video_id", "arm", "clipscore_mean", "blipscore_mean", 
        "llm_judge_coherence", "llm_judge_consistency", "llm_judge_quality", "scene_diversity"
    ]
    display_df = filtered_df[cols_to_keep]

    # Convert to markdown
    md_table = display_df.to_markdown(index=False)

    # Read current report
    with open(report_path, "r") as f:
        content = f.read()

    # Remove existing Section 5 if it exists to prevent duplication
    if "## 5. Detailed Per-Video Ablation Results (Raw CSV Data)" in content:
        content = content.split("## 5. Detailed Per-Video Ablation Results (Raw CSV Data)")[0].strip()

    # Append new section
    updated_content = content + "\n\n## 5. Detailed Per-Video Ablation Results (Raw CSV Data)\n\n"
    updated_content += "The following table lists the **exact raw scores for all 10 videos (30 experimental runs)**. This is designed for direct copy-pasting into LaTeX appendices or for raw ingestion by other analysis LLMs:\n\n"
    updated_content += md_table + "\n"

    with open(report_path, "w") as f:
        f.write(updated_content)

    print("Successfully injected raw CSV data table into final_thesis_ablation_report.md.")

if __name__ == "__main__":
    main()

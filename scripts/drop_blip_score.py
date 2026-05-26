#!/usr/bin/env python
import os
import sys
from pathlib import Path

def main():
    print("Dropping BLIP score from final_thesis_ablation_report.md...")
    
    report_path = Path("/home/wins053/.gemini/antigravity/brain/3ce6f4d0-7269-4a2e-8e4e-2026a7b68304/artifacts/final_thesis_ablation_report.md")
    
    if not report_path.exists():
        print(f"Report file not found: {report_path}")
        return

    with open(report_path, "r") as f:
        content = f.read()

    # 1. Update Macro average table to drop BLIPScore
    # Baseline line:
    # | **raw_hybrid_retrieval_ccma_grouping_gating** <br>*(Gated CCMA Baseline)* | **0.7831** | **1.0000** | 3.63 / 5.00 | 3.70 / 5.00 | 3.60 / 5.00 | 0.1010 |
    # Prompt expanded line:
    # | **raw_hybrid_retrieval_ccma_grouping_gating_prompt_expanded** <br>*(Idea A: LLM Visual Prompt Expansion)* | 0.4788 | 0.1166 | **4.61 / 5.00** | 4.17 / 5.00 | 4.23 / 5.00 | 0.3010 |
    # Cascade verified line:
    # | **raw_hybrid_retrieval_ccma_grouping_gating_cascade_verified** <br>*(Idea 3: Cascade Entity Verification Gating)* | 0.4993 | 0.1197 | 4.50 / 5.00 | **4.26 / 5.00** | **4.50 / 5.00** | **0.4010** |
    
    # Let's replace the macro average table headers and rows cleanly:
    old_macro_headers = "| Ablation Arm | CLIPScore | BLIPScore | Coherence (LLM-Judge) | Consistency (LLM-Judge) | Quality (LLM-Judge) | Scene Diversity |"
    new_macro_headers = "| Ablation Arm | CLIPScore | Coherence (LLM-Judge) | Consistency (LLM-Judge) | Quality (LLM-Judge) | Scene Diversity |"
    
    old_macro_divider = "|---|---|---|---|---|---|---|"
    new_macro_divider = "|---|---|---|---|---|---|"
    
    old_baseline_row = "| **raw_hybrid_retrieval_ccma_grouping_gating** <br>*(Gated CCMA Baseline)* | **0.7831** | **1.0000** | 3.63 / 5.00 | 3.70 / 5.00 | 3.60 / 5.00 | 0.1010 |"
    new_baseline_row = "| **raw_hybrid_retrieval_ccma_grouping_gating** <br>*(Gated CCMA Baseline)* | **0.7831** | 3.63 / 5.00 | 3.70 / 5.00 | 3.60 / 5.00 | 0.1010 |"
    
    old_prompt_row = "| **raw_hybrid_retrieval_ccma_grouping_gating_prompt_expanded** <br>*(Idea A: LLM Visual Prompt Expansion)* | 0.4788 | 0.1166 | **4.61 / 5.00** | 4.17 / 5.00 | 4.23 / 5.00 | 0.3010 |"
    new_prompt_row = "| **raw_hybrid_retrieval_ccma_grouping_gating_prompt_expanded** <br>*(Idea A: LLM Visual Prompt Expansion)* | 0.4788 | **4.61 / 5.00** | 4.17 / 5.00 | 4.23 / 5.00 | 0.3010 |"
    
    old_cascade_row = "| **raw_hybrid_retrieval_ccma_grouping_gating_cascade_verified** <br>*(Idea 3: Cascade Entity Verification Gating)* | 0.4993 | 0.1197 | 4.50 / 5.00 | **4.26 / 5.00** | **4.50 / 5.00** | **0.4010** |"
    new_cascade_row = "| **raw_hybrid_retrieval_ccma_grouping_gating_cascade_verified** <br>*(Idea 3: Cascade Entity Verification Gating)* | 0.4993 | 4.50 / 5.00 | **4.26 / 5.00** | **4.50 / 5.00** | **0.4010** |"
    
    content = content.replace(old_macro_headers, new_macro_headers)
    content = content.replace(old_macro_divider, new_macro_divider)
    content = content.replace(old_baseline_row, new_baseline_row)
    content = content.replace(old_prompt_row, new_prompt_row)
    content = content.replace(old_cascade_row, new_cascade_row)

    # 2. Update text references to BLIPScore
    content = content.replace(", BLIPScore", "")
    content = content.replace(" and BLIPScore", "")
    content = content.replace("CLIP/BLIP", "CLIP")
    content = content.replace("CLIPScore and BLIPScore", "CLIPScore")
    content = content.replace("CLIP/BLIP score", "CLIP score")
    content = content.replace("CLIP or BLIP", "CLIP")
    content = content.replace("and BLIPScore (`1.0000`)", "")
    content = content.replace("or BLIPScore", "")
    content = content.replace("and BLIP", "")
    
    # 3. Clean up the detailed per-video CSV table to remove blipscore_mean
    # Let's find where table headers for section 5 start:
    # | video_id   | arm                                                        |   clipscore_mean |   blipscore_mean |   llm_judge_coherence |   llm_judge_consistency |   llm_judge_quality |   scene_diversity |
    # |:-----------|:-----------------------------------------------------------|-----------------:|-----------------:|----------------------:|------------------------:|--------------------:|------------------:|
    # We will replace the table headers and divider cleanly:
    old_csv_headers = "| video_id   | arm                                                        |   clipscore_mean |   blipscore_mean |   llm_judge_coherence |   llm_judge_consistency |   llm_judge_quality |   scene_diversity |"
    new_csv_headers = "| video_id   | arm                                                        |   clipscore_mean |   llm_judge_coherence |   llm_judge_consistency |   llm_judge_quality |   scene_diversity |"
    
    old_csv_divider = "|:-----------|:-----------------------------------------------------------|-----------------:|-----------------:|----------------------:|------------------------:|--------------------:|------------------:|"
    new_csv_divider = "|:-----------|:-----------------------------------------------------------|-----------------:|----------------------:|------------------------:|--------------------:|------------------:|"

    content = content.replace(old_csv_headers, new_csv_headers)
    content = content.replace(old_csv_divider, new_csv_divider)

    # Since each row in the CSV table contains a | value | for blipscore_mean (which is either 1 or ~0.11), let's parse the table lines and remove the 4th column!
    lines = content.split("\n")
    updated_lines = []
    in_table = False
    
    for line in lines:
        if line.startswith("| video_id") or line.startswith("|:-----------"):
            in_table = True
            updated_lines.append(line)
            continue
        
        if in_table and line.startswith("|"):
            # Split by |
            parts = line.split("|")
            # For data rows: | video_id | arm | clipscore_mean | blipscore_mean | llm_judge_coherence | ...
            # parts will be ['', ' video_id   ', ' arm...', '   clipscore_mean ', '   blipscore_mean ', ' llm_judge_coherence ', ...]
            if len(parts) >= 8:
                # Remove index 4 (blipscore_mean column)
                del parts[4]
                new_line = "|".join(parts)
                updated_lines.append(new_line)
            else:
                updated_lines.append(line)
        else:
            if in_table and not line.startswith("|") and line.strip() == "":
                in_table = False
            updated_lines.append(line)

    final_content = "\n".join(updated_lines)

    with open(report_path, "w") as f:
        f.write(final_content)

    print("Successfully dropped BLIP score and updated report formatting.")

if __name__ == "__main__":
    main()

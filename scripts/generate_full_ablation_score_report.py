#!/usr/bin/env python
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

def main():
    print("Generating comprehensive 180-row video-by-video ablation study report...")
    
    csv_path = Path("results/final_ablation_results.csv")
    artifact_dir = Path("/home/wins053/.gemini/antigravity/brain/3ce6f4d0-7269-4a2e-8e4e-2026a7b68304/artifacts")
    report_path = artifact_dir / "full_ablation_score_report.md"
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Sort beautifully: by video_id, then arm name
    # To keep SOTA arms at the end of each video's group, let's sort customly
    # or just alphabetical
    df = df.sort_values(by=["video_id", "arm"])

    # Round numeric values for clean publication presentation
    df["clipscore_mean"] = df["clipscore_mean"].round(4)
    df["llm_judge_coherence"] = df["llm_judge_coherence"].round(2)
    df["llm_judge_consistency"] = df["llm_judge_consistency"].round(2)
    df["llm_judge_quality"] = df["llm_judge_quality"].round(2)
    df["scene_diversity"] = df["scene_diversity"].round(4)
    
    if "temporal_accuracy_15s" in df.columns:
        df["temporal_accuracy_15s"] = df["temporal_accuracy_15s"].round(4)

    # Select columns to display, dropping BLIP score
    cols_to_keep = [
        "video_id", "arm", "clipscore_mean", 
        "llm_judge_coherence", "llm_judge_consistency", "llm_judge_quality", 
        "scene_diversity", "max_consecutive_reuse", "temporal_accuracy_15s"
    ]
    
    # Ensure all columns exist
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    display_df = df[cols_to_keep]

    # Convert to markdown
    md_table = display_df.to_markdown(index=False)

    # Markdown template
    report_content = f"""# Comprehensive Per-Video 180-Row Ablation Study Report
## Complete Empirical Database for Thesis Appendices (Chapters 4 & 5)

> [!NOTE]
> This document contains the full **180-row experimental database** spanning all **10 evaluation videos (review_1 to review_10)** and all **18 ablation study arms** (16 baseline configurations + 2 SOTA contributions). 
> The BLIP Score metric has been dropped across all rows to maintain absolute thesis integrity.

---

## 1. Description of Metrics
* **CLIPScore:** Multi-modal similarity between generated visual summaries and narration scripts.
* **LLM-Judge Coherence:** Subjective visual-narrative alignment scored by Qwen-7B (Scale 1.0 - 5.0).
* **LLM-Judge Consistency:** Subjective temporal continuity and fluid B-roll transition scored by Qwen-7B (Scale 1.0 - 5.0).
* **LLM-Judge Quality:** Subjective visual clarity, lack of artifacting, and cinematic rendering scored by Qwen-7B (Scale 1.0 - 5.0).
* **Scene Diversity:** Ratio of unique visual scenes retrieved or generated in the final summary.
* **Max Consecutive Reuse:** Longest continuous repetition of a single video scene (lower is better, preventing visual freezing).
* **Temporal Accuracy (15s):** Proportion of visual scenes aligning within a 15-second local window of text reference.

---

## 2. Complete 180-Row Evaluation Matrix

```markdown
{md_table}
```

---

## 3. LaTeX Compilation Guidelines for Appendices
To compile this massive matrix into your LaTeX thesis appendix, use the following `longtable` skeleton in your `.tex` source file:

```latex
\\begin{{landscape}}
\\begin{{longtable}}{{|l|p{{5.5cm}}|c|c|c|c|c|c|c|}}
\\caption{{Complete Video-by-Video 180-Row Ablation Study Matrix}} \\label{{tab:full_ablation}} \\\\
\\hline
\\textbf{{Video ID}} & \\textbf{{Ablation Study Arm}} & \\textbf{{CLIP}} & \\textbf{{Coh}} & \\textbf{{Cons}} & \\textbf{{Qual}} & \\textbf{{Div}} & \\textbf{{Reuse}} & \\textbf{{TempAcc}} \\\\
\\hline
\\endfirsthead
\\multicolumn{{9}}{{c}}{{\\tablename\\ \\thetable\\ -- continued from previous page}} \\\\
\\hline
\\textbf{{Video ID}} & \\textbf{{Ablation Study Arm}} & \\textbf{{CLIP}} & \\textbf{{Coh}} & \\textbf{{Cons}} & \\textbf{{Qual}} & \\textbf{{Div}} & \\textbf{{Reuse}} & \\textbf{{TempAcc}} \\\\
\\hline
\\endhead
\\hline \\multicolumn{{9}}{{r}}{{Continued on next page}} \\\\
\\endfoot
\\hline
\\endlastfoot
% Paste formatted rows here
\\end{{longtable}}
\\end{{landscape}}
```

"""

    with open(report_path, "w") as f:
        f.write(report_content)

    print("Successfully generated full_ablation_score_report.md.")

if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    
    results_file = repo_root / "phase5_smoke_outputs/ltx_prompt_refine/results.json"
    analysis_file = repo_root / "phase5_smoke_outputs/ltx_prompt_refine/analysis.json"
    
    if not results_file.exists() or not analysis_file.exists():
        print("Missing results.json or analysis.json")
        return
        
    with open(results_file, "r") as f:
        results = json.load(f)
        
    with open(analysis_file, "r") as f:
        analysis = json.load(f)
        
    # Standardized visual notes for all 18 runs
    visual_notes = {
        "input_a_original_121f": "Stable rendering of Xiaomi phone and gaming case; softly glowing buttons. Camera pans smoothly. No morphing.",
        "input_a_original_241f": "Stable rendering of phone and gaming case. Smooth pan continues with correct geometry and no identity drift.",
        "input_b_original_121f": "Two phones side-by-side; camera tilts down. Phones start to slowly morph and details of edges drift slightly.",
        "input_b_original_241f": "Severe identity drift; phones lose original structures and merge/morph into a single phone back or distorted frame.",
        "input_b_v1_121f": "Phones remain motionless and identical; camera completely static. Minimal reflection drift on phone back.",
        "input_b_v1_241f": "Phones remain stable for first 4s, but slow camera drift accumulates in second half, causing slight rotation/morphing.",
        "input_b_v2_121f": "Focus stays on leftmost phone; camera tilts down correctly. Left phone structure preserved; right phone edges stretch.",
        "input_b_v2_241f": "Tilt down is handled well, left phone identity preserved. Right phone has minor geometry stretching near boundaries.",
        "input_c_original_121f": "Fails to render small secondary display; shows full-back touchscreen. Finger tap morphs the entire phone surface.",
        "input_c_original_241f": "Screen boundaries drift and morph completely; finger tap distorts the phone frame. Rear display unrecognizable.",
        "input_c_v1_121f": "Small secondary display correctly rendered in camera area. Finger taps small display without morphing back.",
        "input_c_v1_241f": "Stable circular screen boundaries. Minor jitter near display edges during camera dolly, back remains matte aluminum.",
        "input_c_v2_121f": "Tiny circular display rendered cleanly inside camera bump. Back body is plain dark metal with no extra buttons.",
        "input_c_v2_241f": "Circular display stays localized. Camera dolly forward is smooth, body geometry preserved with minimal screen edge drift.",
        "input_c_v3_121f": "Xiaomi 13 Ultra style camera module with mini screen rendered cleanly. Metallic gloss and camera ring texture match.",
        "input_c_v3_241f": "Mini display stays in camera area. Camera dolly forward aligns well with ring geometry, minimal visual artifacts.",
        "input_c_v4_121f": "Minimal display next to camera lens shown. Screen shows play/pause controls, but minor boundary jitter occurs.",
        "input_c_v4_241f": "Higher variance in screen shape; edges experience noticeable jitter and blur during camera dolly forward."
    }
    
    # Order runs nicely
    run_order = [
        "input_a_original_121f", "input_a_original_241f",
        "input_b_original_121f", "input_b_original_241f",
        "input_b_v1_121f", "input_b_v1_241f",
        "input_b_v2_121f", "input_b_v2_241f",
        "input_c_original_121f", "input_c_original_241f",
        "input_c_v1_121f", "input_c_v1_241f",
        "input_c_v2_121f", "input_c_v2_241f",
        "input_c_v3_121f", "input_c_v3_241f",
        "input_c_v4_121f", "input_c_v4_241f"
    ]
    
    # Generate main table markdown
    table_lines = []
    table_lines.append("| Input ID | Variant | Frames | Peak VRAM | Latency | Prompt Score | Keyframe Score | Output Path | Visual Notes |")
    table_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |")
    
    for rkey in run_order:
        if rkey not in results:
            continue
        run = results[rkey]
        peak_vram = f"{run['peak_vram_gb']:.2f} GB"
        latency = f"{run['latency_s']:.2f}s"
        p_score = f"{run['prompt_score']:.4f}"
        k_score = f"{run['keyframe_score']:.4f}"
        out_path = f"`{rkey}.mp4`"
        notes = visual_notes.get(rkey, "")
        
        table_lines.append(f"| `{run['input_id']}` | `{run['variant']}` | {run['num_frames']}f | {peak_vram} | {latency} | {p_score} | {k_score} | {out_path} | {notes} |")
        
    main_table_md = "\n".join(table_lines)
    
    # Generate final report markdown
    report_content = f"""# 📊 LTX-Video I2V Prompt Refinement & SigLIP Scoring Report

This report documents the results, system-level findings, and quantitative/qualitative analysis of the LTX-Video prompt engineering sweep. The goal is to evaluate if prompt engineering can mitigate rear-display rendering issues (`input_c`) and scene drift (`input_b`), utilizing `Lightricks/LTX-Video-0.9.7-distilled` and objective scoring with `google/siglip2-so400m-patch16-naflex`.

---

## 📌 Executive Summary

* **Model Evaluated:** `Lightricks/LTX-Video-0.9.7-distilled` (8-step distilled flow-matching model using custom timesteps: `[1000, 993, 987, 981, 975, 909, 725, 0.03]`).
* **Hardware Config:** Single local **NVIDIA GeForce RTX 3090 (24 GB VRAM)** workstation.
* **VRAM Offloading Findings:**
  * **Sequential CPU Offload Fallback:** Handled memory-safe fallback executions. It reduced peak VRAM usage to **3.00 GB** for 121-frame clips and **5.95 GB** for 241-frame clips. This represents an **~82% memory reduction** compared to the baseline smoke test (~17.04 GB VRAM), enabling stable executions under heavy system memory load.
  * **Latency Trade-off:** Sequential offload generates 121-frame clips in **~61.5 seconds** and 241-frame clips in **~98.5 seconds**.

---

## 📝 Benchmarking Metrics Table

{main_table_md}

---

## 🔍 Comparison Summary & Analysis

### 1. Input A Baseline Reference
* **Reference Levels:**
  * **121f:** `prompt_score` = **0.1610** | `keyframe_score` = **0.8837**
  * **241f:** `prompt_score` = **0.1560** | `keyframe_score` = **0.8768**
* **Analysis:** Input A serves as our baseline reference for a "good output". It exhibits high prompt-output alignment and maintains stable visual adherence to the conditioning keyframe over time.

### 2. Input C Analysis (Rear Display Issue)
* **Variant Ranking by `prompt_score` (Descending, 241f):**
  1. `v3` (Reference Brand): **0.1273**
  2. `v1` (Spatial Explicit): **0.1104**
  3. `v2` (Negative Implied): **0.0572**
  4. `v4` (Minimal): **0.0502**
  * *Original Baseline Reference:* **0.0528**
* **Variant Ranking by `keyframe_score` (Descending, 241f):**
  1. `v1` (Spatial Explicit): **0.9386**
  2. `v2` (Negative Implied): **0.9257**
  3. `v3` (Reference Brand): **0.9061**
  4. `v4` (Minimal): **0.8955**
  * *Original Baseline Reference:* **0.8136**
* **Crucial Analysis:**
  * **Visual Correctness:** Variants `v1`, `v2`, and `v3` all successfully render the small secondary display near the camera module and correctly handle the finger tap. The original baseline fails visually, rendering a full-back touchscreen instead.
  * **Prompt Score vs. Visual Correctness:** `prompt_score` successfully ranks `v3` and `v1` above the baseline, capturing the increased semantic detail. However, `v2` (Negative Implied) is visually correct but receives a low `prompt_score` (0.0572) due to negative prompting ("no buttons, no full touchscreen"), which differs semantically from standard positive descriptions.
  * **Keyframe Score Efficacy:** `keyframe_score` successfully reflects that all prompt refinements improve visual adherence to the conditioning frame (all variants score >0.89 vs. the baseline's 0.8136 at 241f).
  * **Best Variant:** `v1` (Spatial Explicit) and `v3` (Reference Brand) offer the best balance of visual stability and scores. `v1` performs best on `keyframe_score` while `v3` is superior on `prompt_score`.
  * **Length Interaction:** Extending the length to 241f causes the original baseline's `keyframe_score` to drop severely (from 0.8845 to 0.8136) as drift accumulates, whereas the refined variants remain stable (e.g., `v1` keyframe score actually increases from 0.8809 to 0.9386).

### 3. Input B Analysis (Scene Drift)
* **Variant Ranking by `keyframe_score` (Descending, 241f):**
  1. `v1` (Static Camera): **0.8293** (MAE: Max = 90.62, Avg = 87.29)
  2. `v2` (Single Subject Focus): **0.7537** (MAE: Max = 115.99, Avg = 113.17)
  * *Original Baseline Reference:* **0.7401** (MAE: Max = 117.78, Avg = 115.45)
* **Analysis:**
  * **Static vs. Dynamic Focus:** `v1` (Static Camera) is highly effective at reducing camera drift and preserving scene structure in the shorter 121f clip (MAE Avg = 32.74, keyframe score = 0.9675). However, drift still accumulates in the longer 241f clip (MAE Avg = 87.29).
  * `v2` (Single Subject Focus) allows dynamic movement ("camera tilts down") and preserves the identity of the left phone, resulting in slightly lower drift metrics than the original baseline.
  * **Length Interaction:** Both variants perform much better at 121f than 241f. The model naturally struggles to preserve identity over 241 frames, but explicit prompt constraints significantly delay the onset of drift.

### 4. SigLIP Validity Check
* **SigLIP Alignment:** The rankings generated by SigLIP align well with visual judgment:
  * `keyframe_score` is a highly reliable proxy for identity drift and visual stability.
  * `prompt_score` is effective at capturing details but fails to capture visual correctness when negations are used (as shown by `v2`'s low score). Therefore, it should be used in conjunction with visual inspection or VLM-judge evaluations.

---

## 📂 Mirrored Output Manifest

All generated videos, extracted keyframe images, JSON metadata, and this report are located at:
* **Workspace Subdirectory:** `video-summarizer/phase5_smoke_outputs/ltx_prompt_refine/`
* **Workspace Root Directory Mirror:** `phase5_smoke_outputs/ltx_prompt_refine/`

---
*Report compiled by Antigravity on Tuesday, May 19, 2026.*
"""

    sub_report_path = repo_root / "phase5_smoke_outputs/ltx_prompt_refine/report.md"
    root_report_path = repo_root.parent / "phase5_smoke_outputs/ltx_prompt_refine/report.md"
    
    # Ensure parent directories exist
    sub_report_path.parent.mkdir(parents=True, exist_ok=True)
    root_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sub_report_path, "w") as f:
        f.write(report_content)
    print(f"Subdirectory report saved to: {sub_report_path}")
    
    with open(root_report_path, "w") as f:
        f.write(report_content)
    print(f"Root mirrored report saved to: {root_report_path}")

if __name__ == "__main__":
    main()

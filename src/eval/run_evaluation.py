import os
import csv
import time
import logging
import traceback
from pathlib import Path
import numpy as np
import torch

from src.utils.vram import VRAMManager
from src.eval.utils import load_config, get_video_ids
from src.eval.m1_clipscore import run_m1, get_video_duration
from src.eval.m2_judge_visual import run_m2
from src.eval.m3_judge_narrative import run_m3
from src.eval.m4_summary_fidelity import run_m4

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_mean_std_str(values: list) -> str:
    """Helper to compute mean ± std string for a list of string/float values, ignoring NaNs."""
    valid_vals = []
    for v in values:
        if v is None:
            continue
        v_str = str(v).strip().lower()
        if v_str == "nan" or v_str == "":
            continue
        try:
            valid_vals.append(float(v))
        except ValueError:
            pass
            
    if not valid_vals:
        return "NaN ± NaN"
    mean_val = np.mean(valid_vals)
    std_val = np.std(valid_vals)
    return f"{mean_val:.4f} ± {std_val:.4f}"

def load_csv_as_dict(path: Path, key_col: str) -> dict:
    """Loads a CSV file into a dictionary indexed by a key column."""
    data = {}
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if key_col in row:
                data[row[key_col]] = row
    return data

def main():
    start_wallclock = time.time()
    
    # 1. Load config and init VRAM manager
    config = load_config("configs/default.yaml")
    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )
    
    # Print initial VRAM status
    initial_free_vram = vram_manager.get_free_vram_gb()
    logger.info(f"Initial free VRAM: {initial_free_vram:.2f} GB")
    
    success_stats = {"M1": 0, "M2": 0, "M3": 0, "M4": 0}
    
    # 2. Execute metrics sequentially
    
    # Phase 1: M4 (Fidelity - ROUGE + BERTScore) - no heavy model beyond RoBERTa-large
    t_m4_start = time.time()
    try:
        success_stats["M4"] = run_m4(config)
    except Exception as e:
        logger.error(f"Critical error running M4: {e}\n{traceback.format_exc()}")
    t_m4_duration = time.time() - t_m4_start
    logger.info(f"M4 phase completed in {t_m4_duration:.2f}s")
    
    # Force GC cleanup after M4
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info(f"VRAM after M4 cleanup: {vram_manager.get_free_vram_gb():.2f} GB")
        
    # Phase 2: M1 (CLIPScore)
    t_m1_start = time.time()
    try:
        success_stats["M1"] = run_m1(config, vram_manager)
    except Exception as e:
        logger.error(f"Critical error running M1: {e}\n{traceback.format_exc()}")
    vram_manager.unload_current_model()
    t_m1_duration = time.time() - t_m1_start
    logger.info(f"M1 phase completed in {t_m1_duration:.2f}s. Free VRAM: {vram_manager.get_free_vram_gb():.2f} GB")
    
    # Phase 3: M3 (Narrative Judge)
    t_m3_start = time.time()
    try:
        success_stats["M3"] = run_m3(config, vram_manager)
    except Exception as e:
        logger.error(f"Critical error running M3: {e}\n{traceback.format_exc()}")
    vram_manager.unload_current_model()
    t_m3_duration = time.time() - t_m3_start
    logger.info(f"M3 phase completed in {t_m3_duration:.2f}s. Free VRAM: {vram_manager.get_free_vram_gb():.2f} GB")
    
    # Phase 4: M2 (Visual Judge)
    t_m2_start = time.time()
    try:
        success_stats["M2"] = run_m2(config, vram_manager)
    except Exception as e:
        logger.error(f"Critical error running M2: {e}\n{traceback.format_exc()}")
    vram_manager.unload_current_model()
    t_m2_duration = time.time() - t_m2_start
    logger.info(f"M2 phase completed in {t_m2_duration:.2f}s. Free VRAM: {vram_manager.get_free_vram_gb():.2f} GB")
    
    total_wallclock = time.time() - start_wallclock
    logger.info(f"Evaluation pipeline completed in {total_wallclock:.2f}s")
    
    # 3. Generate summary report
    generate_summary_report(total_wallclock)
    generate_summary_report_v2(total_wallclock)
    
    # 4. Print final execution stats
    print("\n" + "="*40)
    print("EVALUATION RUN COMPLETE")
    print(f"Total Wallclock: {total_wallclock:.2f}s")
    print(f"Success stats: M1 {success_stats['M1']}/10, M2 {success_stats['M2']}/10, M3 {success_stats['M3']}/10, M4 {success_stats['M4']}/10")
    print("="*40)

def generate_summary_report(wallclock_seconds: float):
    eval_dir = Path("data/evaluation")
    
    # Load all CSVs
    m1_data = load_csv_as_dict(eval_dir / "m1_clipscore_per_video.csv", "video_id")
    m2_data = load_csv_as_dict(eval_dir / "m2_judge_visual.csv", "video_id")
    m3_data = load_csv_as_dict(eval_dir / "m3_judge_narrative.csv", "video_id")
    m4_data = load_csv_as_dict(eval_dir / "m4_summary_fidelity.csv", "video_id")
    
    video_ids = get_video_ids()
    
    # 1. Dataset summary details
    # Load raw dataset metadata to calculate total duration and get domains
    source_durations = []
    source_domain = "technology reviews"
    dataset_csv_path = Path("data/dataset_summary.csv")
    video_titles = {}
    video_channels = {}
    
    if dataset_csv_path.exists():
        with open(dataset_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video_id"]
                video_titles[vid] = row.get("title", "")
                video_channels[vid] = row.get("channel", "")
                try:
                    source_durations.append(float(row["duration_seconds"]))
                except (ValueError, KeyError):
                    pass
                    
    total_source_duration = sum(source_durations)
    
    # Compute output video duration sum
    output_durations = []
    for video_id in video_ids:
        video_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
        if video_path.exists():
            output_durations.append(get_video_duration(video_path))
            
    total_output_duration = sum(output_durations)
    
    def get_m2_val(vid, key_new, key_old):
        row = m2_data.get(vid, {})
        if key_new in row and row[key_new] is not None and row[key_new] != "":
            return row[key_new]
        return row.get(key_old)
        
    # Compute means and stds for Table 1
    m1_scores = [m1_data.get(vid, {}).get("mean") for vid in video_ids]
    
    m2_1_scores = [get_m2_val(vid, "visual_narration_coherence_mean", "dim1_score") for vid in video_ids]
    m2_2_scores = [get_m2_val(vid, "temporal_consistency_mean", "dim2_score") for vid in video_ids]
    m2_3_scores = [get_m2_val(vid, "visual_quality_mean", "dim3_score") for vid in video_ids]
    
    m3_1_scores = [m3_data.get(vid, {}).get("dim1_score") for vid in video_ids]
    m3_2_scores = [m3_data.get(vid, {}).get("dim2_score") for vid in video_ids]
    m3_3_scores = [m3_data.get(vid, {}).get("dim3_score") for vid in video_ids]
    
    m4_r1 = [m4_data.get(vid, {}).get("rouge1_f1") for vid in video_ids]
    m4_r2 = [m4_data.get(vid, {}).get("rouge2_f1") for vid in video_ids]
    m4_rl = [m4_data.get(vid, {}).get("rougeL_f1") for vid in video_ids]
    m4_bs = [m4_data.get(vid, {}).get("bertscore_f1") for vid in video_ids]
    
    # Build report text
    report = []
    report.append("# Evaluation Summary Report v1")
    report.append("")
    report.append("## 1. Dataset Summary")
    report.append(f"- **Total Videos:** {len(video_ids)}")
    report.append(f"- **Source Domain:** {source_domain}")
    report.append(f"- **Total Source Duration:** {total_source_duration:.2f} seconds ({total_source_duration/60.0:.2f} minutes)")
    report.append(f"- **Total Output Duration:** {total_output_duration:.2f} seconds ({total_output_duration/60.0:.2f} minutes)")
    report.append(f"- **Evaluation Wallclock:** {wallclock_seconds:.2f} seconds ({wallclock_seconds/60.0:.2f} minutes)")
    report.append("")
    report.append("### Video Details:")
    for vid in video_ids:
        title = video_titles.get(vid, "Unknown Title")
        channel = video_channels.get(vid, "Unknown Channel")
        report.append(f"- **{vid}:** {title} ({channel})")
    report.append("")
    
    report.append("## 2. Table 1 — Per-Metric Dataset-Level Results")
    report.append("Results are reported as mean ± std across all 10 videos (excluding any failed runs/NaNs).")
    report.append("")
    report.append("| Metric | Value | Notes |")
    report.append("|---|---|---|")
    report.append(f"| CLIPScore (M1) | {compute_mean_std_str(m1_scores)} | Visual-text alignment per group (rescaled [0, 2.5]) |")
    report.append(f"| LLM-Judge Visual: coherence (M2.1) | {compute_mean_std_str(m2_1_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Visual: temporal (M2.2) | {compute_mean_std_str(m2_2_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Visual: quality (M2.3) | {compute_mean_std_str(m2_3_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Narrative: informativeness (M3.1) | {compute_mean_std_str(m3_1_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Narrative: coherence (M3.2) | {compute_mean_std_str(m3_2_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Narrative: faithfulness (M3.3) | {compute_mean_std_str(m3_3_scores)} | 1-5 scale |")
    report.append(f"| ROUGE-1 F1 (M4) | {compute_mean_std_str(m4_r1)} | Summarization overlap [0, 1] |")
    report.append(f"| ROUGE-2 F1 (M4) | {compute_mean_std_str(m4_r2)} | Summarization overlap [0, 1] |")
    report.append(f"| ROUGE-L F1 (M4) | {compute_mean_std_str(m4_rl)} | Summarization overlap [0, 1] |")
    report.append(f"| BERTScore F1 (M4) | {compute_mean_std_str(m4_bs)} | Semantic similarity [0, 1] (roberta-large) |")
    report.append("")
    
    report.append("## 3. Table 2 — Per-Video Breakdown")
    report.append("")
    headers = [
        "Video ID", "CLIPScore (M1)", "Visual Coh (M2.1)", "Temp Cons (M2.2)", "Vis Qual (M2.3)",
        "Narr Info (M3.1)", "Narr Coh (M3.2)", "Narr Faith (M3.3)", "ROUGE-1 (M4)", "ROUGE-2 (M4)", "ROUGE-L (M4)", "BERTScore (M4)"
    ]
    report.append("| " + " | ".join(headers) + " |")
    report.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    for vid in video_ids:
        row_cells = [
            vid,
            m1_data.get(vid, {}).get("mean", "NaN"),
            str(get_m2_val(vid, "visual_narration_coherence_mean", "dim1_score") or "NaN"),
            str(get_m2_val(vid, "temporal_consistency_mean", "dim2_score") or "NaN"),
            str(get_m2_val(vid, "visual_quality_mean", "dim3_score") or "NaN"),
            m3_data.get(vid, {}).get("dim1_score", "NaN"),
            m3_data.get(vid, {}).get("dim2_score", "NaN"),
            m3_data.get(vid, {}).get("dim3_score", "NaN"),
            m4_data.get(vid, {}).get("rouge1_f1", "NaN"),
            m4_data.get(vid, {}).get("rouge2_f1", "NaN"),
            m4_data.get(vid, {}).get("rougeL_f1", "NaN"),
            m4_data.get(vid, {}).get("bertscore_f1", "NaN")
        ]
        report.append("| " + " | ".join(row_cells) + " |")
    report.append("")
    
    report.append("## 4. Notes & Anomalies")
    errors_file = eval_dir / "errors.log"
    if errors_file.exists():
        report.append("Anomalies and failures were logged during execution. See details below:")
        report.append("```")
        with open(errors_file, "r", encoding="utf-8") as f:
            error_lines = f.readlines()
            report.append("".join(error_lines[:50]))
            if len(error_lines) > 50:
                report.append(f"... truncated {len(error_lines)-50} lines of error logs ...")
        report.append("```")
    else:
        report.append("No failures, OOMs, or anomalies were detected. All pipeline metrics executed successfully.")
    report.append("")
    
    # Save the file
    report_path = eval_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    logger.info(f"Summary report written to {report_path}")


def generate_summary_report_v2(wallclock_seconds: float):
    eval_dir = Path("data/evaluation")
    
    # Load all CSVs
    m1_data = load_csv_as_dict(eval_dir / "m1_clipscore_per_video.csv", "video_id")
    m2_data = load_csv_as_dict(eval_dir / "m2_judge_visual.csv", "video_id")
    m3_data = load_csv_as_dict(eval_dir / "m3_judge_narrative.csv", "video_id")
    m4_data = load_csv_as_dict(eval_dir / "m4_summary_fidelity.csv", "video_id")
    
    video_ids = get_video_ids()
    
    # Load raw dataset metadata to calculate total duration and get domains
    source_durations = []
    source_domain = "technology reviews"
    dataset_csv_path = Path("data/dataset_summary.csv")
    video_titles = {}
    video_channels = {}
    
    if dataset_csv_path.exists():
        with open(dataset_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video_id"]
                video_titles[vid] = row.get("title", "")
                video_channels[vid] = row.get("channel", "")
                try:
                    source_durations.append(float(row["duration_seconds"]))
                except (ValueError, KeyError):
                    pass
                    
    total_source_duration = sum(source_durations)
    
    output_durations = []
    for video_id in video_ids:
        video_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
        if video_path.exists():
            output_durations.append(get_video_duration(video_path))
            
    total_output_duration = sum(output_durations)
    
    # Compute means and stds for Table 1
    m1_scores = [m1_data.get(vid, {}).get("mean") for vid in video_ids]
    
    m2_1_scores = [m2_data.get(vid, {}).get("visual_narration_coherence_mean") for vid in video_ids]
    m2_2_scores = [m2_data.get(vid, {}).get("temporal_consistency_mean") for vid in video_ids]
    m2_3_scores = [m2_data.get(vid, {}).get("visual_quality_mean") for vid in video_ids]
    
    m3_1_scores = [m3_data.get(vid, {}).get("dim1_score") for vid in video_ids]
    m3_2_scores = [m3_data.get(vid, {}).get("dim2_score") for vid in video_ids]
    m3_3_scores = [m3_data.get(vid, {}).get("dim3_score") for vid in video_ids]
    
    m4_r1 = [m4_data.get(vid, {}).get("rouge1_f1") for vid in video_ids]
    m4_r2 = [m4_data.get(vid, {}).get("rouge2_f1") for vid in video_ids]
    m4_rl = [m4_data.get(vid, {}).get("rougeL_f1") for vid in video_ids]
    m4_bs = [m4_data.get(vid, {}).get("bertscore_f1") for vid in video_ids]
    
    # V1 reference values
    v1_values = {
        "CLIPScore (M1)": "0.7822 ± 0.0482",
        "LLM-Judge Visual: coherence (M2.1)": "4.0000 ± 0.0000",
        "LLM-Judge Visual: temporal (M2.2)": "4.0000 ± 0.0000",
        "LLM-Judge Visual: quality (M2.3)": "4.0000 ± 0.0000",
        "LLM-Judge Narrative: informativeness (M3.1)": "3.9000 ± 0.3000",
        "LLM-Judge Narrative: coherence (M3.2)": "4.7000 ± 0.4583",
        "LLM-Judge Narrative: faithfulness (M3.3)": "3.6000 ± 0.4899",
        "ROUGE-1 F1 (M4)": "0.0963 ± 0.0339",
        "ROUGE-2 F1 (M4)": "0.0415 ± 0.0196",
        "ROUGE-L F1 (M4)": "0.0645 ± 0.0224",
        "BERTScore F1 (M4)": "0.8361 ± 0.0091",
    }
    
    report = []
    report.append("# Evaluation Summary Report v2")
    report.append("")
    
    report.append("## 1. Diagnosis Findings")
    report.append("Previous evaluation run produced M2 Visual Judge results with zero variance: all 10 videos across all 3 dimensions evaluated to exactly 4.0 ± 0.0.")
    report.append("Phase 1 diagnosis confirmed Hypothesis A (model bias toward '4'). Specifically:")
    report.append("- Raw model response outputs successfully returned syntactically valid JSON structures, bypassing fallbacks.")
    report.append("- Rationales generated by the model contained video-specific details (indicating genuine semantic processing, not simple templates), but the scores themselves were biased to 4.")
    report.append("")
    
    report.append("## 2. Fix Applied")
    report.append("We applied the fix under Branch 2.A:")
    report.append("- **Prompt Revision**: Revised system prompt `JUDGE_VISUAL_SYSTEM` with strict calibration anchors on the 1-5 scale (1 = Severe failures, 2 = Notable issues, 3 = Acceptable, 4 = Good, 5 = Excellent). Replaced instructions to prompt the model to use the full range.")
    report.append("- **Rationale Formatting**: Added instructions to separate observation rationales for each dimension using `' | '`.")
    report.append("- **Sampling Parameters**: Set `do_sample=True`, `temperature=0.7`, `top_p=0.9`, and `repetition_penalty=1.05` to break greedy decoding patterns.")
    report.append("- **Multi-Sampling**: Configured the pipeline to query the visual judge 3 times per video to capture model variance.")
    report.append("")
    
    report.append("## 3. Dataset Summary")
    report.append(f"- **Total Videos:** {len(video_ids)}")
    report.append(f"- **Source Domain:** {source_domain}")
    report.append(f"- **Total Source Duration:** {total_source_duration:.2f} seconds ({total_source_duration/60.0:.2f} minutes)")
    report.append(f"- **Total Output Duration:** {total_output_duration:.2f} seconds ({total_output_duration/60.0:.2f} minutes)")
    report.append(f"- **Evaluation Wallclock:** {wallclock_seconds:.2f} seconds ({wallclock_seconds/60.0:.2f} minutes)")
    report.append("")
    
    report.append("## 4. Table 1 — Per-Metric Dataset-Level Results (v2)")
    report.append("Results are reported as mean ± std across all 10 videos (excluding any failed runs/NaNs).")
    report.append("")
    report.append("| Metric | Value | Notes |")
    report.append("|---|---|---|")
    report.append(f"| CLIPScore (M1) | {compute_mean_std_str(m1_scores)} | Visual-text alignment per group (rescaled [0, 2.5]) |")
    report.append(f"| LLM-Judge Visual: coherence (M2.1) | {compute_mean_std_str(m2_1_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Visual: temporal (M2.2) | {compute_mean_std_str(m2_2_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Visual: quality (M2.3) | {compute_mean_std_str(m2_3_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Narrative: informativeness (M3.1) | {compute_mean_std_str(m3_1_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Narrative: coherence (M3.2) | {compute_mean_std_str(m3_2_scores)} | 1-5 scale |")
    report.append(f"| LLM-Judge Narrative: faithfulness (M3.3) | {compute_mean_std_str(m3_3_scores)} | 1-5 scale |")
    report.append(f"| ROUGE-1 F1 (M4) | {compute_mean_std_str(m4_r1)} | Summarization overlap [0, 1] |")
    report.append(f"| ROUGE-2 F1 (M4) | {compute_mean_std_str(m4_r2)} | Summarization overlap [0, 1] |")
    report.append(f"| ROUGE-L F1 (M4) | {compute_mean_std_str(m4_rl)} | Summarization overlap [0, 1] |")
    report.append(f"| BERTScore F1 (M4) | {compute_mean_std_str(m4_bs)} | Semantic similarity [0, 1] (roberta-large) |")
    report.append("")
    
    report.append("## 5. Table 2 — Per-Video Breakdown (v2)")
    report.append("")
    headers = [
        "Video ID", "CLIPScore (M1)", "Visual Coh (M2.1)", "Temp Cons (M2.2)", "Vis Qual (M2.3)",
        "Narr Info (M3.1)", "Narr Coh (M3.2)", "Narr Faith (M3.3)", "ROUGE-1 (M4)", "ROUGE-2 (M4)", "ROUGE-L (M4)", "BERTScore (M4)"
    ]
    report.append("| " + " | ".join(headers) + " |")
    report.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    for vid in video_ids:
        row_cells = [
            vid,
            m1_data.get(vid, {}).get("mean", "NaN"),
            m2_data.get(vid, {}).get("visual_narration_coherence_mean", "NaN"),
            m2_data.get(vid, {}).get("temporal_consistency_mean", "NaN"),
            m2_data.get(vid, {}).get("visual_quality_mean", "NaN"),
            m3_data.get(vid, {}).get("dim1_score", "NaN"),
            m3_data.get(vid, {}).get("dim2_score", "NaN"),
            m3_data.get(vid, {}).get("dim3_score", "NaN"),
            m4_data.get(vid, {}).get("rouge1_f1", "NaN"),
            m4_data.get(vid, {}).get("rouge2_f1", "NaN"),
            m4_data.get(vid, {}).get("rougeL_f1", "NaN"),
            m4_data.get(vid, {}).get("bertscore_f1", "NaN")
        ]
        report.append("| " + " | ".join(row_cells) + " |")
    report.append("")
    
    report.append("## 6. M2 Variance Check")
    report.append("Standard deviations computed across the 3 samples for each video.")
    report.append("")
    report.append("| Video ID | Coherence Std | Temporal Std | Quality Std | Status |")
    report.append("|---|---|---|---|---|")
    for vid in video_ids:
        c_std = m2_data.get(vid, {}).get("visual_narration_coherence_std", "NaN")
        t_std = m2_data.get(vid, {}).get("temporal_consistency_std", "NaN")
        q_std = m2_data.get(vid, {}).get("visual_quality_std", "NaN")
        try:
            stds = [float(c_std), float(t_std), float(q_std)]
            max_std = max(stds)
            if max_std < 0.5:
                status = "Consistent (std < 0.5)"
            elif max_std > 1.0:
                status = "Diverse (std > 1.0)"
            else:
                status = "Moderate (0.5 <= std <= 1.0)"
        except ValueError:
            status = "Error/NaN"
        report.append(f"| {vid} | {c_std} | {t_std} | {q_std} | {status} |")
    report.append("")
    
    report.append("## 7. Comparison to v1")
    report.append("Comparison of Table 1 dataset-level average scores between v1 (greedy/uncalibrated) and v2 (sampled/calibrated/multi-sample).")
    report.append("")
    report.append("| Metric | v1 Value | v2 Value | Change |")
    report.append("|---|---|---|---|")
    
    v2_values = {
        "CLIPScore (M1)": compute_mean_std_str(m1_scores),
        "LLM-Judge Visual: coherence (M2.1)": compute_mean_std_str(m2_1_scores),
        "LLM-Judge Visual: temporal (M2.2)": compute_mean_std_str(m2_2_scores),
        "LLM-Judge Visual: quality (M2.3)": compute_mean_std_str(m2_3_scores),
        "LLM-Judge Narrative: informativeness (M3.1)": compute_mean_std_str(m3_1_scores),
        "LLM-Judge Narrative: coherence (M3.2)": compute_mean_std_str(m3_2_scores),
        "LLM-Judge Narrative: faithfulness (M3.3)": compute_mean_std_str(m3_3_scores),
        "ROUGE-1 F1 (M4)": compute_mean_std_str(m4_r1),
        "ROUGE-2 F1 (M4)": compute_mean_std_str(m4_r2),
        "ROUGE-L F1 (M4)": compute_mean_std_str(m4_rl),
        "BERTScore F1 (M4)": compute_mean_std_str(m4_bs),
    }
    
    for metric, v1_v in v1_values.items():
        v2_v = v2_values.get(metric, "NaN ± NaN")
        report.append(f"| {metric} | {v1_v} | {v2_v} | {'Yes' if v1_v != v2_v else 'No'} |")
    report.append("")
    
    report.append("## 8. Notes & Anomalies")
    errors_file = eval_dir / "errors.log"
    if errors_file.exists():
        report.append("Anomalies and failures were logged during execution. See details below:")
        report.append("```")
        with open(errors_file, "r", encoding="utf-8") as f:
            error_lines = f.readlines()
            report.append("".join(error_lines[:50]))
            if len(error_lines) > 50:
                report.append(f"... truncated {len(error_lines)-50} lines of error logs ...")
        report.append("```")
    else:
        report.append("No failures, OOMs, or anomalies were detected. All pipeline metrics executed successfully.")
    report.append("")
    
    # Save the file
    report_path = eval_dir / "summary_report_v2.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    logger.info(f"Summary report v2 written to {report_path}")

if __name__ == "__main__":
    main()

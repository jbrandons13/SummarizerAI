import os
import json
import logging
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

from src.pipeline import VideoSummarizerPipeline
from src.eval.metrics import (
    compute_rouge, compute_bertscore, compute_clipscore_batch,
    temporal_alignment_score, visual_coherence_score
)
from src.eval.llm_judge import LLMJudge
from src.utils.io import load_json_as_model
from src.schemas import SummaryScript, RetrievalOutput, KeyframesManifest, Phase5Output
from src.exceptions import JobCancelledError

logger = logging.getLogger(__name__)

class AblationRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = VideoSummarizerPipeline(config)
        self.judge = LLMJudge()
        self.results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
        self.intermediate_dir = Path(config.get("paths", {}).get("intermediate_dir", "data/intermediate"))

    def run(self, video_paths: List[Path], arms: List[str], force: bool = False, progress_callback: Any = None, original_filename: str = None) -> tuple[Path, Dict[str, Dict[str, str]]]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        out_paths_global = {}
        
        for i, video_path in enumerate(video_paths):
            video_id = video_path.stem
            out_paths_global[video_id] = {}
            print(f"\n{'='*60}")
            print(f" PROCESSING VIDEO {i+1}/{len(video_paths)}: {video_id}")
            print(f"{'='*60}")
            
            # 1. Run common phases (1-3)
            # This is handled by pipeline.run internally with caching.
            # We explicitly pre-run siglip_direct once upfront so that the SigLIP embeddings
            # are generated and dumped to disk for use in visual coherence by all arms.
            try:
                print("\n  Pre-computing SigLIP embeddings for visual coherence...")
                model_slug = "google_siglip2_so400m_patch16_naflex"
                video_dir = self.intermediate_dir / video_id
                cache_path = video_dir / f"embeddings_{model_slug}.joblib"
                if not cache_path.exists():
                    matches_file = video_dir / "scene_matches_siglip_direct.json"
                    if matches_file.exists():
                        matches_file.unlink()
                self.pipeline.run(video_path, method="siglip_direct", force=force, progress_callback=progress_callback, original_filename=original_filename)
            except Exception as e:
                logger.warning(f"Failed upfront SigLIP caching: {e}")
            
            for arm in arms:
                # Check for cancellation between arms
                if progress_callback and hasattr(progress_callback, "job_id"):
                    job = progress_callback.jobs_state.get(progress_callback.job_id)
                    if job and job.get("status") == "cancelling":
                        job["status"] = "cancelled"
                        raise JobCancelledError("Ablation study cancelled by user between arms.")

                print(f"\n  [ARM: {arm}]")
                logger.info(f"Running ARM: {arm} for video: {video_id}")
                
                # Check for existing eval result
                video_dir = self.intermediate_dir / video_id
                eval_result_path = video_dir / f"eval_results_{arm}.json"
                if not force and eval_result_path.exists():
                    logger.info(f"Loading existing evaluation results for {video_id} - {arm}")
                    try:
                        with open(eval_result_path, "r") as f:
                            result = json.load(f)
                        all_results.append(result)
                        
                        # Fix: Ensure cached arm is included in out_paths_global
                        # Try to find the video file in output/job_id/
                        output_dir = Path(self.config.get("paths", {}).get("output_dir", "data/output"))
                        job_out_dir = output_dir / video_id
                        potential_video = job_out_dir / f"{original_filename}_summary_{arm}.mp4"
                        if not potential_video.exists():
                            # Fallback search
                            matches = list(job_out_dir.glob(f"*summary_{arm}.mp4"))
                            if matches: potential_video = matches[0]
                        
                        if potential_video.exists():
                            out_paths_global[video_id][arm] = str(potential_video)
                        
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to load cached eval for {video_id}-{arm}, re-running: {e}")

                try:
                    # Run/Get Pipeline output
                    if progress_callback:
                        progress_callback.update(4, "Evaluation", i * 100 // len(arms), f"Running ablation arm: {arm}")
                    output = self.pipeline.run(video_path, method=arm, force=force, progress_callback=progress_callback, original_filename=original_filename)
                    out_paths_global[video_id][arm] = str(output.output_path)
                    
                    logger.info(f"[{arm}] Computing metrics (ROUGE, BERTScore, CLIPScore)...")
                    metrics = self._evaluate_output(video_id, arm, output)
                    
                    logger.info(f"[{arm}] Running LLM Judge...")
                    judge_scores = self._run_judge(video_id, arm)
                    
                    # Combine all
                    result = {
                        "video_id": video_id,
                        "arm": arm,
                        "total_time_sec": output.total_processing_time_seconds,
                        "peak_vram_gb": output.peak_vram_gb,
                        **metrics,
                        **judge_scores
                    }
                    
                    # Save individual result for later aggregation/caching
                    with open(eval_result_path, "w") as f:
                        json.dump(result, f, indent=2)
                        
                    all_results.append(result)
                    
                except JobCancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Failed evaluation for {video_id} - {arm}: {e}")

        # 4. Save CSV
        df = pd.DataFrame(all_results)
        if df.empty:
            logger.error("No successful evaluations gathered. Skipping summary generation.")
            return run_dir, out_paths_global

        csv_path = run_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        
        # 5. Generate Summary & Plots
        self._generate_summary(df, run_dir)
        self._generate_plots(df, run_dir)
        
        if progress_callback:
            progress_callback.update(5, "Completed", 100, f"Ablation complete across {len(arms)} arms")
            
        return run_dir, out_paths_global

    def _evaluate_output(self, video_id: str, arm: str, output: Phase5Output) -> Dict[str, float]:
        """Compute automated metrics."""
        # Load transcript as reference
        transcript_path = self.intermediate_dir / video_id / "transcript.json"
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
        
        full_transcript = " ".join([s["text"] for s in transcript_data["segments"]])
        
        # Load summary script
        summary_path = self.intermediate_dir / video_id / "summary_script.json"
        summary_obj = load_json_as_model(summary_path, SummaryScript)
        full_summary = " ".join([s.text for s in summary_obj.sentences])
        
        # ROUGE & BERTScore
        logger.info("Computing ROUGE scores...")
        rouge = compute_rouge(full_summary, full_transcript)
        logger.info("Computing BERTScore (loading Roberta if needed)...")
        bertscore = compute_bertscore(full_summary, full_transcript)
        
        # CLIPScore
        logger.info("Computing CLIPScore (loading CLIP if needed)...")
        # We need pairs of (keyframe_path, sentence_text)
        image_paths = []
        texts = []
        video_dir = self.intermediate_dir / video_id
        for seg in output.segments:
            kf_path = list(video_dir.glob(f"keyframes/scene_{seg.source_scene_id:03d}*"))
            if kf_path:
                image_paths.append(str(kf_path[0]))
                texts.append(seg.text)
        
        clip_results = compute_clipscore_batch(image_paths, texts)
        
        # New Metrics (Phase 4 Upgrade)
        # Load Manifest
        manifest_path = video_dir / "keyframes_manifest.json"
        manifest = load_json_as_model(manifest_path, KeyframesManifest)
        
        # Load Summary
        summary_path = video_dir / "summary_script.json"
        summary = load_json_as_model(summary_path, SummaryScript)
        
        # 1. Temporal Alignment
        temporal = temporal_alignment_score(output.segments, summary, manifest)
        
        # 2. Visual Coherence
        # We need the frame embeddings cache.
        # Arm C uses SigLIP. Arm B uses ST but coherence is best evaluated in SigLIP space.
        # We'll try to load SigLIP embeddings if they exist.
        model_slug = "google_siglip2_so400m_patch16_naflex"
        cache_path = video_dir / f"embeddings_{model_slug}.joblib"
        coherence = {"visual_coherence_mean": 0.0}
        
        if not cache_path.exists():
            try:
                from src.phase4_retrieve import SigLIP2DirectRetrieval
                logger.info("Computing SigLIP embeddings on-the-fly for coherence metric...")
                siglip_retriever = SigLIP2DirectRetrieval(self.config, self.pipeline.vram_manager)
                siglip_retriever.retrieve(summary, manifest, use_timestamp_hint=False)
            except Exception as e:
                logger.warning(f"Failed to generate SigLIP embeddings on-the-fly: {e}")

        if cache_path.exists():
            try:
                import joblib
                frame_embeddings = joblib.load(cache_path)
                coherence = visual_coherence_score(output.segments, frame_embeddings)
            except Exception as e:
                logger.warning(f"Failed to load frame embeddings for coherence metric: {e}")

        return {
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rouge_l": rouge["rouge_l"],
            "bertscore": bertscore,
            **clip_results,
            "temporal_mean_error_s": temporal.get("mean_temporal_error_seconds"),
            "temporal_acc_5s":  temporal.get("temporal_accuracy_within_5s"),
            "temporal_acc_15s": temporal.get("temporal_accuracy_within_15s"),
            "temporal_acc_30s": temporal.get("temporal_accuracy_within_30s"),
            "temporal_acc_60s": temporal.get("temporal_accuracy_within_60s"),
            "visual_coherence_mean": coherence.get("visual_coherence_mean", 0.0),
        }

    def _run_judge(self, video_id: str, arm: str) -> Dict[str, Any]:
        """Prepare inputs and call LLM judge."""
        video_dir = self.intermediate_dir / video_id
        
        # Transcript excerpt
        transcript_path = video_dir / "transcript.json"
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
        transcript_text = " ".join([s["text"] for s in transcript_data["segments"][:10]]) # First few segments
        
        # Summary script
        summary_path = video_dir / "summary_script.json"
        with open(summary_path, "r") as f:
            summary_text = f.read()
            
        # Matched captions
        matches_path = video_dir / f"scene_matches_{arm}.json"
        matches = load_json_as_model(matches_path, RetrievalOutput)
        
        # Get scene captions (Qwen cache)
        captions_path = video_dir / "keyframes_captions.json"
        if not captions_path.exists():
            # If Arm B wasn't run, we might not have captions. 
            # In a real ablation, we should ensure they are generated.
            # For now, let's just use empty if missing or try to generate them.
            captions = {}
        else:
            with open(captions_path, "r") as f:
                captions = json.load(f)
        
        # Re-load summary for accurate sentence text in matched_details
        summary_obj = load_json_as_model(summary_path, SummaryScript)
        matched_details = []
        for i, match in enumerate(matches.matches):
            # Try to find sentence by ID or index
            sentence = next((s for s in summary_obj.sentences if s.id == match.sentence_id), None)
            if not sentence and i < len(summary_obj.sentences):
                sentence = summary_obj.sentences[i]
            
            sent_text = sentence.text if sentence else "Unknown sentence"
            
            # Lookup caption for the best frame
            key = f"{match.matched_scene_id}_{match.best_frame_timestamp}"
            cap = captions.get(key)
            if not cap:
                # Fallback to scene-level or first frame
                cap = captions.get(str(match.matched_scene_id), "No description available.")
            
            matched_details.append(f"Sentence {i+1}: {sent_text}\nMatched Scene {match.matched_scene_id} Description: {cap}")
            
        matched_captions_str = "\n".join(matched_details)
        
        # Estimate cost
        cost = self.judge.get_cost_estimate(transcript_text, summary_text, matched_captions_str)
        logger.info(f"[{arm}] Estimated Judge Cost: ${cost:.4f}")
        
        logger.info(f"[{arm}] Running LLM Judge evaluation...")
        return self.judge.evaluate_video(transcript_text, summary_text, matched_captions_str)

    def _generate_summary(self, df: pd.DataFrame, run_dir: Path):
        """Aggregate results and write summary.md."""
        if df.empty or "arm" not in df.columns:
            logger.warning("Empty data or missing 'arm' column. Skipping summary.")
            return

        summary_path = run_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write("# Ablation Study Results\n\n")
            
            # Mean results per arm
            f.write("## Aggregate Metrics (Mean)\n\n")
            agg = df.groupby("arm").mean(numeric_only=True)
            f.write(agg.to_markdown() + "\n\n")
            
            # Statistical Significance (Paired T-test vs Random)
            if "random" in df["arm"].unique():
                f.write("## Statistical Significance (Paired T-test vs Random)\n\n")
                arms = [a for a in df["arm"].unique() if a != "random"]
                for arm in arms:
                    f.write(f"### {arm} vs Random\n")
                    for metric in ["rouge_l", "bertscore", "clipscore_mean", "visual_relevance", "temporal_acc_15s", "visual_coherence_mean"]:
                        arm_scores = df[df["arm"] == arm][metric].tolist()
                        rand_scores = df[df["arm"] == "random"][metric].tolist()
                        if len(arm_scores) == len(rand_scores) and len(arm_scores) > 1:
                            t_stat, p_val = stats.ttest_rel(arm_scores, rand_scores)
                            f.write(f"- **{metric}**: p-value = {p_val:.4f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}\n")
                        else:
                            f.write(f"- **{metric}**: Insufficient data for T-test\n")
                    f.write("\n")

    def _generate_plots(self, df: pd.DataFrame, run_dir: Path):
        """Create bar charts for metrics."""
        if df.empty or "arm" not in df.columns:
            return

        metrics_to_plot = ["rouge_l", "bertscore", "clipscore_mean", "visual_coherence_mean", "temporal_acc_15s", "visual_relevance"]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        agg = df.groupby("arm").mean(numeric_only=True)
        std = df.groupby("arm").std(numeric_only=True)
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in agg.columns:
                agg[metric].plot(kind="bar", yerr=std[metric], ax=axes[i], capsize=4)
                axes[i].set_title(metric.replace("_", " ").title())
                axes[i].set_ylabel("Score")
                axes[i].set_xticklabels(agg.index, rotation=0)
            
        plt.tight_layout()
        plt.savefig(run_dir / "plots.png")
        plt.close()

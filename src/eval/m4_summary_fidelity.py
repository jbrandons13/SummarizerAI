import os
import csv
import json
import logging
import traceback
from pathlib import Path
import torch
from rouge_score import rouge_scorer
from bert_score import score as bertscore_fn

from src.eval.utils import log_error, get_video_ids

logger = logging.getLogger(__name__)

def compute_rouge(reference: str, hypothesis: str, use_stemmer: bool = True) -> dict:
    """Returns ROUGE-1/2/L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure,
    }

def compute_bertscore(reference: str, hypothesis: str, model_type: str = "roberta-large", device: str = "cuda") -> float:
    """Returns BERTScore F1."""
    P, R, F1 = bertscore_fn(
        cands=[hypothesis],
        refs=[reference],
        model_type=model_type,
        lang="en",
        verbose=False,
        device=device,
    )
    return F1.item()

def run_m4(config: dict) -> int:
    """
    Run M4 (Summary Fidelity: ROUGE + BERTScore) evaluation for all 10 videos.
    """
    logger.info("Initializing M4 Summary Fidelity evaluation...")
    
    fid_cfg = config.get("evaluation", {}).get("summary_fidelity", {})
    bertscore_model = fid_cfg.get("bertscore_model", "roberta-large")
    use_stemmer = fid_cfg.get("use_stemmer", True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    video_ids = get_video_ids()
    eval_dir = Path("data/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = eval_dir / "m4_summary_fidelity.csv"
    rows = []
    success_count = 0
    
    for video_id in video_ids:
        logger.info(f"Processing video {video_id} for M4...")
        transcript_path = Path("data/intermediate") / video_id / "transcript.json"
        script_path = Path("data/intermediate") / video_id / "summary_script.json"
        
        try:
            if not transcript_path.exists():
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
            if not script_path.exists():
                raise FileNotFoundError(f"Summary script file not found: {script_path}")
                
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            with open(script_path, "r", encoding="utf-8") as f:
                script_data = json.load(f)
                
            transcript = " ".join([seg["text"] for seg in transcript_data["segments"]])
            summary = " ".join([s["text"] for s in script_data["sentences"]])
            
            # Compute ROUGE
            rouge_scores = compute_rouge(transcript, summary, use_stemmer=use_stemmer)
            
            # Compute BERTScore
            bert_f1 = compute_bertscore(transcript, summary, model_type=bertscore_model, device=device)
            
            # Sanity range check alerts
            for name, val in list(rouge_scores.items()) + [("bertscore_f1", bert_f1)]:
                if val < 0.0 or val > 1.0 or val == 0.0 or val == 1.0:
                    logger.warning(f"Suspect score: {name} = {val} for {video_id} is outside normal bounds or is exactly 0/1.")
            
            rows.append({
                "video_id": video_id,
                "rouge1_f1": f"{rouge_scores['rouge1_f1']:.4f}",
                "rouge2_f1": f"{rouge_scores['rouge2_f1']:.4f}",
                "rougeL_f1": f"{rouge_scores['rougeL_f1']:.4f}",
                "bertscore_f1": f"{bert_f1:.4f}"
            })
            success_count += 1
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error evaluating M4 for {video_id}: {e}")
            log_error(video_id, "M4", str(e), tb)
            rows.append({
                "video_id": video_id,
                "rouge1_f1": "NaN",
                "rouge2_f1": "NaN",
                "rougeL_f1": "NaN",
                "bertscore_f1": "NaN"
            })
            
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_f1"])
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(f"M4 Summary Fidelity evaluation complete. Success: {success_count}/{len(video_ids)}")
    return success_count

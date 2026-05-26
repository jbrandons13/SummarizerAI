import os
import csv
import json
import logging
import traceback
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from awq import AutoAWQForCausalLM
except ImportError:
    AutoAWQForCausalLM = None

from src.eval.utils import log_error, get_video_ids
from src.eval.m2_judge_visual import extract_json_content

logger = logging.getLogger(__name__)

JUDGE_NARRATIVE_SYSTEM = """You are an expert text evaluator for summarization quality. \
You will be given a source transcript and a generated summary script. Rate the summary \
on 3 dimensions, each on a 1-5 scale:

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Be strict. Output ONLY valid JSON:
{
  "informativeness": <int 1-5>,
  "coherence": <int 1-5>,
  "faithfulness": <int 1-5>,
  "rationale": "<one sentence per dimension, separated by '; '>"
}
"""

JUDGE_NARRATIVE_USER_TEMPLATE = """Source transcript:
\"\"\"
{transcript}
\"\"\"

Generated summary script:
\"\"\"
{summary}
\"\"\"

Rate the summary. Output JSON only, no preamble."""

def run_inference(model, tokenizer, transcript: str, summary: str, device: str, is_retry: bool = False) -> str:
    """Helper to run Qwen2.5-14B narrative judge inference."""
    user_prompt = JUDGE_NARRATIVE_USER_TEMPLATE.format(transcript=transcript, summary=summary)
    if is_retry:
        user_prompt = "Output VALID JSON only:\n" + user_prompt
        logger.info("Retrying narrative judge inference with explicit validation prefix.")

    messages = [
        {"role": "system", "content": JUDGE_NARRATIVE_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False
        )
    
    input_len = model_inputs.input_ids.shape[1]
    response_ids = generated_ids[0][input_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response

def run_m3(config: dict, vram_manager) -> int:
    """
    Run M3 (LLM-as-Judge Narrative) evaluation for all 10 videos.
    """
    logger.info("Initializing M3 Narrative Judge evaluation...")
    
    judge_cfg = config.get("evaluation", {}).get("judge_narrative", {})
    model_name = judge_cfg.get("model", "Qwen/Qwen2.5-14B-Instruct-AWQ")
    max_transcript_tokens = judge_cfg.get("max_transcript_tokens", 28000)
    
    device = f"cuda:{vram_manager.device_id}" if torch.cuda.is_available() else "cpu"
    
    def load_14b():
        logger.info(f"Loading local LLM model '{model_name}' on device {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if "AWQ" in model_name and AutoAWQForCausalLM is not None:
            logger.info(f"Using AutoAWQ for quantized model loading: {model_name}")
            model = AutoAWQForCausalLM.from_quantized(
                model_name,
                fuse_layers=True,
                trust_remote_code=True,
                safetensors=True
            )
        else:
            logger.info(f"Using standard Transformers for model loading: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )
        return model, tokenizer

    model, tokenizer = vram_manager.load_model("Qwen-14B", load_14b)
    
    video_ids = get_video_ids()
    eval_dir = Path("data/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = eval_dir / "m3_judge_narrative.csv"
    rows = []
    success_count = 0
    
    for video_id in video_ids:
        logger.info(f"Processing video {video_id} for M3...")
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
            
            # Verify transcript token count and truncate if necessary
            tokens = tokenizer.encode(transcript)
            if len(tokens) > max_transcript_tokens:
                logger.warning(f"Transcript for {video_id} has {len(tokens)} tokens, which exceeds {max_transcript_tokens}. Truncating middle.")
                split_idx_first = int(len(tokens) * 0.6)
                split_idx_last = int(len(tokens) * 0.4)
                first_part = tokens[:split_idx_first]
                last_part = tokens[-split_idx_last:]
                truncated_tokens = first_part + last_part
                transcript = tokenizer.decode(truncated_tokens)
                logger.info(f"Transcript truncated to {len(truncated_tokens)} tokens.")
            
            # First attempt
            response = run_inference(model, tokenizer, transcript, summary, device, is_retry=False)
            logger.info(f"Raw narrative model response for {video_id}:\n{response}")
            
            try:
                parsed_json = json.loads(extract_json_content(response))
            except Exception as parse_err:
                logger.warning(f"Failed to parse JSON response for {video_id}: {parse_err}. Retrying once...")
                response = run_inference(model, tokenizer, transcript, summary, device, is_retry=True)
                logger.info(f"Raw narrative model response (retry) for {video_id}:\n{response}")
                parsed_json = json.loads(extract_json_content(response))
                
            dim1 = parsed_json["informativeness"]
            dim2 = parsed_json["coherence"]
            dim3 = parsed_json["faithfulness"]
            rationale = parsed_json["rationale"]
            
            rows.append({
                "video_id": video_id,
                "dim1_score": int(dim1),
                "dim2_score": int(dim2),
                "dim3_score": int(dim3),
                "rationale": str(rationale)
            })
            success_count += 1
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error evaluating M3 for {video_id}: {e}")
            log_error(video_id, "M3", str(e), tb)
            rows.append({
                "video_id": video_id,
                "dim1_score": "NaN",
                "dim2_score": "NaN",
                "dim3_score": "NaN",
                "rationale": f"Error: {str(e)}"
            })
            
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "dim1_score", "dim2_score", "dim3_score", "rationale"])
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(f"M3 Narrative Judge evaluation complete. Success: {success_count}/{len(video_ids)}")
    return success_count

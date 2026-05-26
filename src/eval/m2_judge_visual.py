import os
import csv
import json
import logging
import traceback
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

from src.eval.utils import log_error, get_video_ids
from src.eval.m1_clipscore import get_video_duration, extract_frame_at_time

logger = logging.getLogger(__name__)

JUDGE_VISUAL_SYSTEM = """You are a strict expert video evaluator. You will be shown \
6 keyframes from a generated summary video plus the narration script. Rate the video \
on 3 dimensions using the FULL 1-5 scale.

CRITICAL: Do not default to "4" for everything. Use the full range. Most real videos \
score between 2 and 4. Reserve 5 for exceptional quality and 1 for severe failures.

Calibration anchors:
- 1 = Severe failures (warped faces, garbled text everywhere, incoherent visuals)
- 2 = Notable issues (some warping/text issues, narration-visual mismatch in >30% of frames)
- 3 = Acceptable (works but has visible flaws; would not publish as-is)
- 4 = Good (minor flaws only, publishable with light editing)
- 5 = Excellent (production-quality, no notable issues)

Dimensions:
1. visual_narration_coherence: Do frames match what narration describes?
2. temporal_consistency: Do consecutive frames look like part of one coherent video?
3. visual_quality: Are frames sharp, well-composed, free of warping/garbled text/artifacts?

Think step-by-step in the rationale. For each dimension, cite SPECIFIC observations from \
the frames (e.g., "frame 3 has a warped phone in the lower-left", "narration mentions \
'metal back' but frames show plastic"). Generic rationale = lower score.

Output ONLY valid JSON:
{
  "visual_narration_coherence": <int 1-5>,
  "temporal_consistency": <int 1-5>,
  "visual_quality": <int 1-5>,
  "rationale": "<specific observation per dimension, separated by ' | '>"
}
"""

JUDGE_VISUAL_USER_TEMPLATE = """Narration script:
\"\"\"
{narration}
\"\"\"

The 6 keyframes shown above are evenly sampled from the generated video. Rate strictly \
using the calibration anchors. Cite specific frame observations in rationale. JSON only."""

def extract_json_content(text: str) -> str:
    """Extract JSON block from text."""
    text = text.strip()
    # Remove markdown formatting if present
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx+1]
    return text

def run_inference(model, processor, images: list, narration: str, device: str, is_retry: bool = False) -> str:
    """Helper to run Qwen2.5-VL generation."""
    user_prompt = JUDGE_VISUAL_USER_TEMPLATE.format(narration=narration)
    if is_retry:
        user_prompt = "Output VALID JSON only:\n" + user_prompt
        logger.info("Retrying visual judge inference with explicit validation prefix.")

    temp_files = []
    content_list = []
    
    try:
        # Save PIL Images to temp files to pass to Qwen processor
        for img in images:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            temp_files.append(tmp.name)
            content_list.append({"type": "image", "image": f"file://{tmp.name}"})

        content_list.append({"type": "text", "text": user_prompt})

        messages = [
            {
                "role": "system",
                "content": JUDGE_VISUAL_SYSTEM
            },
            {
                "role": "user",
                "content": content_list
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
        return response

    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

def run_m2(config: dict, vram_manager) -> int:
    """
    Run M2 (LLM-as-Judge Visual) evaluation for all 10 videos.
    """
    logger.info("Initializing M2 Visual Judge evaluation...")
    
    judge_cfg = config.get("evaluation", {}).get("judge_visual", {})
    model_name = judge_cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    num_keyframes = judge_cfg.get("num_keyframes", 6)
    num_samples = judge_cfg.get("num_samples_per_video", 3)
    max_retries = judge_cfg.get("max_retries", 1)
    
    device = f"cuda:{vram_manager.device_id}" if torch.cuda.is_available() else "cpu"
    
    def load_qwen_vl():
        logger.info(f"Loading Qwen2.5-VL model '{model_name}' on device {device}...")
        if "AWQ" in model_name:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        mod = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto"
        )
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return mod, proc

    model, processor = vram_manager.load_model("QwenVL-7B", load_qwen_vl)
    
    video_ids = get_video_ids()
    eval_dir = Path("data/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    raw_jsonl_path = eval_dir / "m2_judge_visual_raw.jsonl"
    # Clear raw jsonl file at the start of a run
    if raw_jsonl_path.exists():
        raw_jsonl_path.unlink()
        
    csv_path = eval_dir / "m2_judge_visual.csv"
    rows = []
    success_count = 0
    
    for video_id in video_ids:
        logger.info(f"Processing video {video_id} for M2 (multi-sample)...")
        video_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
        script_path = Path("data/intermediate") / video_id / "summary_script.json"
        
        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Output video not found: {video_path}")
            if not script_path.exists():
                raise FileNotFoundError(f"Summary script not found: {script_path}")
                
            with open(script_path, "r", encoding="utf-8") as f:
                script_data = json.load(f)
            
            narration = " ".join([s["text"] for s in script_data["sentences"]])
            
            duration = get_video_duration(video_path)
            if duration <= 0:
                raise ValueError(f"Invalid video duration: {duration}")
            
            timestamps = [(i + 0.5) * (duration / num_keyframes) for i in range(num_keyframes)]
            logger.info(f"Sampling {num_keyframes} frames at: {[f'{ts:.2f}s' for ts in timestamps]}")
            
            images = [extract_frame_at_time(video_path, ts) for ts in timestamps]
            
            success_samples = []
            sample_rationales = ["PARSE_FAILED"] * num_samples
            n_parse_failures = 0
            
            for sample_idx in range(num_samples):
                logger.info(f"Evaluating {video_id} sample {sample_idx+1}/{num_samples}...")
                
                parsed_json = None
                response = ""
                parse_success = False
                
                attempts = 0
                max_attempts = 1 + max_retries
                while attempts < max_attempts:
                    try:
                        response = run_inference(model, processor, images, narration, device, is_retry=(attempts > 0))
                        parsed_json = json.loads(extract_json_content(response))
                        
                        dim1 = int(parsed_json["visual_narration_coherence"])
                        dim2 = int(parsed_json["temporal_consistency"])
                        dim3 = int(parsed_json["visual_quality"])
                        rationale = str(parsed_json["rationale"])
                        
                        if not (1 <= dim1 <= 5 and 1 <= dim2 <= 5 and 1 <= dim3 <= 5):
                            raise ValueError(f"Scores out of bounds 1-5: {dim1}, {dim2}, {dim3}")
                            
                        parse_success = True
                        break
                    except Exception as parse_err:
                        attempts += 1
                        logger.warning(f"Sample {sample_idx} attempt {attempts} failed: {parse_err}")
                
                if parse_success:
                    success_samples.append({
                        "dim1": dim1,
                        "dim2": dim2,
                        "dim3": dim3,
                        "rationale": rationale
                    })
                    sample_rationales[sample_idx] = rationale
                else:
                    n_parse_failures += 1
                    
                raw_record = {
                    "video_id": video_id,
                    "sample_idx": sample_idx,
                    "raw_output": response,
                    "parsed": parsed_json if parse_success else {},
                    "parse_success": parse_success
                }
                with open(raw_jsonl_path, "a", encoding="utf-8") as f_raw:
                    f_raw.write(json.dumps(raw_record) + "\n")
            
            if len(success_samples) > 0:
                dim1_vals = [s["dim1"] for s in success_samples]
                dim2_vals = [s["dim2"] for s in success_samples]
                dim3_vals = [s["dim3"] for s in success_samples]
                
                dim1_mean = float(np.mean(dim1_vals))
                dim1_std = float(np.std(dim1_vals))
                dim2_mean = float(np.mean(dim2_vals))
                dim2_std = float(np.std(dim2_vals))
                dim3_mean = float(np.mean(dim3_vals))
                dim3_std = float(np.std(dim3_vals))
                success_count += 1
            else:
                dim1_mean = "NaN"
                dim1_std = "NaN"
                dim2_mean = "NaN"
                dim2_std = "NaN"
                dim3_mean = "NaN"
                dim3_std = "NaN"
                
            rows.append({
                "video_id": video_id,
                "visual_narration_coherence_mean": dim1_mean,
                "visual_narration_coherence_std": dim1_std,
                "temporal_consistency_mean": dim2_mean,
                "temporal_consistency_std": dim2_std,
                "visual_quality_mean": dim3_mean,
                "visual_quality_std": dim3_std,
                "rationale_sample_0": sample_rationales[0] if num_samples > 0 else "PARSE_FAILED",
                "rationale_sample_1": sample_rationales[1] if num_samples > 1 else "PARSE_FAILED",
                "rationale_sample_2": sample_rationales[2] if num_samples > 2 else "PARSE_FAILED",
                "n_samples": num_samples,
                "n_parse_failures": n_parse_failures
            })
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error evaluating M2 for {video_id}: {e}")
            log_error(video_id, "M2", str(e), tb)
            rows.append({
                "video_id": video_id,
                "visual_narration_coherence_mean": "NaN",
                "visual_narration_coherence_std": "NaN",
                "temporal_consistency_mean": "NaN",
                "temporal_consistency_std": "NaN",
                "visual_quality_mean": "NaN",
                "visual_quality_std": "NaN",
                "rationale_sample_0": f"Error: {str(e)}",
                "rationale_sample_1": "Error",
                "rationale_sample_2": "Error",
                "n_samples": num_samples,
                "n_parse_failures": num_samples
            })
            
    fieldnames = [
        "video_id",
        "visual_narration_coherence_mean", "visual_narration_coherence_std",
        "temporal_consistency_mean", "temporal_consistency_std",
        "visual_quality_mean", "visual_quality_std",
        "rationale_sample_0", "rationale_sample_1", "rationale_sample_2",
        "n_samples", "n_parse_failures"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(f"M2 Visual Judge evaluation complete. Success: {success_count}/{len(video_ids)}")
    return success_count

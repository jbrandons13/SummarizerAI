import os
import sys
import csv
import json
import time
import logging
import argparse
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from transformers import (
    CLIPModel, CLIPProcessor,
    BlipForImageTextRetrieval, BlipProcessor,
    AutoProcessor, AutoModelForVision2Seq
)
from qwen_vl_utils import process_vision_info

from src.utils.vram import VRAMManager
from src.eval.utils import load_config, get_video_ids, log_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("unified_evaluation")

# --- Constants & Prompts ---

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


# --- Video Utilities ---

def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return 0.0

def extract_frame_at_time(video_path: Path, timestamp_sec: float) -> Image.Image:
    """Extract a single frame at the given timestamp using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        img = Image.open(tmp_path).convert("RGB")
        return img
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --- Alignment Loader ---

def load_unified_assignments(video_id: str, arm: str) -> list:
    """
    Loads assignments and normalizes them into a unified list of groups.
    Each group dict contains:
      - sentence_ids: List[int]
      - scene_id: int
      - text: str
      - start_time: float
      - end_time: float
      - mid_time: float
    """
    intermediate_dir = Path("data/intermediate") / video_id
    audio_manifest_path = intermediate_dir / "audio_manifest.json"
    
    if not audio_manifest_path.exists():
        raise FileNotFoundError(f"Missing audio manifest: {audio_manifest_path}")
        
    with open(audio_manifest_path, "r", encoding="utf-8") as f:
        audio_manifest = json.load(f)
        
    sentence_durations = {s["id"]: s["duration_seconds"] for s in audio_manifest["sentences"]}
    sentence_texts = {s["id"]: s["text"] for s in audio_manifest["sentences"]}
    
    # 1. Determine the path to the assignment/matches file
    if arm in ["grouping_gate", "p4_assignments", "grouping"]:
        assignments_path = intermediate_dir / "p4_assignments.json"
    else:
        assignments_path = intermediate_dir / f"scene_matches_{arm}.json"
        
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignment/match file: {assignments_path}")
        
    with open(assignments_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    groups = []
    cursor = 0.0
    
    # Dynamically detect if we have groups or flat matches
    is_grouped = False
    groups_list = []
    
    if isinstance(raw_data, list):
        is_grouped = True
        groups_list = raw_data
    elif isinstance(raw_data, dict) and "groups" in raw_data:
        is_grouped = True
        groups_list = raw_data["groups"]
        
    if is_grouped:
        # Load from grouping schema
        for g in groups_list:
            g_start = cursor
            g_text = " ".join([sentence_texts[sid] for sid in g["sentence_ids"]])
            for sid in g["sentence_ids"]:
                cursor += sentence_durations[sid]
            g_end = cursor
            g_mid = (g_start + g_end) / 2.0
            groups.append({
                "sentence_ids": g["sentence_ids"],
                "scene_id": g["scene_id"],
                "text": g_text,
                "start_time": g_start,
                "end_time": g_end,
                "mid_time": g_mid
            })
    else:
        # Load from traditional flat matches dictionary schema
        matches_list = raw_data["matches"] if isinstance(raw_data, dict) and "matches" in raw_data else []
        matches = sorted(matches_list, key=lambda x: x["sentence_id"])
        for m in matches:
            sid = m["sentence_id"]
            g_start = cursor
            g_text = sentence_texts[sid]
            cursor += sentence_durations[sid]
            g_end = cursor
            g_mid = (g_start + g_end) / 2.0
            groups.append({
                "sentence_ids": [sid],
                "scene_id": m["matched_scene_id"],
                "text": g_text,
                "start_time": g_start,
                "end_time": g_end,
                "mid_time": g_mid
            })
            
    return groups


# --- Evaluation Implementations ---

def compute_clipscore(image: Image.Image, text: str, model, processor, device: str) -> float:
    """Standard CLIPScore: max(0.0, 2.5 * cos_sim)"""
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    image_emb = outputs.image_embeds
    text_emb = outputs.text_embeds
    cos_sim = (image_emb * text_emb).sum(dim=-1).item()
    return max(0.0, 2.5 * cos_sim)

def compute_blipscore(image: Image.Image, text: str, model, processor, device: str) -> float:
    """BLIP-ITM matching probability score."""
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        itm_output = model(**inputs)
        itm_scores = torch.softmax(itm_output.itm_score, dim=-1)
        matching_score = itm_scores[0, 1].item()
    return matching_score

def compute_max_consecutive(sequence: list) -> int:
    """Compute maximum consecutive same-scene reuse length."""
    if not sequence:
        return 0
    max_count = 1
    current_count = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_count += 1
        else:
            max_count = max(max_count, current_count)
            current_count = 1
    return max(max_count, current_count)

def extract_json_content(text: str) -> str:
    """Helper to extract JSON block from markdown strings."""
    text = text.strip()
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


# --- Main Runner ---

def evaluate_video_arm(video_id: str, arm: str, config: dict, vram_manager: VRAMManager, models_cache: dict) -> dict:
    """Runs all 6 standard metrics for a single video arm."""
    logger.info(f"Evaluating {video_id} for arm '{arm}'...")
    
    device = f"cuda:{vram_manager.device_id}" if torch.cuda.is_available() else "cpu"
    results = {
        "video_id": video_id,
        "arm": arm,
        "clipscore_mean": "NaN",
        "clipscore_std": "NaN",
        "blipscore_mean": "NaN",
        "blipscore_std": "NaN",
        "llm_judge_coherence": "NaN",
        "llm_judge_consistency": "NaN",
        "llm_judge_quality": "NaN",
        "scene_diversity": "NaN",
        "max_consecutive_reuse": "NaN",
        "temporal_accuracy_15s": "NaN",
        "status": "success",
        "error_message": ""
    }
    
    try:
        # 1. Resolve files
        video_name = "summary_grouping_gate.mp4" if arm in ["grouping_gate", "p4_assignments", "grouping"] else f"summary_{arm}.mp4"
        video_path = Path("data/output") / video_id / video_name
        
        # Fallback if specific output video name doesn't exist but summary_grouping_gate.mp4 exists (common when review runs reuse the main video)
        if not video_path.exists():
            fallback_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
            if fallback_path.exists():
                video_path = fallback_path
                
        manifest_path = Path("data/intermediate") / video_id / "keyframes_manifest.json"
        summary_path = Path("data/intermediate") / video_id / "summary_script.json"
        
        if not video_path.exists():
            raise FileNotFoundError(f"Output video not found: {video_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Keyframe manifest not found: {manifest_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary script not found: {summary_path}")
            
        # 2. Load manifest & summary scripts
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            
        scenes_dict = {s["id"]: s for s in manifest_data["scenes"]}
        sentences_dict = {s["id"]: s for s in summary_data["sentences"]}
        
        # 3. Load unified assignments
        groups = load_unified_assignments(video_id, arm)
        
        # Reconstruct sentence-level scene sequence
        scene_sequence = []
        for g in groups:
            for sid in g["sentence_ids"]:
                scene_sequence.append(g["scene_id"])
                
        # 4. Compute direct structural metrics (Scene Diversity, Max Consecutive Reuse, Temporal Acc)
        unique_scenes = len(set(scene_sequence))
        total_assignments = len(scene_sequence)
        
        # Metric: Scene Diversity
        results["scene_diversity"] = float((unique_scenes / total_assignments) if total_assignments > 0 else 0.0)
        
        # Metric: Max Consecutive Reuse
        results["max_consecutive_reuse"] = int(compute_max_consecutive(scene_sequence))
        
        # Metric: Temporal Accuracy @ 15s
        temporal_hits = 0
        temporal_total = 0
        for g in groups:
            for sid in g["sentence_ids"]:
                sentence = sentences_dict.get(sid)
                scene = scenes_dict.get(g["scene_id"])
                if not sentence or not scene:
                    continue
                hint = sentence.get("source_timestamp_hint")
                if not hint or len(hint) < 2:
                    continue
                # Use keyframe timestamp
                ts = scene.get("keyframe_timestamp", (scene.get("start_seconds", 0.0) + scene.get("end_seconds", 0.0)) / 2.0)
                
                # Check absolute distance to LLM hint
                if hint[0] <= ts <= hint[1]:
                    error = 0.0
                else:
                    error = min(abs(ts - hint[0]), abs(ts - hint[1]))
                    
                if error <= 15.0:
                    temporal_hits += 1
                temporal_total += 1
                
        results["temporal_accuracy_15s"] = float(temporal_hits / temporal_total if temporal_total > 0 else 0.0)
        
        # 5. Extract Frames from output video for vision-text alignments
        video_dur = get_video_duration(video_path)
        logger.info(f"Video {video_id} duration: {video_dur:.2f}s")
        
        frames_and_texts = []
        for idx, g in enumerate(groups):
            ts = max(0.0, min(g["mid_time"], video_dur - 0.01))
            try:
                frame = extract_frame_at_time(video_path, ts)
                frames_and_texts.append((frame, g["text"]))
            except Exception as fe:
                logger.error(f"Failed to extract frame at {ts}s: {fe}")
                
        if not frames_and_texts:
            raise ValueError("No video frames could be extracted for evaluation.")
            
        # 6. Evaluate CLIPScore
        clip_model, clip_proc = models_cache["clip"]
        clip_scores = []
        for frame, text in frames_and_texts:
            score = compute_clipscore(frame, text, clip_model, clip_proc, device)
            clip_scores.append(score)
        results["clipscore_mean"] = float(np.mean(clip_scores))
        results["clipscore_std"] = float(np.std(clip_scores))
        
        # 7. Evaluate BLIPScore
        blip_model, blip_proc = models_cache["blip"]
        blip_scores = []
        for frame, text in frames_and_texts:
            score = compute_blipscore(frame, text, blip_model, blip_proc, device)
            blip_scores.append(score)
        results["blipscore_mean"] = float(np.mean(blip_scores))
        results["blipscore_std"] = float(np.std(blip_scores))
        
        # 8. Evaluate LLM-Judge (3 dimensions)
        qwen_model, qwen_proc = models_cache["qwen"]
        
        # Sample exactly 6 keyframes evenly across the video
        num_keyframes = 6
        judge_timestamps = [(i + 0.5) * (video_dur / num_keyframes) for i in range(num_keyframes)]
        judge_images = []
        for ts in judge_timestamps:
            ts_clamped = max(0.0, min(ts, video_dur - 0.01))
            judge_images.append(extract_frame_at_time(video_path, ts_clamped))
            
        full_narration = " ".join([s["text"] for s in summary_data["sentences"]])
        
        # Perform 3 judge trials to capture model variance (MT-bench multi-sampling)
        judge_trials = []
        for trial in range(3):
            temp_files = []
            content_list = []
            try:
                for img in judge_images:
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    img.save(tmp.name)
                    temp_files.append(tmp.name)
                    content_list.append({"type": "image", "image": f"file://{tmp.name}"})
                content_list.append({"type": "text", "text": JUDGE_VISUAL_USER_TEMPLATE.format(narration=full_narration)})
                
                messages = [
                    {"role": "system", "content": JUDGE_VISUAL_SYSTEM},
                    {"role": "user", "content": content_list}
                ]
                
                prompt_text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = qwen_proc(
                    text=[prompt_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    gen_ids = qwen_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.05
                    )
                    gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                    response = qwen_proc.batch_decode(
                        gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    
                parsed_json = json.loads(extract_json_content(response))
                judge_trials.append(parsed_json)
            except Exception as je:
                logger.warning(f"Qwen judge trial {trial+1} failed: {je}")
            finally:
                for f in temp_files:
                    try:
                        Path(f).unlink(missing_ok=True)
                    except Exception:
                        pass
                        
        if judge_trials:
            # Average scores over successful trials
            coherences = [int(t["visual_narration_coherence"]) for t in judge_trials if "visual_narration_coherence" in t]
            consistencies = [int(t["temporal_consistency"]) for t in judge_trials if "temporal_consistency" in t]
            qualities = [int(t["visual_quality"]) for t in judge_trials if "visual_quality" in t]
            
            results["llm_judge_coherence"] = float(np.mean(coherences)) if coherences else "NaN"
            results["llm_judge_consistency"] = float(np.mean(consistencies)) if consistencies else "NaN"
            results["llm_judge_quality"] = float(np.mean(qualities)) if qualities else "NaN"
        else:
            raise ValueError("All LLM-Judge evaluation trials failed.")
            
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error evaluating {video_id}: {e}")
        log_error(video_id, "unified_eval", str(e), tb)
        results["status"] = "failed"
        results["error_message"] = str(e)
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluator for Narrated Video Summarization")
    parser.add_argument("--video", type=str, help="Specific video ID (e.g. review_1)")
    parser.add_argument("--arm", type=str, default="grouping_gate", help="Ablation arm or retrieval method (e.g. grouping_gate, caption_temporal_ccma)")
    parser.add_argument("--all", action="store_true", help="Evaluate all sorted review videos")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Initialize VRAM Manager
    vram_manager = VRAMManager(
        device_id=config.get("vram", {}).get("device_id", 0),
        limit_gb=config.get("vram", {}).get("limit_gb", 22.0)
    )
    
    # 2. Get Video list
    if args.all:
        video_ids = get_video_ids()
    elif args.video:
        video_ids = [args.video]
    else:
        video_ids = get_video_ids()[:1]  # Default to review_1 if none specified
        
    logger.info(f"Target videos for evaluation: {video_ids}")
    
    # 3. Model Loading Helper functions (sequential VRAM friendly loading)
    models_cache = {}
    
    def load_clip():
        cache_path = os.path.expanduser("~/models/clip_vit_l14")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_path, local_files_only=True)
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_path, local_files_only=True).to("cuda").eval()
        return model, proc
    
    # Load BLIP
    def load_blip():
        logger.info("Loading BLIP ITM model Salesforce/blip-itm-base-coco...")
        proc = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to("cuda").eval()
        return model, proc
        
    # Load Qwen VL
    def load_qwen():
        logger.info("Loading Qwen2.5-VL model Qwen/Qwen2.5-VL-7B-Instruct...")
        proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        return model, proc

    # Initialize results container
    aggregated_results = []
    
    # 4. Sequentially execute metrics to strictly stay within VRAM bounds
    
    # Step 4A: Load Structural Metrics (doesn't need heavy weights, runs first)
    structural_records = []
    for video_id in video_ids:
        try:
            groups = load_unified_assignments(video_id, args.arm)
            scene_sequence = [g["scene_id"] for g in groups for sid in g["sentence_ids"]]
            unique_scenes = len(set(scene_sequence))
            total_assignments = len(scene_sequence)
            div = float(1.0 - (unique_scenes / total_assignments) if total_assignments > 0 else 0.0)
            max_reuse = int(compute_max_consecutive(scene_sequence))
            
            # Load metadata for temporal
            manifest_path = Path("data/intermediate") / video_id / "keyframes_manifest.json"
            summary_path = Path("data/intermediate") / video_id / "summary_script.json"
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
            scenes_dict = {s["id"]: s for s in manifest_data["scenes"]}
            sentences_dict = {s["id"]: s for s in summary_data["sentences"]}
            
            temporal_hits = 0
            temporal_total = 0
            for g in groups:
                for sid in g["sentence_ids"]:
                    sentence = sentences_dict.get(sid)
                    scene = scenes_dict.get(g["scene_id"])
                    if not sentence or not scene:
                        continue
                    hint = sentence.get("source_timestamp_hint")
                    if not hint or len(hint) < 2:
                        continue
                    ts = scene.get("keyframe_timestamp", (scene.get("start_seconds", 0.0) + scene.get("end_seconds", 0.0)) / 2.0)
                    error = 0.0 if hint[0] <= ts <= hint[1] else min(abs(ts - hint[0]), abs(ts - hint[1]))
                    if error <= 15.0:
                        temporal_hits += 1
                    temporal_total += 1
            temp_acc = float(temporal_hits / temporal_total if temporal_total > 0 else 0.0)
            
            structural_records.append({
                "video_id": video_id,
                "scene_diversity": div,
                "max_consecutive_reuse": max_reuse,
                "temporal_accuracy_15s": temp_acc
            })
        except Exception as e:
            logger.error(f"Error preparing structural metrics for {video_id}: {e}")
            structural_records.append({
                "video_id": video_id,
                "scene_diversity": "NaN",
                "max_consecutive_reuse": "NaN",
                "temporal_accuracy_15s": "NaN"
            })
            
    # Step 4B: Evaluate CLIPScore
    clip_scores_map = {}
    try:
        clip_model, clip_proc = vram_manager.load_model("CLIP", load_clip)
        for video_id in video_ids:
            try:
                groups = load_unified_assignments(video_id, args.arm)
                video_name = "summary_grouping_gate.mp4" if args.arm in ["grouping_gate", "p4_assignments", "grouping"] else f"summary_{args.arm}.mp4"
                video_path = Path("data/output") / video_id / video_name
                if not video_path.exists():
                    video_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
                video_dur = get_video_duration(video_path)
                
                scores = []
                for g in groups:
                    ts = max(0.0, min(g["mid_time"], video_dur - 0.01))
                    frame = extract_frame_at_time(video_path, ts)
                    scores.append(compute_clipscore(frame, g["text"], clip_model, clip_proc, f"cuda:{vram_manager.device_id}"))
                clip_scores_map[video_id] = (float(np.mean(scores)), float(np.std(scores)))
            except Exception as e:
                logger.error(f"CLIP evaluation failed for {video_id}: {e}")
                clip_scores_map[video_id] = ("NaN", "NaN")
    finally:
        vram_manager.unload_current_model()
        torch.cuda.empty_cache()
        
    # Step 4C: Evaluate BLIPScore
    blip_scores_map = {}
    try:
        blip_model, blip_proc = vram_manager.load_model("BLIP", load_blip)
        for video_id in video_ids:
            try:
                groups = load_unified_assignments(video_id, args.arm)
                video_name = "summary_grouping_gate.mp4" if args.arm in ["grouping_gate", "p4_assignments", "grouping"] else f"summary_{args.arm}.mp4"
                video_path = Path("data/output") / video_id / video_name
                if not video_path.exists():
                    video_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
                video_dur = get_video_duration(video_path)
                
                scores = []
                for g in groups:
                    ts = max(0.0, min(g["mid_time"], video_dur - 0.01))
                    frame = extract_frame_at_time(video_path, ts)
                    scores.append(compute_blipscore(frame, g["text"], blip_model, blip_proc, f"cuda:{vram_manager.device_id}"))
                blip_scores_map[video_id] = (float(np.mean(scores)), float(np.std(scores)))
            except Exception as e:
                logger.error(f"BLIP evaluation failed for {video_id}: {e}")
                blip_scores_map[video_id] = ("NaN", "NaN")
    finally:
        vram_manager.unload_current_model()
        torch.cuda.empty_cache()

    # Step 4D: Evaluate LLM-Judge (Qwen-VL)
    judge_scores_map = {}
    try:
        qwen_model, qwen_proc = vram_manager.load_model("QwenVL-7B", load_qwen)
        for video_id in video_ids:
            try:
                video_name = "summary_grouping_gate.mp4" if args.arm in ["grouping_gate", "p4_assignments", "grouping"] else f"summary_{args.arm}.mp4"
                video_path = Path("data/output") / video_id / video_name
                if not video_path.exists():
                    video_path = Path("data/output") / video_id / "summary_grouping_gate.mp4"
                video_dur = get_video_duration(video_path)
                
                num_keyframes = 6
                judge_timestamps = [(i + 0.5) * (video_dur / num_keyframes) for i in range(num_keyframes)]
                judge_images = []
                for ts in judge_timestamps:
                    ts_clamped = max(0.0, min(ts, video_dur - 0.01))
                    judge_images.append(extract_frame_at_time(video_path, ts_clamped))
                    
                summary_path = Path("data/intermediate") / video_id / "summary_script.json"
                with open(summary_path, "r") as f:
                    sum_data = json.load(f)
                full_narration = " ".join([s["text"] for s in sum_data["sentences"]])
                
                trial_results = []
                for trial in range(3):
                    temp_files = []
                    content_list = []
                    try:
                        for img in judge_images:
                            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                            img.save(tmp.name)
                            temp_files.append(tmp.name)
                            content_list.append({"type": "image", "image": f"file://{tmp.name}"})
                        content_list.append({"type": "text", "text": JUDGE_VISUAL_USER_TEMPLATE.format(narration=full_narration)})
                        
                        messages = [
                            {"role": "system", "content": JUDGE_VISUAL_SYSTEM},
                            {"role": "user", "content": content_list}
                        ]
                        
                        prompt_text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        
                        inputs = qwen_proc(
                            text=[prompt_text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                        ).to(f"cuda:{vram_manager.device_id}")
                        
                        with torch.no_grad():
                            gen_ids = qwen_model.generate(
                                **inputs,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.05
                            )
                            gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                            response = qwen_proc.batch_decode(
                                gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )[0]
                            
                        parsed_json = json.loads(extract_json_content(response))
                        trial_results.append(parsed_json)
                    except Exception as trial_err:
                        logger.warning(f"Trial {trial+1} failed for {video_id}: {trial_err}")
                    finally:
                        for f in temp_files:
                            try:
                                Path(f).unlink(missing_ok=True)
                            except Exception:
                                pass
                                
                if trial_results:
                    coherences = [int(t["visual_narration_coherence"]) for t in trial_results if "visual_narration_coherence" in t]
                    consistencies = [int(t["temporal_consistency"]) for t in trial_results if "temporal_consistency" in t]
                    qualities = [int(t["visual_quality"]) for t in trial_results if "visual_quality" in t]
                    judge_scores_map[video_id] = (
                        float(np.mean(coherences)) if coherences else "NaN",
                        float(np.mean(consistencies)) if consistencies else "NaN",
                        float(np.mean(qualities)) if qualities else "NaN"
                    )
                else:
                    judge_scores_map[video_id] = ("NaN", "NaN", "NaN")
            except Exception as e:
                logger.error(f"Judge evaluation failed for {video_id}: {e}")
                judge_scores_map[video_id] = ("NaN", "NaN", "NaN")
    finally:
        vram_manager.unload_current_model()
        torch.cuda.empty_cache()
        
    # 5. Consolidate results
    unified_rows = []
    for idx, struct_rec in enumerate(structural_records):
        video_id = struct_rec["video_id"]
        c_mean, c_std = clip_scores_map.get(video_id, ("NaN", "NaN"))
        b_mean, b_std = blip_scores_map.get(video_id, ("NaN", "NaN"))
        j_coh, j_con, j_qual = judge_scores_map.get(video_id, ("NaN", "NaN", "NaN"))
        
        row = {
            "video_id": video_id,
            "arm": args.arm,
            "clipscore_mean": c_mean,
            "clipscore_std": c_std,
            "blipscore_mean": b_mean,
            "blipscore_std": b_std,
            "llm_judge_coherence": j_coh,
            "llm_judge_consistency": j_con,
            "llm_judge_quality": j_qual,
            "scene_diversity": struct_rec["scene_diversity"],
            "max_consecutive_reuse": struct_rec["max_consecutive_reuse"],
            "temporal_accuracy_15s": struct_rec["temporal_accuracy_15s"]
        }
        unified_rows.append(row)
        
    # Write to data/evaluation/unified_evaluation_results.csv
    eval_dir = Path("data/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / f"unified_eval_{args.arm}.csv"
    
    fieldnames = [
        "video_id", "arm", "clipscore_mean", "clipscore_std", 
        "blipscore_mean", "blipscore_std", "llm_judge_coherence", 
        "llm_judge_consistency", "llm_judge_quality", 
        "scene_diversity", "max_consecutive_reuse", "temporal_accuracy_15s"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unified_rows)
        
    # Print results summary table
    print("\n" + "="*80)
    print(f"UNIFIED EVALUATION RESULTS FOR ARM: {args.arm}")
    print("="*80)
    print(f"{'Video ID':<10} | {'CLIPScore':<9} | {'BLIPScore':<9} | {'Judge Coh/Con/Qual':<18} | {'Diversity':<9} | {'MaxReuse':<8} | {'TempAcc15s':<10}")
    print("-"*80)
    for r in unified_rows:
        clip_str = f"{r['clipscore_mean']:.4f}" if isinstance(r['clipscore_mean'], float) else "NaN"
        blip_str = f"{r['blipscore_mean']:.4f}" if isinstance(r['blipscore_mean'], float) else "NaN"
        
        coh_str = f"{r['llm_judge_coherence']:.1f}" if isinstance(r['llm_judge_coherence'], float) else "NaN"
        con_str = f"{r['llm_judge_consistency']:.1f}" if isinstance(r['llm_judge_consistency'], float) else "NaN"
        q_str = f"{r['llm_judge_quality']:.1f}" if isinstance(r['llm_judge_quality'], float) else "NaN"
        judge_str = f"{coh_str}/{con_str}/{q_str}"
        
        div_str = f"{r['scene_diversity']:.4f}" if isinstance(r['scene_diversity'], float) else "NaN"
        reuse_str = f"{r['max_consecutive_reuse']}" if isinstance(r['max_consecutive_reuse'], int) else "NaN"
        temp_str = f"{r['temporal_accuracy_15s']:.4f}" if isinstance(r['temporal_accuracy_15s'], float) else "NaN"
        
        print(f"{r['video_id']:<10} | {clip_str:<9} | {blip_str:<9} | {judge_str:<18} | {div_str:<9} | {reuse_str:<8} | {temp_str:<10}")
    print("="*80)
    print(f"Results saved to: {csv_path}\n")

if __name__ == "__main__":
    main()

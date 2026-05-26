import os
import json
import logging
import signal
import sys
import subprocess
from pathlib import Path
from PIL import Image
import soundfile as sf
import torch
from typing import Optional, Union

from src.utils.vram import VRAMManager
from src.utils.io import load_json_as_model

logger = logging.getLogger(__name__)

# Global tracker for paused PIDs to resume in signal handler
_paused_ollama_pids = []

def _signal_handler(signum, frame):
    logger.warning(f"Signal {signum} received. Cleaning up...")
    if _paused_ollama_pids:
        logger.info(f"Emergency resuming Ollama processes: {_paused_ollama_pids}")
        for pid in _paused_ollama_pids:
            try:
                os.kill(pid, signal.SIGCONT)
            except:
                pass
    sys.exit(1)

def register_signal_handlers():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


class PromptBuilder:
    """Stage A: construct LTX prompts via Qwen2.5-VL multimodal LLM."""
    
    def __init__(self, vram_manager: VRAMManager, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"):
        self.vram = vram_manager
        self.model_id = model_id
        
    def _pause_ollama(self):
        try:
            result = subprocess.run(["pgrep", "-f", "ollama"], capture_output=True, text=True)
            pids = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            my_pid = str(os.getpid())
            pids = [pid for pid in pids if pid != my_pid]
            
            if pids:
                logger.info(f"Pausing Ollama processes: {pids}")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGSTOP)
                        _paused_ollama_pids.append(int(pid))
                    except ProcessLookupError:
                        pass
                    except PermissionError as pe:
                        logger.warning(f"No permission to pause PID {pid}: {pe}")
            else:
                logger.info("No running Ollama processes found to pause.")
        except Exception as e:
            logger.warning(f"Failed to pause Ollama: {e}")

    def _resume_ollama(self):
        if _paused_ollama_pids:
            logger.info(f"Resuming Ollama processes: {_paused_ollama_pids}")
            for pid in _paused_ollama_pids:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.error(f"Failed to resume PID {pid}: {e}")
            _paused_ollama_pids.clear()

    def build_prompts(self, video_id: str, rebuild_prompts: bool = False, intermediate_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Read Phase 2/3/4 outputs, generate LTX prompts for all generate groups.
        Save to data/intermediate/{video_id}/ltx_prompts.json
        Return path to generated json.
        """
        register_signal_handlers()
        
        base_dir = Path(intermediate_dir) if intermediate_dir is not None else Path("data/intermediate")
        intermediate_dir = base_dir / video_id
        prompts_json_path = intermediate_dir / "ltx_prompts.json"
        
        if prompts_json_path.exists() and not rebuild_prompts:
            logger.info(f"Existing ltx_prompts.json found at {prompts_json_path}. Skipping Stage A prompt building.")
            return prompts_json_path

        p4_assignments_path = intermediate_dir / "p4_assignments.json"
        summary_script_path = intermediate_dir / "summary_script.json"
        keyframes_manifest_path = intermediate_dir / "keyframes_manifest.json"
        
        if not p4_assignments_path.exists():
            raise FileNotFoundError(f"Missing Phase 4 assignments: {p4_assignments_path}")
        if not summary_script_path.exists():
            raise FileNotFoundError(f"Missing summary script: {summary_script_path}")
        if not keyframes_manifest_path.exists():
            raise FileNotFoundError(f"Missing keyframe manifest: {keyframes_manifest_path}")

        # Load input data
        with open(p4_assignments_path, "r") as f:
            assignments = json.load(f)
            
        with open(summary_script_path, "r") as f:
            summary_script = json.load(f)
            
        with open(keyframes_manifest_path, "r") as f:
            keyframes_manifest = json.load(f)

        scenes_map = {s["id"]: s for s in keyframes_manifest["scenes"]}
        sentences_list = summary_script["sentences"]

        output_groups = []
        generate_groups_to_process = []
        
        # Prepare metadata for each group
        for group_idx, group in enumerate(assignments):
            action = group["action"]
            sentence_ids = group["sentence_ids"]
            
            # Lookup sentences (note: summary_script sentences are objects with key "id", e.g. 1, 2, ...
            # while sentence_ids in group are 0-indexed indices into summary_script["sentences"] list)
            group_sentences = [sentences_list[sid] for sid in sentence_ids]
            
            # 1. Narration
            narration = " ".join(s["text"] for s in group_sentences)
            
            # 2. Keywords
            keywords = []
            for s in group_sentences:
                for kw in s.get("keywords", []):
                    if kw not in keywords:
                        keywords.append(kw)
                        
            # 3. Audio duration
            audio_duration = 0.0
            audio_dir = intermediate_dir / "audio"
            for sid in sentence_ids:
                wav_path = audio_dir / f"sentence_{sid:03d}.wav"
                if wav_path.exists():
                    try:
                        info = sf.info(wav_path)
                        audio_duration += info.duration
                    except Exception as e:
                        logger.error(f"Failed to read audio duration for {wav_path}: {e}")
                else:
                    logger.warning(f"Audio file not found: {wav_path}")
            
            if action != "generate":
                # Skip retrieve groups
                output_groups.append({
                    "group_id": group_idx,
                    "sentence_ids": sentence_ids,
                    "action": action,
                    "audio_duration_seconds": audio_duration,
                    "num_frames": None,
                    "keyframe_path": None,
                    "keyframe_preprocessed_path": None,
                    "narration": narration,
                    "keywords": keywords,
                    "prompt": None
                })
                continue
                
            # 4. Pick keyframe
            scene_id = group["scene_id"]
            scene = scenes_map.get(scene_id)
            if not scene:
                raise ValueError(f"Scene ID {scene_id} not found in keyframes manifest")
                
            # Find the frame closest to center of timestamp_hint_merged
            hint_merged = group["timestamp_hint_merged"]
            hint_center = sum(hint_merged) / 2.0
            
            timestamps = scene.get("multi_frame_timestamps", [])
            paths = scene.get("multi_frame_paths", [])
            
            if timestamps and paths:
                closest_idx = min(range(len(timestamps)), key=lambda idx: abs(timestamps[idx] - hint_center))
                keyframe_rel_path = paths[closest_idx]
            else:
                keyframe_rel_path = scene.get("keyframe_path")
                
            if not keyframe_rel_path:
                raise ValueError(f"No keyframe paths found for scene {scene_id}")
                
            keyframe_abs_path = intermediate_dir / keyframe_rel_path
            
            # Preprocess keyframe (resize and center crop to 768x512)
            preprocessed_rel_path = f"keyframes_ltx/group_{group_idx:03d}_keyframe_768x512.jpg"
            preprocessed_abs_path = intermediate_dir / preprocessed_rel_path
            
            # Crop/resize
            self._preprocess_keyframe(keyframe_abs_path, preprocessed_abs_path)
            
            # 5. Decide num_frames
            num_frames = 121 if audio_duration <= 4.0 else 241
            
            group_data = {
                "group_id": group_idx,
                "sentence_ids": sentence_ids,
                "action": action,
                "audio_duration_seconds": audio_duration,
                "num_frames": num_frames,
                "keyframe_path": str(keyframe_rel_path),
                "keyframe_preprocessed_path": str(preprocessed_rel_path),
                "narration": narration,
                "keywords": keywords,
                "prompt": None
            }
            
            output_groups.append(group_data)
            generate_groups_to_process.append((group_data, preprocessed_abs_path))
            
        if generate_groups_to_process:
            self._pause_ollama()
            try:
                # Load Qwen-VL model via VRAMManager
                def loader():
                    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                    if "AWQ" in self.model_id:
                        from awq import AutoAWQForCausalLM
                        model = AutoAWQForCausalLM.from_quantized(
                            self.model_id, 
                            fuse_layers=False, 
                            trust_remote_code=True, 
                            device_map="auto"
                        )
                    else:
                        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            self.model_id,
                            torch_dtype="auto",
                            device_map="auto",
                            trust_remote_code=True
                        )
                    processor = AutoProcessor.from_pretrained(self.model_id)
                    return model, processor
                
                model, processor = self.vram.load_model(f"Qwen2.5-VL ({self.model_id})", loader)
                
                system_prompt = (
                    "You are a visual prompt engineer for an image-to-video diffusion model (LTX-Video).\n"
                    "Your task: given a reference image (a frame from a tech review video), a narration script, \n"
                    "and a list of visual keywords, write a single English prompt (max 80 words) that describes \n"
                    "what the generated video clip should show.\n\n"
                    "Rules for the prompt:\n"
                    "1. Start with the main subject visible in the image (be specific: brand, color, shape, key features).\n"
                    "2. Describe natural camera motion (e.g. \"slow camera pan\", \"smooth dolly forward\", \"static camera with subtle handheld feel\"). For complex scenes with multiple objects, prefer \"static camera, no zoom\".\n"
                    "3. Include details from the narration that are visually depictable.\n"
                    "4. Use the keywords as anchors for specific visual elements.\n"
                    "5. End with style cues: \"tech product review style, soft studio lighting, dark background, shallow depth of field\".\n"
                    "6. Do NOT use negative phrasing (\"no buttons\", \"without X\"). Use positive description only.\n"
                    "7. Do NOT mention sound, music, or text overlays.\n"
                    "8. Single paragraph, no bullet points.\n\n"
                    "Output the prompt text directly, no preamble."
                )
                
                from qwen_vl_utils import process_vision_info
                
                for group_data, keyframe_path in generate_groups_to_process:
                    logger.info(f"Generating prompt for group {group_data['group_id']}...")
                    narration = group_data["narration"]
                    keywords_str = ", ".join(group_data["keywords"])
                    
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(keyframe_path)},
                                {"type": "text", "text": f"Narration: {narration}\nKeywords: {keywords_str}"}
                            ]
                        }
                    ]
                    
                    try:
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(
                            text=[text], 
                            images=image_inputs, 
                            videos=video_inputs, 
                            padding=True, 
                            return_tensors="pt"
                        ).to("cuda")
                        
                        with torch.no_grad():
                            generated_ids = model.generate(
                                **inputs, 
                                max_new_tokens=100, 
                                do_sample=False, 
                                temperature=0.0
                            )
                        
                        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                        output_text = processor.batch_decode(
                            generated_ids_trimmed, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                        )[0].strip()
                        
                        # Clean up formatting if any quote wraps exist
                        if output_text.startswith('"') and output_text.endswith('"'):
                            output_text = output_text[1:-1].strip()
                            
                        logger.info(f"Generated LTX prompt: {output_text}")
                        group_data["prompt"] = output_text
                        
                    except Exception as e:
                        logger.error(f"Failed to generate prompt for group {group_data['group_id']}: {e}", exc_info=True)
                        group_data["prompt"] = None
                        
            finally:
                # Clear local references to allow garbage collection
                model = None
                processor = None
                # Always unload Qwen-VL model to free VRAM
                self.vram.unload_current_model()
                self._resume_ollama()

        # Save to prompts json
        output_data = {
            "video_id": video_id,
            "groups": output_groups
        }
        
        with open(prompts_json_path, "w") as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Stage A prompts written to {prompts_json_path}")
        return prompts_json_path

    def _preprocess_keyframe(self, src_path: Path, dst_path: Path):
        """Resize and center crop a keyframe image to 768x512."""
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src_path).convert("RGB")
        target_w, target_h = 768, 512
        orig_w, orig_h = img.size
        
        # Calculate scale factor to cover the target dimensions
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop to 768x512
        left = (new_w - target_w) / 2
        top = (new_h - target_h) / 2
        right = left + target_w
        bottom = top + target_h
        
        img_cropped = img_resized.crop((left, top, right, bottom))
        img_cropped.save(dst_path, "JPEG", quality=95)
        logger.info(f"Preprocessed keyframe {src_path} saved to {dst_path}")

import os
import gc
import time
import json
import logging
import signal
import sys
import subprocess
from pathlib import Path
from PIL import Image
import torch
from typing import Optional, Union

# 1. Monkeypatch diffusers to fix the LTX retrieve_timesteps bug in both pipelines
# This must be done BEFORE loading pipelines.
try:
    import diffusers.pipelines.ltx.pipeline_ltx as pl_t2v
    import diffusers.pipelines.ltx.pipeline_ltx_image2video as pl_i2v

    orig_t2v_retrieve = pl_t2v.retrieve_timesteps
    orig_i2v_retrieve = pl_i2v.retrieve_timesteps

    def patched_t2v(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
        if timesteps is not None:
            sigmas = None
        return orig_t2v_retrieve(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs)

    def patched_i2v(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
        if timesteps is not None:
            sigmas = None
        return orig_i2v_retrieve(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs)

    pl_t2v.retrieve_timesteps = patched_t2v
    pl_i2v.retrieve_timesteps = patched_i2v
except Exception as e:
    print(f"Error applying LTX retrieve_timesteps monkeypatch: {e}")

from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from src.utils.vram import VRAMManager
from src.utils.ffmpeg_ops import cut_video_segment

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


class LTXRunner:
    """Stage B: generate I2V clips using LTX-Video 0.9.7 distilled."""
    
    def __init__(self, vram_manager: VRAMManager, model_path: str = "/home/wins053/models/ltx_video_distilled"):
        self.vram = vram_manager
        # Expand user path if model_path starts with ~
        self.model_path = os.path.expanduser(model_path)
        
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

    def generate_clips(self, video_id: str, rebuild_clips: bool = False, intermediate_dir: Optional[Union[str, Path]] = None) -> list:
        """
        Read ltx_prompts.json, generate clips for all generate groups,
        save to data/intermediate/{video_id}/generated/group_{group_id}.mp4.
        Return list of generated clip paths.
        Skip groups that already have output file (resume support).
        """
        register_signal_handlers()
        
        base_dir = Path(intermediate_dir) if intermediate_dir is not None else Path("data/intermediate")
        intermediate_dir = base_dir / video_id
        prompts_json_path = intermediate_dir / "ltx_prompts.json"
        
        if not prompts_json_path.exists():
            raise FileNotFoundError(f"Missing prompts file: {prompts_json_path}")
            
        with open(prompts_json_path, "r") as f:
            prompt_data = json.load(f)
            
        groups = prompt_data.get("groups", [])
        
        # Check how many groups actually need generation
        groups_to_generate = []
        generated_dir = intermediate_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        
        for g in groups:
            if g.get("action") == "generate":
                group_id = g["group_id"]
                output_path = generated_dir / f"group_{group_id:03d}.mp4"
                if output_path.exists() and not rebuild_clips:
                    logger.info(f"Clip for group {group_id} already exists. Skipping.")
                else:
                    groups_to_generate.append(g)
                    
        if not groups_to_generate:
            logger.info("No clips need to be generated (all skipped or completed).")
            return [generated_dir / f"group_{g['group_id']:03d}.mp4" for g in groups if g.get("action") == "generate"]

        self._pause_ollama()
        
        # We keep track of peak VRAM and inference times
        metrics = {}
        metrics_file = intermediate_dir / "generation_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
            except:
                pass

        use_sequential_offload = False
        custom_timesteps = [1000, 993, 987, 981, 975, 909, 725, 0.03]
        
        pipeline = None
        
        def load_pipeline():
            nonlocal pipeline
            # Clear local reference first to allow garbage collection
            pipeline = None
            # Unload current model first to force reload in vram manager
            self.vram.unload_current_model()
            # Clear VRAM before load
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            def loader():
                pipe = LTXImageToVideoPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16
                )
                pipe.vae.enable_tiling()
                if use_sequential_offload:
                    logger.info("Enabling sequential CPU offloading for LTX...")
                    pipe.enable_sequential_cpu_offload()
                else:
                    logger.info("Enabling standard model CPU offloading for LTX...")
                    pipe.enable_model_cpu_offload()
                return pipe
                
            pipeline = self.vram.load_model("LTX-Video", loader)

        try:
            load_pipeline()
            
            generated_paths = []
            for g in groups_to_generate:
                group_id = g["group_id"]
                prompt = g["prompt"]
                num_frames = g["num_frames"]
                audio_duration = g["audio_duration_seconds"]
                keyframe_preprocessed_path = intermediate_dir / g["keyframe_preprocessed_path"]
                
                output_path = generated_dir / f"group_{group_id:03d}.mp4"
                
                logger.info(f"Generating clip for group {group_id} ({num_frames} frames, duration {audio_duration:.2f}s)...")
                logger.info(f"Prompt: {prompt}")
                
                if not keyframe_preprocessed_path.exists():
                    logger.error(f"Preprocessed keyframe not found: {keyframe_preprocessed_path}. Skipping.")
                    continue
                    
                image = Image.open(keyframe_preprocessed_path).convert("RGB")
                
                success = False
                attempts = 0
                need_reload = False
                while not success and attempts < 2:
                    attempts += 1
                    if need_reload:
                        logger.warning("VRAM OOM detected. Deferring reload with sequential CPU offloading...")
                        use_sequential_offload = True
                        load_pipeline()
                        need_reload = False
                        
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        start_time = time.time()
                        
                        generator = torch.Generator("cuda").manual_seed(42)
                        output = pipeline(
                            prompt=prompt,
                            image=image,
                            num_frames=num_frames,
                            width=768,
                            height=512,
                            guidance_scale=1.0,
                            timesteps=custom_timesteps,
                            generator=generator
                        )
                        
                        latency = time.time() - start_time
                        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        
                        logger.info(f"Generation successful for group {group_id} (latency={latency:.2f}s, VRAM={peak_vram:.2f} GB)")
                        
                        # Export frames
                        frames = output.frames[0] if isinstance(output.frames[0], list) else output.frames
                        
                        # Save to a temporary clip path first
                        temp_clip_path = generated_dir / f"temp_group_{group_id:03d}.mp4"
                        export_to_video(frames, str(temp_clip_path), fps=30)
                        
                        # Trim to exact audio_duration
                        logger.info(f"Trimming generated clip to exact audio duration: {audio_duration:.2f}s")
                        cut_video_segment(temp_clip_path, 0.0, audio_duration, output_path, reencode=False)
                        
                        # Cleanup temp file
                        if temp_clip_path.exists():
                            os.remove(temp_clip_path)
                            
                        generated_paths.append(output_path)
                        success = True
                        
                        # Save metrics
                        metrics[f"group_{group_id}"] = {
                            "latency_seconds": latency,
                            "peak_vram_gb": peak_vram,
                            "offload_style": "sequential" if use_sequential_offload else "model",
                            "audio_duration": audio_duration,
                            "num_frames": num_frames
                        }
                        
                    except Exception as e:
                        err_msg = str(e)
                        logger.error(f"Error during LTX generation for group {group_id}: {err_msg}")
                        is_oom = "OutOfMemoryError" in err_msg or "OOM" in err_msg or "out of memory" in err_msg.lower()
                        
                        # Clear local references to allow garbage collection
                        output = None
                        
                        if is_oom and not use_sequential_offload:
                            need_reload = True
                            e = None
                        else:
                            logger.error(f"Generation failed for group {group_id} on attempt {attempts}: {e}")
                            break
                            
            # Write metrics
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
                
            return generated_paths
            
        finally:
            pipeline = None
            self.vram.unload_current_model()
            self._resume_ollama()

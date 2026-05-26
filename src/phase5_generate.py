import os
import argparse
import logging
import yaml
from pathlib import Path

from src.utils.vram import VRAMManager
from src.phase5_prompt_builder import PromptBuilder
from src.phase5_ltx_runner import LTXRunner

logger = logging.getLogger(__name__)

def run_phase5_generate(
    video_id: str, 
    config: dict, 
    vram_manager: VRAMManager, 
    rebuild_prompts: bool = False, 
    rebuild_clips: bool = False
) -> list:
    """
    Run Phase 5 clip generation pipeline.
    Stage A: Prompt Construction using Qwen-VL.
    Stage B: LTX Clip Generation using LTX-Video.
    """
    logger.info(f"Starting Phase 5 clip generation for video: {video_id}")
    
    # Resolve intermediate_dir from config
    intermediate_dir = config.get("paths", {}).get("intermediate_dir", "data/intermediate")
    
    # 1. Resolve model config paths
    qwen_cfg = config.get("models", {}).get("qwen_vl", {})
    # Use 3B AWQ as default for prompts if not configured
    qwen_model_id = qwen_cfg.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    
    ltx_cfg = config.get("models", {}).get("ltx", {})
    ltx_model_path = ltx_cfg.get("model_path", "/home/wins053/models/ltx_video_distilled")
    
    # 2. Stage A: Prompt Construction
    prompt_builder = PromptBuilder(vram_manager=vram_manager, model_id=qwen_model_id)
    prompts_json_path = prompt_builder.build_prompts(video_id=video_id, rebuild_prompts=rebuild_prompts, intermediate_dir=intermediate_dir)
    
    # 3. Stage B: Clip Generation
    ltx_runner = LTXRunner(vram_manager=vram_manager, model_path=ltx_model_path)
    generated_clips = ltx_runner.generate_clips(video_id=video_id, rebuild_clips=rebuild_clips, intermediate_dir=intermediate_dir)
    
    logger.info(f"Phase 5 generation completed for video: {video_id}. Generated {len(generated_clips)} clips.")
    return generated_clips

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Orchestrate Phase 5 LTX generation.")
    parser.add_argument("--video-id", type=str, required=True, help="Video ID (e.g. review_1)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--rebuild-prompts", action="store_true", help="Force rebuild prompts even if JSON exists")
    parser.add_argument("--rebuild-clips", action="store_true", help="Force rebuild clips even if MP4 exists")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    vram_manager = VRAMManager()
    
    try:
        clips = run_phase5_generate(
            video_id=args.video_id,
            config=config,
            vram_manager=vram_manager,
            rebuild_prompts=args.rebuild_prompts,
            rebuild_clips=args.rebuild_clips
        )
        print(f"Success! Generated clips: {clips}")
    except Exception as e:
        logger.error(f"Generation orchestration failed: {e}", exc_info=True)
        exit(1)

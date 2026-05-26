import os
import yaml
import logging
import json
import re
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.pipeline import VideoSummarizerPipeline
from src.utils.io import load_json_as_model
from src.schemas import SummaryScript, TranscriptSchema
from src.phase2_summarize import Phase2Summarizer, SYSTEM_PROMPT
from src.models.llm_wrapper import LocalBackend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Monkey-patch LocalBackend to fix device map
def patch_load_model(self):
    logger.info(f"Loading model (patched): {self.model_name}")
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    device_id = self.vram.device_id
    device_map = {"": f"cuda:{device_id}"}
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        device_map=device_map,
        trust_remote_code=True
    )

LocalBackend._load_model = patch_load_model

def debug_review_9():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    pipeline = VideoSummarizerPipeline(config)
    llm_backend = pipeline.llm_backend
    
    video_id = "review_9"
    transcript_path = Path(f"data/intermediate/{video_id}/transcript.json")
    
    if not transcript_path.exists():
        logger.error(f"Transcript for {video_id} not found at {transcript_path}")
        return

    p2 = Phase2Summarizer(llm_backend, config.get("summarization", {}))
    
    transcript = load_json_as_model(transcript_path, TranscriptSchema)
    chunks = p2._chunk_transcript(transcript)
    schema_json = SummaryScript.model_json_schema()
    sys_prompt = SYSTEM_PROMPT.format(
        target_duration=60, 
        schema_json=json.dumps(schema_json)
    )
    user_prompt = f"VIDEO_ID: {video_id}\n\nTRANSCRIPT:\n" + chunks[0]
    
    logger.info("Generating for review_9...")
    response = llm_backend.generate(sys_prompt, user_prompt)
    
    print("\n=== RAW LLM RESPONSE FOR REVIEW_9 ===")
    print(response)
    print("=====================================\n")
    
    try:
        data = p2._extract_json(response)
        print("Successfully extracted JSON!")
        print(json.dumps(data, indent=2)[:500] + "...")
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    debug_review_9()

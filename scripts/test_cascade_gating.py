#!/usr/bin/env python
import os
import sys
import yaml
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.vram import VRAMManager
from src.schemas import SummaryScript, KeyframesManifest, KeyframeScene
from src.phase4_retrieve import RetrievalGate, RetrievalGateConfig, Sentence, Scene
# SigLIPModel import removed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_cascade_gating")

def main():
    print("\n" + "="*80)
    print("TESTING SOTA INNOVATION: CASCADE ENTITY VERIFICATION GATING (QWEN-VL)")
    print("="*80 + "\n")

    # Load config
    config_path = Path("configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Force enable cascade verification and ensure model names match
    config["phase4"]["enable_cascade_verification"] = True
    # Let's use 3B AWQ for speed in testing
    config["models"]["qwen_vl"]["model_name"] = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

    vram_manager = VRAMManager(device_id=0, limit_gb=22.0)

    # Load summary and keyframes manifest for review_1
    video_id = "review_1"
    summary_path = Path(f"data/intermediate/{video_id}/summary_script.json")
    manifest_path = Path(f"data/intermediate/{video_id}/keyframes_manifest.json")

    if not summary_path.exists() or not manifest_path.exists():
        logger.error("Source intermediate files for review_1 not found.")
        return

    with open(summary_path, "r") as f:
        summary = SummaryScript.model_validate_json(f.read())
    
    with open(manifest_path, "r") as f:
        manifest = KeyframesManifest.model_validate_json(f.read())

    # Map Summary sentences to P4 Sentences
    p4_sentences = [
        Sentence(
            id=s.id,
            text=s.text,
            timestamp_hint=(float(s.source_timestamp_hint[0]), float(s.source_timestamp_hint[1]))
        )
        for s in summary.sentences
    ]

    # Initialize SigLIP text tower and encode scene embeddings
    logger.info("Initializing SigLIP text encoder to extract sentence and scene matching...")
    
    def load_siglip():
        from transformers import AutoProcessor, Siglip2Model
        model_name = config["models"]["siglip"]["model_name"]
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Siglip2Model.from_pretrained(model_name, ignore_mismatched_sizes=True).to("cuda").eval()
        return model, processor

    siglip_model, siglip_proc = vram_manager.load_model("SigLIP2", load_siglip)
    
    # Simple encoder wrapper class
    class TextEncoderWrapper:
        def __init__(self, model, proc):
            self.model = model
            self.proc = proc
        def encode(self, text: str):
            import torch
            inputs = self.proc(text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    features = getattr(features, "pooler_output", features[0])
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy()

    encoder = TextEncoderWrapper(siglip_model, siglip_proc)

    # Prepare scene embeddings (Normally cached, we simulate a mock representation)
    import numpy as np
    p4_scenes = []
    for sc in manifest.scenes:
        # Create a mock normal pooled embedding of same size (768 or similar)
        # In real pipeline this is loaded from cache. We'll populate mock array for test
        p4_scenes.append(Scene(
            id=sc.id,
            start=sc.start_seconds,
            end=sc.end_seconds,
            embedding=np.random.normal(size=1152) # google/siglip2-so400m uses dim 1152
        ))

    logger.info(f"Loaded {len(p4_sentences)} sentences and {len(p4_scenes)} scenes.")

    # Instantiate RetrievalGate with Cascade Gating enabled!
    gate = RetrievalGate(
        text_encoder=encoder,
        config=RetrievalGateConfig(
            gate_threshold=0.02,
            extend_epsilon=0.03,
            max_group_size=5,
            enable_cascade_verification=True
        ),
        vram_manager=vram_manager,
        pipeline_config=config,
        manifest=manifest
    )

    logger.info("Running Cascade Entity Verification Gate on review_1...")
    # This will load Qwen-VL, check assignments, and override if false!
    assignments = gate.run(p4_sentences, p4_scenes)

    print("\n" + "="*80)
    print("CASCADE GATING RUN COMPLETED: ASSIGNMENTS REPORT")
    print("="*80)
    for a in assignments:
        print(
            f"Group Sentences: {a.sentence_ids}\n"
            f"  Matched Scene: {a.scene_id}\n"
            f"  Weighted Sim: {a.best_similarity:.4f}\n"
            f"  FINAL ACTION: {a.action} " + ("🚫 (OVERRIDDEN TO GENERATE BY QWEN-VL)" if a.action == "generate" and a.best_similarity >= 0.12 else "✅ (APPROVED / RETRIEVED)")
        )
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

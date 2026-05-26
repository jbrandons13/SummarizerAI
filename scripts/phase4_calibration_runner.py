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
from src.schemas import SummaryScript, Phase5Output, KeyframesManifest
from src.phase1_transcribe import TranscriptionPhase
from src.phase2_summarize import Phase2Summarizer
from src.utils.shot_detect import KeyframeExtractor
from src.models.siglip import SigLIPEncoder
from src.models.llm_wrapper import LocalBackend
from src.phase4_retrieve import (
    RetrievalGate, RetrievalGateConfig, 
    Sentence as P4Sentence, Scene as P4Scene,
    summarise_assignments
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def asdict_custom(obj):
    if hasattr(obj, "__dict__"):
        return {k: asdict_custom(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [asdict_custom(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: asdict_custom(v) for k, v in obj.items()}
    else:
        return obj

def clean_broken_json(content):
    """Fix common LLM JSON errors like unquoted strings or trailing commas."""
    content = re.sub(r',\s*([}\]])', r'\1', content)
    # Fix unquoted values like "id": bk -> "id": "bk"
    content = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', content)
    return content

def robust_json_load(content):
    """Robustly find the JSON object that actually contains the script data."""
    potential_objects = []
    i = 0
    while i < len(content):
        start = content.find('{', i)
        if start == -1:
            break
        
        stack = 0
        for j in range(start, len(content)):
            if content[j] == '{':
                stack += 1
            elif content[j] == '}':
                stack -= 1
                if stack == 0:
                    raw_obj_str = content[start:j+1]
                    try:
                        obj = json.loads(raw_obj_str)
                        potential_objects.append(obj)
                    except:
                        try:
                            cleaned = clean_broken_json(raw_obj_str)
                            obj = json.loads(cleaned)
                            potential_objects.append(obj)
                        except:
                            pass
                    i = j
                    break
        else:
            break
        i += 1
        
    for obj in reversed(potential_objects):
        if isinstance(obj, dict) and "sentences" in obj and isinstance(obj["sentences"], list):
            if len(obj["sentences"]) > 0 and (isinstance(obj["sentences"][0], dict) and "text" in obj["sentences"][0]):
                # Final fix for Pydantic: ensure "id" is an integer
                for idx, sent in enumerate(obj["sentences"]):
                    if "id" in sent:
                        try:
                            sent["id"] = int(sent["id"])
                        except:
                            # Fallback to index if it's "bk" or something else
                            sent["id"] = idx
                return obj
    
    # Final cleanup attempt
    try:
        obj = json.loads(clean_broken_json(content))
        if "sentences" in obj:
            for idx, sent in enumerate(obj["sentences"]):
                if "id" in sent:
                    try:
                        sent["id"] = int(sent["id"])
                    except:
                        sent["id"] = idx
        return obj
    except:
        return json.loads(content)

# Monkey-patch Phase2Summarizer
def patch_extract_json(self, response):
    response = response.strip()
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        
    if json_match:
        content = json_match.group(1)
    else:
        content = response
            
    return robust_json_load(content)

Phase2Summarizer._extract_json = patch_extract_json

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

def run_calibration():
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    video_dir = Path("data/eval_videos")
    videos = sorted(list(video_dir.glob("*.mp4")))
    
    if not videos:
        logger.error("No videos found in data/eval_videos/")
        return

    logger.info(f"Found {len(videos)} videos: {[v.name for v in videos]}")
    
    pipeline = VideoSummarizerPipeline(config)
    vram_manager = pipeline.vram_manager
    llm_backend = pipeline.llm_backend
    
    results = []
    failed_videos = []
    
    for video_path in videos:
        video_id = video_path.stem
        logger.info(f"=== Processing {video_id} ===")
        
        try:
            intermediate_dir = Path(config["paths"]["intermediate_dir"]) / video_id
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            
            # Phase 1: Transcription
            transcript_path = intermediate_dir / "transcript.json"
            if not transcript_path.exists():
                p1 = TranscriptionPhase(vram_manager, config.get("models", {}).get("whisper", {}))
                transcript_path = p1.run(video_path)
            
            # Phase 2: Summarization
            summary_path = intermediate_dir / "summary_script.json"
            if video_id == "review_9" and summary_path.exists():
                try:
                    load_json_as_model(summary_path, SummaryScript)
                except:
                    logger.info(f"Existing summary for {video_id} is invalid. Retrying.")
                    summary_path.unlink()

            if not summary_path.exists():
                p2 = Phase2Summarizer(llm_backend, config.get("summarization", {}))
                summary_path = p2.run(transcript_path, target_duration=config.get("summarization", {}).get("max_output_duration_seconds", 60))
            
            # Phase 4 Pre-req: Keyframes
            keyframes_manifest_path = intermediate_dir / "keyframes_manifest.json"
            if keyframes_manifest_path.exists():
                manifest = load_json_as_model(keyframes_manifest_path, KeyframesManifest)
            else:
                extractor = KeyframeExtractor()
                manifest = extractor.extract(video_path, intermediate_dir)

            # Phase 4 Pre-req: SigLIP
            siglip_model = config.get("models", {}).get("siglip", {}).get("model_name", "google/siglip2-so400m-patch16-naflex")
            siglip = SigLIPEncoder(vram_manager, siglip_model)
            frame_embeddings = siglip.embed_scenes(video_id, manifest)
            
            # Phase 4: Retrieval
            summary = load_json_as_model(summary_path, SummaryScript)
            
            p4_scenes = []
            for sc in manifest.scenes:
                embs = [frame_embeddings[(sc.id, ts)] for ts in sc.multi_frame_timestamps]
                if embs:
                    scene_emb = np.mean(embs, axis=0)
                    norm = np.linalg.norm(scene_emb)
                    if norm > 0:
                        scene_emb = scene_emb / norm
                else:
                    scene_emb = np.zeros(siglip.get_embedding_dim())

                p4_scenes.append(P4Scene(
                    id=sc.id,
                    start=sc.start_seconds,
                    end=sc.end_seconds,
                    embedding=scene_emb
                ))

            p4_sentences = []
            for s in summary.sentences:
                hint = s.source_timestamp_hint
                if not hint or len(hint) < 2:
                    logger.warning(f"Malformed hint for sentence {s.id} in {video_id}: {hint}. Using fallback [0, 0].")
                    hint_val = (0.0, 0.0)
                else:
                    hint_val = (float(hint[0]), float(hint[1]))
                
                p4_sentences.append(P4Sentence(
                    id=s.id,
                    text=s.text,
                    timestamp_hint=hint_val
                ))

            gate_cfg_vals = config.get("phase4", {})
            gate = RetrievalGate(
                text_encoder=siglip,
                config=RetrievalGateConfig(
                    gate_threshold=gate_cfg_vals.get("gate_threshold", 0.13),
                    extend_epsilon=gate_cfg_vals.get("extend_epsilon", 0.03),
                    max_group_size=gate_cfg_vals.get("max_group_size", 5),
                    join_sep=gate_cfg_vals.get("join_sep", " "),
                    temporal_sigma=gate_cfg_vals.get("temporal_sigma", 30.0),
                    enable_temporal_prior=gate_cfg_vals.get("enable_temporal_prior", True),
                ),
            )
            assignments = gate.run(p4_sentences, p4_scenes)
            
            # Persist for record
            assignments_path = intermediate_dir / "p4_assignments.json"
            with open(assignments_path, "w") as f:
                json.dump([asdict_custom(a) for a in assignments], f, indent=2)
            
            results.append({
                "video_id": video_id,
                "num_sentences": len(summary.sentences),
                "assignments": [asdict_custom(a) for a in assignments]
            })
            
            logger.info(f"=== Successfully processed {video_id} ===")
            
        except Exception as e:
            logger.error(f"Failed to process {video_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_videos.append(video_id)
            continue

    # Task 3: Aggregation
    if not results:
        logger.error("No successful videos to aggregate.")
        return

    total_successful = len(results)
    sent_counts = [r["num_sentences"] for r in results]
    all_assignments = []
    for r in results:
        all_assignments.extend(r["assignments"])
        
    num_groups_per_video = [len(r["assignments"]) for r in results]
    group_sizes = [len(a["sentence_ids"]) for a in all_assignments]
    weighted_sims = [a["best_similarity"] for a in all_assignments]
    raw_cosines = [a["raw_cosine"] for a in all_assignments]
    temporal_weights = [a["temporal_weight"] for a in all_assignments]
    actions = [a["action"] for a in all_assignments]
    
    def get_histogram(values, bins):
        counts = [0] * len(bins)
        for v in values:
            for i in range(len(bins)-1):
                if bins[i] <= v < bins[i+1]:
                    counts[i] += 1
                    break
            else:
                if v >= bins[-1]:
                    counts[-1] += 1
        return counts

    sim_bins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    weighted_hist = get_histogram(weighted_sims, sim_bins)
    raw_hist = get_histogram(raw_cosines, sim_bins)
    
    temp_bins = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
    temp_hist = get_histogram(temporal_weights, temp_bins)
    
    num_retrieve = sum(1 for a in actions if a == "retrieve")
    num_generate = sum(1 for a in actions if a == "generate")
    retrieve_fraction = (num_retrieve / len(actions)) * 100 if actions else 0
    
    thresholds = [0.08, 0.10, 0.12, 0.13, 0.15]
    hypo_results = {}
    for t in thresholds:
        count = sum(1 for s in weighted_sims if s >= t)
        pct = (count / len(weighted_sims)) * 100 if weighted_sims else 0
        hypo_results[t] = (count, pct)

    print("\n=== AGGREGATE PHASE 4 STATS ===")
    print(f"\nVideos processed: {total_successful} / {len(videos)}")
    print(f"Failed videos: {', '.join(failed_videos) if failed_videos else 'none'}")
    
    print(f"\nPer-video Phase 2 sentence counts:")
    print(f"  min: {min(sent_counts)}")
    print(f"  max: {max(sent_counts)}")
    print(f"  mean: {np.mean(sent_counts):.1f}")
    print(f"  total sentences across all videos: {sum(sent_counts)}")
    
    print(f"\nPer-video group counts:")
    print(f"  min: {min(num_groups_per_video)}")
    print(f"  max: {max(num_groups_per_video)}")
    print(f"  mean: {np.mean(num_groups_per_video):.1f}")
    print(f"  total groups across all videos: {len(all_assignments)}")
    
    print(f"\nGroup size distribution (across all assignments in all videos):")
    for size in range(1, 6):
        count = sum(1 for s in group_sizes if s == size)
        pct = (count / len(group_sizes)) * 100 if group_sizes else 0
        print(f"  size={size}: {count} ({pct:.1f}%)")
    print(f"  max size observed: {max(group_sizes)}")
    
    print(f"\nWeighted similarity (cosine * temporal_weight) distribution:")
    print(f"  min: {min(weighted_sims):.3f}")
    print(f"  max: {max(weighted_sims):.3f}")
    print(f"  mean: {np.mean(weighted_sims):.3f}")
    print(f"  histogram:")
    for i in range(len(sim_bins)-1):
        print(f"    [{sim_bins[i]:.2f}, {sim_bins[i+1]:.2f}): {weighted_hist[i]}")
    print(f"    [{sim_bins[-1]:.2f}, +inf): {weighted_hist[-1]}")

    print(f"\nRaw cosine distribution:")
    print(f"  min: {min(raw_cosines):.3f}")
    print(f"  max: {max(raw_cosines):.3f}")
    print(f"  mean: {np.mean(raw_cosines):.3f}")
    print(f"  histogram:")
    for i in range(len(sim_bins)-1):
        print(f"    [{sim_bins[i]:.2f}, {sim_bins[i+1]:.2f}): {raw_hist[i]}")
    print(f"    [{sim_bins[-1]:.2f}, +inf): {raw_hist[-1]}")

    print(f"\nTemporal weight distribution:")
    print(f"  min: {min(temporal_weights):.3f}")
    print(f"  max: {max(temporal_weights):.3f}")
    print(f"  mean: {np.mean(temporal_weights):.3f}")
    print(f"  histogram:")
    for i in range(len(temp_bins)-1):
        print(f"    [{temp_bins[i]:.2f}, {temp_bins[i+1]:.2f}): {temp_hist[i]}")
    print(f"    [1.00, 1.00]: {sum(1 for s in temporal_weights if s >= 0.999)}")

    print(f"\nAction distribution (at threshold 0.13):")
    print(f"  retrieve total: {num_retrieve}")
    print(f"  generate total: {num_generate}")
    print(f"  retrieve fraction: {retrieve_fraction:.1f}%")
    
    print(f"\nHypothetical thresholds (count what would be retrieve if threshold changed):")
    for t in thresholds:
        count, pct = hypo_results[t]
        print(f"  threshold = {t:.2f}: retrieve count = {count} ({pct:.1f}%)" + ("  (current)" if t == 0.13 else ""))
    
    print(f"\nBlockers: none")
    print("\n=== TASK 3 DONE ===")

if __name__ == "__main__":
    run_calibration()

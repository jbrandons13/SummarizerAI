import os
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import cv2
from scenedetect import SceneManager, open_video, ContentDetector
from scenedetect.frame_timecode import FrameTimecode

from src.schemas import (
    KeyframeScene, KeyframesManifest, RetrievalOutput, 
    SceneMatch, AlternativeMatch, SummaryScript
)
from src.utils.vram import VRAMManager
from src.utils.ffmpeg_ops import extract_frame_at
from src.exceptions import FFmpegError

logger = logging.getLogger(__name__)

class KeyframeExtractor:
    """Detects scenes in a video and extracts representative keyframes."""
    
    def __init__(self, threshold: float = 27.0, min_scenes: int = 5):
        self.threshold = threshold
        self.min_scenes = min_scenes

    def extract(self, video_path: Path, output_dir: Path, progress_callback: Any = None) -> KeyframesManifest:
        """
        Detect scenes and extract 1 keyframe per scene at the midpoint.
        If scenes < min_scenes, supplement with uniform sampling.
        """
        video_id = video_path.stem
        keyframes_dir = output_dir / "keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Detect scenes
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        
        scenes_data: List[KeyframeScene] = []
        
        # 2. Convert detected scenes to KeyframeScene objects
        for i, (start, end) in enumerate(scene_list):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()
            mid_sec = (start_sec + end_sec) / 2.0
            
            scenes_data.append(KeyframeScene(
                id=i,
                start_seconds=start_sec,
                end_seconds=end_sec,
                keyframe_path="", # To be filled
                keyframe_timestamp=mid_sec
            ))
            
        # 3. Fallback: Supplement if too few scenes
        duration = video.duration.get_seconds()
        if len(scenes_data) < self.min_scenes:
            logger.info(f"Detected only {len(scenes_data)} scenes. Supplementing with uniform sampling.")
            interval = 5.0 # seconds
            current = 0.0
            while current < duration:
                # Check if this timestamp is already covered by a scene midpoint (roughly)
                is_covered = any(abs(s.keyframe_timestamp - (current + 2.5)) < 2.5 for s in scenes_data)
                if not is_covered:
                    start_s = current
                    end_s = min(current + interval, duration)
                    mid_s = (start_s + end_s) / 2.0
                    scenes_data.append(KeyframeScene(
                        id=len(scenes_data),
                        start_seconds=start_s,
                        end_seconds=end_s,
                        keyframe_path="",
                        keyframe_timestamp=mid_s
                    ))
                current += interval
                
        # Sort scenes by timestamp
        scenes_data.sort(key=lambda x: x.start_seconds)
        # Re-assign IDs
        for i, s in enumerate(scenes_data):
            s.id = i
            
        # 4. Extract keyframes
        for s in scenes_data:
            kf_name = f"scene_{s.id:03d}.jpg"
            kf_path = keyframes_dir / kf_name
            # Relative path for manifest
            s.keyframe_path = f"keyframes/{kf_name}"
            
            try:
                self._extract_with_quality(video_path, s.keyframe_timestamp, kf_path, quality=90)
                if progress_callback:
                    pct = int(10 + (s.id / len(scenes_data)) * 20)
                    progress_callback.update(4, "Visual Retrieval", pct, f"Extracted keyframe {s.id+1}/{len(scenes_data)}")
            except Exception as e:
                logger.error(f"Failed to extract keyframe for scene {s.id}: {e}")
                
        manifest = KeyframesManifest(video_id=video_id, scenes=scenes_data)
        
        # Save manifest
        manifest_path = output_dir / "keyframes_manifest.json"
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))
            
        return manifest

    def _extract_with_quality(self, video_path: Path, timestamp: float, out_path: Path, quality: int = 90):
        """Extract frame using ffmpeg with specific quality."""
        import subprocess
        # ffmpeg -ss {ts} -i {video} -vframes 1 -q:v {scale} {out}
        # q:v 2 is roughly 90-95% quality. q:v 1 is best.
        # scale = (100 - quality) / 10 is a rough mapping if quality is 1-31.
        # But usually q:v 2 is standard for high quality.
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(out_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)

class RetrievalBackend(ABC):
    """Abstract base class for retrieval arms."""
    
    def __init__(self, config: Dict[str, Any], vram_manager: Optional[VRAMManager] = None):
        self.config = config
        self.vram_manager = vram_manager

    @abstractmethod
    def retrieve(self, summary: SummaryScript, manifest: KeyframesManifest, progress_callback: Any = None) -> RetrievalOutput:
        """Match sentences to scenes."""
        pass

    def greedy_match(self, all_pairs: List[Tuple[int, int, float]], num_sentences: int, num_scenes: int, allow_reuse: bool = False) -> List[SceneMatch]:
        """
        Greedy matching with optional no-reuse constraint.
        all_pairs: List of (sentence_id, scene_id, score)
        """
        sorted_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)
        used_scenes = set()
        matches: Dict[int, SceneMatch] = {}
        
        # Track alternatives for each sentence
        sentence_alternatives: Dict[int, List[AlternativeMatch]] = {i: [] for i in range(num_sentences)}
        for sent_id, scene_id, score in sorted_pairs:
            if len(sentence_alternatives[sent_id]) < 5:
                sentence_alternatives[sent_id].append(AlternativeMatch(scene_id=scene_id, score=score))

        for sent_id, scene_id, score in sorted_pairs:
            if sent_id in matches:
                continue
            
            # If no-reuse and scene already used, skip UNLESS we have more sentences than scenes
            if not allow_reuse and scene_id in used_scenes:
                if num_scenes >= num_sentences:
                    continue
                # If fewer scenes than sentences, we MUST reuse eventually, 
                # but let's try to fill unique ones first or just allow reuse.
                # The user says "unless fewer scenes than sentences".
            
            matches[sent_id] = SceneMatch(
                sentence_id=sent_id,
                matched_scene_id=scene_id,
                score=score,
                alternatives=sentence_alternatives[sent_id]
            )
            used_scenes.add(scene_id)
            
        # Ensure all sentences have a match (fallback if somehow missed)
        if len(matches) < num_sentences:
            for i in range(num_sentences):
                if i not in matches:
                    # Pick highest score available for this sentence
                    if sentence_alternatives[i]:
                        best = sentence_alternatives[i][0]
                        matches[i] = SceneMatch(
                            sentence_id=i,
                            matched_scene_id=best.scene_id,
                            score=best.score,
                            alternatives=sentence_alternatives[i]
                        )
        
        return [matches[i] for i in range(num_sentences)]

class RandomRetrieval(RetrievalBackend):
    """Arm A: Random baseline."""
    
    def retrieve(self, summary: SummaryScript, manifest: KeyframesManifest, progress_callback: Any = None) -> RetrievalOutput:
        # Seed RNG from video_id for determinism
        random.seed(manifest.video_id)
        
        scenes = manifest.scenes
        num_sentences = len(summary.sentences)
        num_scenes = len(scenes)
        
        scene_ids = [s.id for s in scenes]
        random.shuffle(scene_ids)
        
        matches: List[SceneMatch] = []
        for i in range(num_sentences):
            # If we run out of unique scenes, we have to reuse (or just wrap around)
            chosen_scene_id = scene_ids[i % num_scenes]
            matches.append(SceneMatch(
                sentence_id=i,
                matched_scene_id=chosen_scene_id,
                score=0.0,
                alternatives=[]
            ))
            
        return RetrievalOutput(
            video_id=summary.video_id,
            retrieval_method="random",
            matches=matches
        )

class SigLIP2DirectRetrieval(RetrievalBackend):
    """Arm C: SigLIP 2 direct text-image retrieval."""
    
    def retrieve(self, summary: SummaryScript, manifest: KeyframesManifest, use_timestamp_hint: bool = True, progress_callback: Any = None) -> RetrievalOutput:
        from transformers import AutoProcessor, Siglip2Model
        from PIL import Image
        
        model_name = "google/siglip2-so400m-patch16-naflex"
        
        if progress_callback:
            progress_callback.update(4, "Visual Retrieval", 30, "Preparing SigLIP 2 engine...")

        def loader():
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = Siglip2Model.from_pretrained(
                model_name, ignore_mismatched_sizes=True
            ).to("cuda")
            return model, processor

        model, processor = self.vram_manager.load_model(f"SigLIP2 ({model_name})", loader)
        
        if progress_callback:
            progress_callback.update(4, "Visual Retrieval", 45, "Encoding summary script...")

        num_sentences = len(summary.sentences)
        num_scenes = len(manifest.scenes)
        
        # 1. Encode text
        texts = [s.text for s in summary.sentences]
        # SigLIP limit 64 tokens
        text_inputs = processor(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            if not isinstance(text_features, torch.Tensor):
                # Try to get the tensor output if it's a BaseModelOutput
                text_features = getattr(text_features, "pooler_output", text_features[0])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        # 2. Encode images
        all_pairs = []
        image_paths = [Path("data/intermediate") / manifest.video_id / s.keyframe_path for s in manifest.scenes]
        
        for scene_idx, img_path in enumerate(image_paths):
            image = Image.open(img_path).convert("RGB")
            img_inputs = processor(images=image, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                image_features = model.get_image_features(**img_inputs)
                if not isinstance(image_features, torch.Tensor):
                    image_features = getattr(image_features, "pooler_output", image_features[0])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity between all texts and this image
                similarities = (text_features @ image_features.T).squeeze(-1).cpu().tolist()
                
                if not isinstance(similarities, list):
                    similarities = [similarities]
                
            if progress_callback:
                pct = int(10 + (scene_idx / len(image_paths)) * 80)
                progress_callback.update(4, "Visual Retrieval", pct, f"SigLIP processing scene {scene_idx+1}/{len(image_paths)}")

            for sent_idx, score in enumerate(similarities):
                # Apply temporal preference bonus
                final_score = score
                if use_timestamp_hint:
                    hint = summary.sentences[sent_idx].source_timestamp_hint
                    if hint and len(hint) >= 2:
                        start_hint, end_hint = hint
                        scene = manifest.scenes[scene_idx]
                        # If scene midpoint is within hint window (with 2s buffer)
                        if (start_hint - 2.0) <= scene.keyframe_timestamp <= (end_hint + 2.0):
                            final_score += 0.1
                            
                all_pairs.append((sent_idx, scene_idx, final_score))
                
        # Unload model
        self.vram_manager.load_model("None (Cleanup)", lambda: None)
        
        matches = self.greedy_match(all_pairs, num_sentences, num_scenes, allow_reuse=(num_scenes < num_sentences))
        
        return RetrievalOutput(
            video_id=summary.video_id,
            retrieval_method="siglip_direct",
            matches=matches
        )

class CaptionCosineRetrieval(RetrievalBackend):
    """Arm B: Caption + Cosine similarity (Qwen2.5-VL + SentenceTransformer)."""
    
    def retrieve(self, summary: SummaryScript, manifest: KeyframesManifest, language: str = "en", progress_callback: Any = None) -> RetrievalOutput:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        from sentence_transformers import SentenceTransformer, util
        from PIL import Image
        
        video_dir = Path("data/intermediate") / manifest.video_id
        cache_path = video_dir / "keyframes_captions.json"
        
        # 1. Load or Generate Captions
        captions: Dict[str, str] = {}
        if cache_path.exists():
            with open(cache_path, "r") as f:
                captions = json.load(f)
                
        # Fill missing captions
        missing_scenes = [s for s in manifest.scenes if str(s.id) not in captions]
        
        if missing_scenes:
            # Use model from config or fallback to 3B
            model_name = self.config.get("models", {}).get("qwen_vl", {}).get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
            def loader():
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                # For AWQ models, we still use the standard loader but ensure it doesn't 
                # trigger the gptqmodel error by passing basic parameters or using AutoAWQ
                if "AWQ" in model_name:
                    from awq import AutoAWQForCausalLM
                    # We load as CausalLM but we must be careful with vision features
                    model = AutoAWQForCausalLM.from_quantized(
                        model_name, fuse_layers=False, 
                        trust_remote_code=True, device_map="auto"
                    )
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype="auto", device_map="auto", 
                        trust_remote_code=True
                    )
                processor = AutoProcessor.from_pretrained(model_name)
                return model, processor
            
            model, processor = self.vram_manager.load_model(f"Qwen2.5-VL ({model_name})", loader)
            
            prompt = "Describe what is happening in this video frame in one concise sentence (max 20 words). Focus on: people, objects, actions, setting. Do not speculate."
            
            for scene in missing_scenes:
                img_path = video_dir / scene.keyframe_path
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
                
                generated_ids = model.generate(**inputs, max_new_tokens=50)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                captions[str(scene.id)] = output_text
                
                if progress_callback:
                    pct = int(10 + (scene.id / len(missing_scenes)) * 70)
                    progress_callback.update(4, "Visual Retrieval", pct, f"Qwen captioning scene {scene.id+1}/{len(missing_scenes)}")
                
            # Unload Qwen
            self.vram_manager.load_model("None (Cleanup)", lambda: None)
            
            # Save cache
            with open(cache_path, "w") as f:
                json.dump(captions, f, indent=2)
                
        # 2. Embedding & Matching
        st_model_name = "sentence-transformers/all-MiniLM-L12-v2"
        if language.lower() in ["id", "indonesian"]:
            st_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            
        st_model = SentenceTransformer(st_model_name)
        
        sentence_texts = [s.text for s in summary.sentences]
        scene_captions = [captions[str(s.id)] for s in manifest.scenes]
        
        sent_embeddings = st_model.encode(sentence_texts, convert_to_tensor=True)
        cap_embeddings = st_model.encode(scene_captions, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(sent_embeddings, cap_embeddings)
        
        all_pairs = []
        num_sentences = len(summary.sentences)
        num_scenes = len(manifest.scenes)
        
        for i in range(num_sentences):
            for j in range(num_scenes):
                score = float(cosine_scores[i][j])
                # Temporal bonus
                hint = summary.sentences[i].source_timestamp_hint
                if hint and len(hint) >= 2:
                    start_hint, end_hint = hint
                    scene = manifest.scenes[j]
                    if (start_hint - 2.0) <= scene.keyframe_timestamp <= (end_hint + 2.0):
                        score += 0.1
                all_pairs.append((i, j, score))
                
        matches = self.greedy_match(all_pairs, num_sentences, num_scenes, allow_reuse=(num_scenes < num_sentences))
        
        return RetrievalOutput(
            video_id=summary.video_id,
            retrieval_method="caption_cosine",
            matches=matches
        )

class Phase4Retrieval:
    """Orchestrator for Phase 4: Semantic Visual Retrieval."""
    
    def __init__(self, config: Dict[str, Any], vram_manager: VRAMManager):
        self.config = config
        self.vram_manager = vram_manager
        self.extractor = KeyframeExtractor()
        
    def run(self, video_path: Path, summary: SummaryScript, language: str = "en", method: str = "siglip_direct", progress_callback: Any = None) -> Dict[str, RetrievalOutput]:
        video_id = video_path.stem
        output_dir = Path("data/intermediate") / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Keyframe Extraction
        manifest_path = output_dir / "keyframes_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = KeyframesManifest.model_validate_json(f.read())
            if progress_callback:
                progress_callback.update(4, "Visual Retrieval", 10, "Loaded existing keyframes")
        else:
            if progress_callback:
                progress_callback.update(4, "Visual Retrieval", 5, "Starting scene detection")
            manifest = self.extractor.extract(video_path, output_dir, progress_callback=progress_callback)
            
        # 2. Retrieval Arms
        results = {}
        
        # Based on method, run specific arm or all
        arms_to_run = [method] if method != "all" else ["random", "siglip_direct", "caption_cosine"]
        
        total_arms = len(arms_to_run)
        for i, arm_name in enumerate(arms_to_run):
            if progress_callback:
                progress_callback.update(4, "Visual Retrieval", 40 + int((i/total_arms)*50), f"Running retrieval arm: {arm_name}")
                
            if arm_name == "random":
                arm = RandomRetrieval(self.config, self.vram_manager)
                results["random"] = arm.retrieve(summary, manifest)
            elif arm_name == "siglip_direct":
                arm = SigLIP2DirectRetrieval(self.config, self.vram_manager)
                results["siglip_direct"] = arm.retrieve(summary, manifest)
            elif arm_name == "caption_cosine":
                arm = CaptionCosineRetrieval(self.config, self.vram_manager)
                results["caption_cosine"] = arm.retrieve(summary, manifest, language=language)

        # Save results
        for m, output in results.items():
            out_file = output_dir / f"scene_matches_{m}.json"
            with open(out_file, "w") as f:
                f.write(output.model_dump_json(indent=2))
                
        if progress_callback:
            progress_callback.update(4, "Visual Retrieval", 100, "Phase 4 complete")
                
        return results

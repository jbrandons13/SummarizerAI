import os
print("DEBUG: PHASE 4 MODULE LOADED")
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence, Protocol, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

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
import joblib
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

def min_max_normalize(scores):
    """Normalize array to [0, 1] range."""
    if isinstance(scores, (list, tuple)):
        scores = np.array(scores)
    elif torch.is_tensor(scores):
        scores = scores.cpu().numpy()
        
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-6:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min)

def compute_temporal_scores(sentence_timestamp_hint, keyframe_timestamps, sigma=30.0):
    """
    Compute temporal proximity scores between a sentence's source timestamp
    and all keyframe timestamps.
    
    Args:
        sentence_timestamp_hint: [start, end] from source_timestamp_hint
        keyframe_timestamps: list of floats, midpoint timestamp of each keyframe's scene
        sigma: controls how quickly score decays with distance (in seconds).
    
    Returns:
        numpy array of scores in [0, 1], one per keyframe
    """
    import math
    if sentence_timestamp_hint is None or len(sentence_timestamp_hint) < 2:
        # No timestamp info available, return uniform scores (no temporal bias)
        return np.ones(len(keyframe_timestamps)) / max(len(keyframe_timestamps), 1)
    
    mid = (sentence_timestamp_hint[0] + sentence_timestamp_hint[1]) / 2.0
    scores = np.array([
        math.exp(-((kf_ts - mid) ** 2) / (2 * sigma ** 2))
        for kf_ts in keyframe_timestamps
    ])
    return scores

class KeyframeExtractor:
    """Detects scenes in a video and extracts representative keyframes."""
    
    def __init__(self, threshold: float = 27.0, min_scenes: int = 5):
        self.threshold = threshold
        self.min_scenes = min_scenes

    def extract(self, video_path: Path, output_dir: Path, progress_callback: Any = None) -> KeyframesManifest:
        """
        Detect scenes and extract multi-frame keyframes per scene.
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
            
            scenes_data.append(KeyframeScene(
                id=i,
                start_seconds=start_sec,
                end_seconds=end_sec,
                keyframe_path="", # midpoint (filled for compat)
                keyframe_timestamp=(start_sec + end_sec) / 2.0
            ))
            
        # 3. Fallback: Supplement if too few scenes
        duration = video.duration.get_seconds()
        if len(scenes_data) < self.min_scenes:
            logger.info(f"Detected only {len(scenes_data)} scenes. Supplementing with uniform sampling.")
            interval = 5.0 # seconds
            current = 0.0
            while current < duration:
                is_covered = any(abs(s.keyframe_timestamp - (current + 2.5)) < 2.5 for s in scenes_data)
                if not is_covered:
                    start_s = current
                    end_s = min(current + interval, duration)
                    scenes_data.append(KeyframeScene(
                        id=len(scenes_data),
                        start_seconds=start_s,
                        end_seconds=end_s,
                        keyframe_path="",
                        keyframe_timestamp=(start_s + end_s) / 2.0
                    ))
                current += interval
                
        # Sort and re-assign IDs
        scenes_data.sort(key=lambda x: x.start_seconds)
        for i, s in enumerate(scenes_data):
            s.id = i
            
        # 4. Multi-frame sampling & extraction
        for s in scenes_data:
            start_sec, end_sec = s.start_seconds, s.end_seconds
            scene_duration = end_sec - start_sec
            
            if scene_duration < 1.5:
                # Short scene: 1 frame at midpoint
                frame_timestamps = [(start_sec + end_sec) / 2]
            else:
                # Roughly 1 frame per 2s, clamped to [3, 5]
                num_samples = min(5, max(3, int(scene_duration / 2)))
                frame_timestamps = np.linspace(
                    start_sec + 0.5, end_sec - 0.5, num_samples
                ).tolist()
            
            s.multi_frame_timestamps = frame_timestamps
            s.multi_frame_paths = []
            
            for j, ts in enumerate(frame_timestamps):
                kf_name = f"scene_{s.id:03d}_f{j:02d}.jpg"
                kf_path = keyframes_dir / kf_name
                rel_path = f"keyframes/{kf_name}"
                s.multi_frame_paths.append(rel_path)
                
                # Use first frame as fallback for keyframe_path (compat)
                if j == 0:
                    s.keyframe_path = rel_path
                
                try:
                    self._extract_with_quality(video_path, ts, kf_path, quality=90)
                except Exception as e:
                    logger.error(f"Failed to extract frame {j} for scene {s.id}: {e}")
            
            if progress_callback:
                pct = int(10 + (s.id / len(scenes_data)) * 20)
                progress_callback.update(4, "Visual Retrieval", pct, f"Extracted {len(s.multi_frame_paths)} frames for scene {s.id+1}/{len(scenes_data)}")
                
        manifest = KeyframesManifest(video_id=video_id, scenes=scenes_data)
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

    def apply_temporal_prior(
        self, 
        sim_matrix: np.ndarray, 
        summary: SummaryScript, 
        manifest: KeyframesManifest,
        use_temporal: bool = True,
        beta: float = 0.3,
        sigma: float = 30.0
    ) -> np.ndarray:
        """
        Normalize semantic scores and optionally apply temporal prior.
        """
        num_sentences, num_scenes = sim_matrix.shape
        kf_timestamps = [s.keyframe_timestamp for s in manifest.scenes]
        
        new_sim_matrix = np.zeros_like(sim_matrix)
        for i in range(num_sentences):
            semantic_scores = sim_matrix[i]
            if use_temporal:
                temporal_scores = compute_temporal_scores(
                    summary.sentences[i].source_timestamp_hint,
                    kf_timestamps,
                    sigma=sigma
                )
                new_sim_matrix[i] = (1 - beta) * min_max_normalize(semantic_scores) + beta * min_max_normalize(temporal_scores)
            else:
                new_sim_matrix[i] = min_max_normalize(semantic_scores)
        return new_sim_matrix

    def greedy_assign(self, sim_matrix: np.ndarray, allow_reuse: bool = True) -> List[int]:
        """
        Greedy assignment: per-sentence argmax.
        """
        num_sentences, num_scenes = sim_matrix.shape
        if allow_reuse:
            return np.argmax(sim_matrix, axis=1).tolist()
        else:
            # Simple greedy without reuse (prefer best global matches first)
            assignment = [-1] * num_sentences
            used_scenes = set()
            
            # (score, sent_idx, scene_idx)
            pairs = []
            for i in range(num_sentences):
                for j in range(num_scenes):
                    pairs.append((sim_matrix[i, j], i, j))
            
            pairs.sort(key=lambda x: x[0], reverse=True)
            
            for score, i, j in pairs:
                if assignment[i] == -1 and j not in used_scenes:
                    assignment[i] = j
                    used_scenes.add(j)
            
            # Fill remainders if we ran out of unique scenes
            for i in range(num_sentences):
                if assignment[i] == -1:
                    assignment[i] = int(np.argmax(sim_matrix[i]))
            
            return assignment

    def hungarian_align(self, sim_matrix: np.ndarray, reuse_penalty: float = 0.2) -> List[int]:
        print(f"DEBUG: Hungarian called! reuse_p={reuse_penalty}, sim shape={sim_matrix.shape}")
        """
        Global optimal assignment via Hungarian algorithm.
        """
        N, M = sim_matrix.shape
        K = max(3, (N // M) + 1)
        cost_matrix = np.tile(-sim_matrix, (1, K))
        for k in range(K):
            cost_matrix[:, k * M:(k + 1) * M] += k * reuse_penalty

        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        assignment = [int(col_idx[i] % M) for i in range(N)]
        return assignment

    def dp_sequence_align(
        self,
        sim_matrix: np.ndarray,
        scenes: List[KeyframeScene],
        video_duration: float,
        jump_penalty: float = 0.3,
        reuse_bonus: float = 0.3,
        backward_penalty: float = 0.5,
    ) -> List[int]:
        """
        Viterbi-style DP for sentence-to-scene assignment with transition costs.
        """
        N, M = sim_matrix.shape
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1e9)

        if N == 1:
            return [int(np.argmax(sim_matrix[0]))]

        scene_time = np.array([s.keyframe_timestamp for s in scenes])
        dt_matrix = (scene_time[None, :] - scene_time[:, None]) / max(video_duration, 1e-6)

        transition_matrix = np.where(
            dt_matrix >= 0,
            jump_penalty * dt_matrix,
            jump_penalty * np.abs(dt_matrix) + backward_penalty,
        )
        np.fill_diagonal(transition_matrix, -reuse_bonus)

        dp = np.full((N, M), -np.inf)
        backptr = np.full((N, M), -1, dtype=int)
        dp[0] = sim_matrix[0]

        for i in range(1, N):
            candidates = dp[i - 1][:, None] - transition_matrix
            dp[i] = sim_matrix[i] + candidates.max(axis=0)
            backptr[i] = candidates.argmax(axis=0)

        assignment = [0] * N
        assignment[N - 1] = int(np.argmax(dp[N - 1]))
        for i in range(N - 2, -1, -1):
            assignment[i] = int(backptr[i + 1][assignment[i + 1]])

        return assignment

    def cv_align_sequence(
        self,
        sim_matrix: np.ndarray,
        scenes: List[KeyframeScene],
        video_duration: float,
        jump_penalty: float = 0.01,
        reuse_bonus: float = 0.01,
        backward_penalty: float = 0.5,
        k_max: int = 3,
        lam: float = 0.1,
    ) -> List[int]:
        """
        Constrained Viterbi Alignment (CV-Align).
        
        Augments vanilla DP with a consecutive-reuse counter as part of state.
        Hard cap K_max on consecutive same-scene assignments.
        Soft penalty lambda grows with reuse count.
        
        State: (sentence i, scene j, reuse_count r) where r in [0, k_max - 1].
        
        Args:
            sim_matrix: (N, M) numpy array of similarity scores
            scenes: list of KeyframeScene objects (for temporal info)
            video_duration: total video length in seconds
            jump_penalty: cost coefficient for forward/backward jumps (jp)
            reuse_bonus: bonus for staying on same scene (rb)
            backward_penalty: additional cost for backward jumps (bp)
            k_max: hard cap on consecutive reuse (must be >= 1)
            lam: soft penalty multiplier for reuse count
        
        Returns:
            List[int] of length N, where each entry is the assigned scene index.
        """
        N, M = sim_matrix.shape
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1e9)

        if N == 1:
            return [int(np.argmax(sim_matrix[0]))]
        
        if k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {k_max}")

        scene_time = np.array([s.keyframe_timestamp for s in scenes])
        
        # DP table: shape (N, M, K_max)
        # DP[i][j][r] = max score reaching state (i, j, r)
        dp = np.full((N, M, k_max), -np.inf)
        
        # Backpointers: store (prev_j, prev_r) for each state
        bp_j = np.full((N, M, k_max), -1, dtype=int)
        bp_r = np.full((N, M, k_max), -1, dtype=int)
        
        # Initialization: i = 0, only r = 0 is valid
        dp[0, :, 0] = sim_matrix[0]
        # All other r values for i = 0 remain -inf
        
        # Fill DP
        for i in range(1, N):
            valid_r_max = min(i + 1, k_max)
            valid_r_prev_max = min(i, k_max)
            
            for j in range(M):  # current scene
                for r in range(valid_r_max):  # current reuse count
                    # Find best (j', r') -> (j, r) transition
                    best_score = -np.inf
                    best_j_prev = -1
                    best_r_prev = -1
                    
                    if r > 0:
                        # Must be a stay transition: j_prev == j, r_prev == r - 1
                        j_prev = j
                        r_prev = r - 1
                        if dp[i-1, j_prev, r_prev] > -np.inf:
                            cost = -reuse_bonus + lam * r_prev
                            best_score = dp[i-1, j_prev, r_prev] - cost
                            best_j_prev = j_prev
                            best_r_prev = r_prev
                    else:
                        # Must be a jump transition: j_prev != j, r_prev can be anything
                        for j_prev in range(M):
                            if j_prev == j:
                                continue
                            dt = (scene_time[j] - scene_time[j_prev]) / max(video_duration, 1e-6)
                            if dt >= 0:
                                cost = jump_penalty * dt
                            else:
                                cost = jump_penalty * abs(dt) + backward_penalty
                                
                            for r_prev in range(valid_r_prev_max):
                                if dp[i-1, j_prev, r_prev] > -np.inf:
                                    score = dp[i-1, j_prev, r_prev] - cost
                                    if score > best_score:
                                        best_score = score
                                        best_j_prev = j_prev
                                        best_r_prev = r_prev
                    
                    if best_score > -np.inf:
                        dp[i, j, r] = sim_matrix[i, j] + best_score
                        bp_j[i, j, r] = best_j_prev
                        bp_r[i, j, r] = best_r_prev
        
        # Find best terminal state at i = N-1
        flat_idx = np.argmax(dp[N-1])
        final_j, final_r = np.unravel_index(flat_idx, (M, k_max))
        
        # Backtrack
        assignment = [0] * N
        assignment[N-1] = int(final_j)
        cur_j, cur_r = int(final_j), int(final_r)
        
        for i in range(N-1, 0, -1):
            prev_j = bp_j[i, cur_j, cur_r]
            prev_r = bp_r[i, cur_j, cur_r]
            assignment[i-1] = int(prev_j)
            cur_j, cur_r = int(prev_j), int(prev_r)
        
        # Verify constraint satisfaction (sanity check)
        max_consec = 1
        cur_consec = 1
        for i in range(1, N):
            if assignment[i] == assignment[i-1]:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 1
        assert max_consec <= k_max, f"CV-Align constraint violated: max_consec={max_consec}, k_max={k_max}"
        
        return assignment

    def ccma_align_sequence(
        self,
        sim_matrix: np.ndarray,
        scenes: List[KeyframeScene],
        video_duration: float,
        c_max: int = 3,
        reuse_penalty: float = 0.2,
        jump_penalty: float = 0.01,
        backward_penalty: float = 0.5,
    ) -> List[int]:
        """
        Capacity-Constrained Monotonic Alignment (CCMA).
        
        A 3D DP algorithm that explicitly bounds the consecutive reuse of any scene
        to mathematically eliminate the 'scene-attractor' failure mode.
        """
        N, M = sim_matrix.shape
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1e9)

        if N == 1:
            return [int(np.argmax(sim_matrix[0]))]
            
        if c_max < 1:
            raise ValueError(f"c_max must be >= 1, got {c_max}")

        scene_time = np.array([s.keyframe_timestamp for s in scenes])
        
        dp = np.full((N, M, c_max), -np.inf)
        bp_j = np.full((N, M, c_max), -1, dtype=int)
        bp_c = np.full((N, M, c_max), -1, dtype=int)
        
        dp[0, :, 0] = sim_matrix[0]
        
        for i in range(1, N):
            prev_best_c = np.max(dp[i-1], axis=1)
            prev_best_c_idx = np.argmax(dp[i-1], axis=1)
            
            for j in range(M):
                best_score = -np.inf
                best_j_prev = -1
                best_c_prev = -1
                
                for j_prev in range(M):
                    if j_prev == j:
                        continue
                    
                    dt = (scene_time[j] - scene_time[j_prev]) / max(video_duration, 1e-6)
                    if dt >= 0:
                        cost = jump_penalty * dt
                    else:
                        cost = jump_penalty * abs(dt) + backward_penalty
                        
                    score = prev_best_c[j_prev] - cost
                    if score > best_score:
                        best_score = score
                        best_j_prev = j_prev
                        best_c_prev = prev_best_c_idx[j_prev]
                        
                if best_score > -np.inf:
                    dp[i, j, 0] = sim_matrix[i, j] + best_score
                    bp_j[i, j, 0] = best_j_prev
                    bp_c[i, j, 0] = best_c_prev
                    
                for c_idx in range(1, min(i + 1, c_max)):
                    prev_score = dp[i-1, j, c_idx-1]
                    if prev_score > -np.inf:
                        dp[i, j, c_idx] = sim_matrix[i, j] + prev_score - reuse_penalty
                        bp_j[i, j, c_idx] = j
                        bp_c[i, j, c_idx] = c_idx - 1

        flat_idx = np.argmax(dp[N-1])
        final_j, final_c = np.unravel_index(flat_idx, (M, c_max))
        
        assignment = [0] * N
        assignment[N-1] = int(final_j)
        
        cur_j, cur_c = int(final_j), int(final_c)
        for i in range(N-1, 0, -1):
            prev_j = bp_j[i, cur_j, cur_c]
            prev_c = bp_c[i, cur_j, cur_c]
            assignment[i-1] = int(prev_j)
            cur_j, cur_c = int(prev_j), int(prev_c)
            
        return assignment



    def compute_path_score(self, sim_matrix, assignment, transition_matrix=None):
        """Helper for sanity checks."""
        s = sim_matrix[0, assignment[0]]
        if transition_matrix is not None:
            for i in range(1, len(assignment)):
                s += sim_matrix[i, assignment[i]]
                s -= transition_matrix[assignment[i - 1], assignment[i]]
        else:
            for i in range(1, len(assignment)):
                s += sim_matrix[i, assignment[i]]
        return s

class RandomRetrieval(RetrievalBackend):
    """Arm A: Random baseline."""
    
    def retrieve(self, summary: SummaryScript, manifest: KeyframesManifest, progress_callback: Any = None, method_name: str = None) -> RetrievalOutput:
        # Seed RNG from video_id for determinism
        import random
        random.seed(summary.video_id)
        
        scenes = manifest.scenes
        num_sentences = len(summary.sentences)
        num_scenes = len(scenes)
        
        scene_ids = [s.id for s in scenes]
        random.shuffle(scene_ids)
        
        matches: List[SceneMatch] = []
        for i in range(num_sentences):
            # If we run out of unique scenes, we have to reuse (or just wrap around)
            chosen_scene_id = scene_ids[i % num_scenes]
            scene = next(s for s in scenes if s.id == chosen_scene_id)
            matches.append(SceneMatch(
                sentence_id=i,
                matched_scene_id=chosen_scene_id,
                score=0.0,
                best_frame_path=scene.keyframe_path,
                best_frame_timestamp=scene.keyframe_timestamp,
                alternatives=[]
            ))
            
        return RetrievalOutput(
            video_id=summary.video_id,
            retrieval_method=method_name or "random",
            matches=matches
        )

class SigLIP2DirectRetrieval(RetrievalBackend):
    """Arm C: SigLIP 2 direct text-image retrieval."""
    
    def retrieve(self, summary: SummaryScript, manifest: KeyframesManifest, use_timestamp_hint: bool = True, progress_callback: Any = None, method_name: str = None) -> RetrievalOutput:
        from transformers import AutoProcessor, Siglip2Model
        from PIL import Image
        
        model_name = "google/siglip2-so400m-patch16-naflex"
        model_slug = model_name.replace("/", "_").replace("-", "_")
        video_dir = Path("data/intermediate") / manifest.video_id
        cache_path = video_dir / f"embeddings_{model_slug}.joblib"
        
        num_sentences = len(summary.sentences)
        num_scenes = len(manifest.scenes)
        
        # 1. Load or Compute Embeddings
        # frame_embeddings: Dict[Tuple[scene_id: int, frame_timestamp: float], np.ndarray]
        frame_embeddings = {}
        if cache_path.exists():
            frame_embeddings = joblib.load(cache_path)
            logger.info(f"Loaded SigLIP embeddings from cache: {cache_path}")
        else:
            if progress_callback:
                progress_callback.update(4, "Visual Retrieval", 30, "Preparing SigLIP 2 engine...")

            def loader():
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                model = Siglip2Model.from_pretrained(
                    model_name, ignore_mismatched_sizes=True
                ).to("cuda")
                return model, processor

            model, processor = self.vram_manager.load_model(f"SigLIP2 ({model_name})", loader)
            
            # Encode all frames
            all_frames = []
            for scene in manifest.scenes:
                for path, ts in zip(scene.multi_frame_paths, scene.multi_frame_timestamps):
                    all_frames.append((scene.id, ts, video_dir / path))
            
            for i, (scene_id, ts, img_path) in enumerate(all_frames):
                image = Image.open(img_path).convert("RGB")
                img_inputs = processor(images=image, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    image_features = model.get_image_features(**img_inputs)
                    if not isinstance(image_features, torch.Tensor):
                        image_features = getattr(image_features, "pooler_output", image_features[0])
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    frame_embeddings[(scene_id, ts)] = image_features.squeeze(0).cpu().numpy()
                
                if progress_callback and i % 5 == 0:
                    pct = int(30 + (i / len(all_frames)) * 50)
                    progress_callback.update(4, "Visual Retrieval", pct, f"SigLIP encoding frame {i+1}/{len(all_frames)}")

            # Save cache
            joblib.dump(frame_embeddings, cache_path)
            # Unload model
            self.vram_manager.load_model("None (Cleanup)", lambda: None)

        # 2. Encode Text
        def text_loader():
            from transformers import AutoProcessor, Siglip2Model
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = Siglip2Model.from_pretrained(model_name, ignore_mismatched_sizes=True).to("cuda")
            return model, processor
            
        model, processor = self.vram_manager.load_model(f"SigLIP2 ({model_name})", text_loader)
        
        texts = [s.text for s in summary.sentences]
        text_inputs = processor(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(text_features, "pooler_output", text_features[0])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embs = text_features.cpu().numpy()
        
        self.vram_manager.load_model("None (Cleanup)", lambda: None)

        # 3. Scoring with top-k mean pooling
        # sim_matrix[i, j] is score of sentence i with scene j
        sim_matrix = np.zeros((num_sentences, num_scenes))
        # best_frames[(sent_idx, scene_idx)] = (path, ts)
        best_frames: Dict[Tuple[int, int], Tuple[str, float]] = {}
        
        k_pool = self.config.get("keyframe_extraction", {}).get("top_k", 2)
        
        for j, scene in enumerate(manifest.scenes):
            scene_frame_embs = [frame_embeddings[(scene.id, ts)] for ts in scene.multi_frame_timestamps]
            scene_frame_embs = np.stack(scene_frame_embs) # (num_frames, dim)
            
            # Scores for all frames in this scene against all sentences
            # text_embs: (num_sentences, dim), scene_frame_embs: (num_frames, dim)
            # frame_scores: (num_sentences, num_frames)
            frame_scores = text_embs @ scene_frame_embs.T
            
            for i in range(num_sentences):
                scores = frame_scores[i]
                k = min(k_pool, len(scores))
                top_k_indices = np.argsort(-scores)[:k]
                sim_matrix[i, j] = np.mean(scores[top_k_indices])
                
                # Winning frame for this (sentence, scene) pair
                best_idx = int(np.argmax(scores))
                best_frames[(i, j)] = (scene.multi_frame_paths[best_idx], scene.multi_frame_timestamps[best_idx])

        # 4. Apply Temporal Guidance
        ret_cfg = self.config.get("retrieval", {})
        use_temporal = ret_cfg.get("use_temporal_guidance", True) and use_timestamp_hint
        sim_matrix = self.apply_temporal_prior(
            sim_matrix, summary, manifest,
            use_temporal=use_temporal,
            beta=ret_cfg.get("temporal_weight", 0.3),
            sigma=ret_cfg.get("temporal_sigma", 30.0)
        )

        # 5. Matching Algorithm
        matching_algo = ret_cfg.get("matching_algorithm", "dp")
        if matching_algo == "greedy":
            assignment = self.greedy_assign(sim_matrix, allow_reuse=(num_scenes < num_sentences))
        elif matching_algo == "hungarian":
            reuse_p = ret_cfg.get("hungarian_reuse_penalty", 0.2)
            assignment = self.hungarian_align(sim_matrix, reuse_penalty=reuse_p)
        elif matching_algo == "dp":
            video_dur = max(s.end_seconds for s in manifest.scenes)
            jump_p = ret_cfg.get("dp_jump_penalty", 0.3)
            reuse_b = ret_cfg.get("dp_reuse_bonus", 0.3)
            back_p = ret_cfg.get("dp_backward_penalty", 0.5)
            assignment = self.dp_sequence_align(sim_matrix, manifest.scenes, video_dur, jump_penalty=jump_p, reuse_bonus=reuse_b, backward_penalty=back_p)
        elif matching_algo == "cv_align":
            video_dur = max(s.end_seconds for s in manifest.scenes)
            jump_p = ret_cfg.get("dp_jump_penalty", 0.01)
            reuse_b = ret_cfg.get("dp_reuse_bonus", 0.01)
            back_p = ret_cfg.get("dp_backward_penalty", 0.5)
            k_max = ret_cfg.get("cv_align_k_max", 3)
            lam = ret_cfg.get("cv_align_lambda", 0.1)
            assignment = self.cv_align_sequence(
                sim_matrix, manifest.scenes, video_dur,
                jump_penalty=jump_p, reuse_bonus=reuse_b, backward_penalty=back_p,
                k_max=k_max, lam=lam
            )
        elif matching_algo == "ccma":
            video_dur = max(s.end_seconds for s in manifest.scenes)
            c_max = ret_cfg.get("ccma_c_max", 3)
            reuse_p = ret_cfg.get("ccma_reuse_penalty", 0.2)
            jump_p = ret_cfg.get("dp_jump_penalty", 0.01)  # share with DP
            back_p = ret_cfg.get("dp_backward_penalty", 0.5)  # share with DP
            assignment = self.ccma_align_sequence(
                sim_matrix, manifest.scenes, video_dur,
                c_max=c_max, reuse_penalty=reuse_p, 
                jump_penalty=jump_p, backward_penalty=back_p
            )

        else:
            assignment = self.greedy_assign(sim_matrix)

        # 6. Convert to SceneMatch
        matches = []
        for i, scene_idx in enumerate(assignment):
            best_path, best_ts = best_frames.get((i, scene_idx), ("", 0.0))
            top_indices = np.argsort(-sim_matrix[i])[:5]
            alternatives = [
                AlternativeMatch(scene_id=int(idx), score=float(sim_matrix[i, idx]))
                for idx in top_indices
            ]
            matches.append(SceneMatch(
                sentence_id=i,
                matched_scene_id=int(scene_idx),
                score=float(sim_matrix[i, scene_idx]),
                best_frame_path=best_path,
                best_frame_timestamp=best_ts,
                alternatives=alternatives
            ))
            
        return RetrievalOutput(
            video_id=summary.video_id,
            retrieval_method=method_name or ("siglip_temporal" if use_timestamp_hint else "siglip_direct"),
            matches=matches
        )

class CaptionCosineRetrieval(RetrievalBackend):
    """Arm B: Caption + Cosine similarity (Qwen2.5-VL + SentenceTransformer)."""
    
    def retrieve(self, summary: SummaryScript, manifest: KeyframesManifest, language: str = "en", use_timestamp_hint: bool = True, progress_callback: Any = None, method_name: str = None) -> RetrievalOutput:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        from sentence_transformers import SentenceTransformer, util
        from PIL import Image
        
        video_dir = Path("data/intermediate") / manifest.video_id
        # Note: We keep the old filename for captions but now it stores multi-frame
        cache_path = video_dir / "keyframes_captions.json"
        
        num_sentences = len(summary.sentences)
        num_scenes = len(manifest.scenes)
        
        # 1. Load or Generate Captions
        # captions: Dict[str, str] where key is "scene_id_frame_ts"
        captions: Dict[str, str] = {}
        if cache_path.exists():
            with open(cache_path, "r") as f:
                captions = json.load(f)
                
        missing_frames = []
        cap_cfg = self.config.get("keyframe_extraction", {})
        max_frames_per_scene = cap_cfg.get("frames_per_scene_caption", 3)

        for scene in manifest.scenes:
            # Only use up to max_frames_per_scene for captioning to save cost
            frames_to_cap = scene.multi_frame_timestamps[:max_frames_per_scene]
            for ts in frames_to_cap:
                key = f"{scene.id}_{ts}"
                if key not in captions:
                    # Find path for this ts
                    idx = scene.multi_frame_timestamps.index(ts)
                    path = scene.multi_frame_paths[idx]
                    missing_frames.append((scene.id, ts, video_dir / path, key))
        
        if missing_frames:
            model_name = self.config.get("models", {}).get("qwen_vl", {}).get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
            def loader():
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                if "AWQ" in model_name:
                    from awq import AutoAWQForCausalLM
                    model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, trust_remote_code=True, device_map="auto")
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_name)
                return model, processor
            
            model, processor = self.vram_manager.load_model(f"Qwen2.5-VL ({model_name})", loader)
            prompt = "Describe what is happening in this video frame in one concise sentence (max 20 words). Focus on: people, objects, actions, setting."
            
            for i, (scene_id, ts, img_path, key) in enumerate(missing_frames):
                messages = [{"role": "user", "content": [{"type": "image", "image": str(img_path)}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
                
                generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=0.0)
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                captions[key] = output_text
                if progress_callback:
                    pct = int(10 + (i / len(missing_frames)) * 70)
                    progress_callback.update(4, "Visual Retrieval", pct, f"Qwen captioning frame {i+1}/{len(missing_frames)}")
            
            self.vram_manager.load_model("None (Cleanup)", lambda: None)
            with open(cache_path, "w") as f:
                json.dump(captions, f, indent=2)
                
        # 2. Embedding & Scoring
        st_model_name = "sentence-transformers/all-MiniLM-L12-v2"
        if language.lower() in ["id", "indonesian"]:
            st_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        st_model = SentenceTransformer(st_model_name)
        
        sentence_texts = [s.text for s in summary.sentences]
        sent_embs = st_model.encode(sentence_texts, convert_to_tensor=True)
        
        sim_matrix = np.zeros((num_sentences, num_scenes))
        best_frames: Dict[Tuple[int, int], Tuple[str, float]] = {}
        k_pool = cap_cfg.get("top_k", 2)

        # Pre-calculate caption embeddings per scene
        for j, scene in enumerate(manifest.scenes):
            scene_keys = [f"{scene.id}_{ts}" for ts in scene.multi_frame_timestamps[:max_frames_per_scene]]
            scene_caps = [captions[k] for k in scene_keys]
            cap_embs = st_model.encode(scene_caps, convert_to_tensor=True) # (num_frames, dim)
            
            # Scores: (num_sentences, num_frames)
            scores = util.cos_sim(sent_embs, cap_embs).cpu().numpy()
            
            for i in range(num_sentences):
                s = scores[i]
                k = min(k_pool, len(s))
                top_k_indices = np.argsort(-s)[:k]
                sim_matrix[i, j] = np.mean(s[top_k_indices])
                
                best_idx = int(np.argmax(s))
                best_frames[(i, j)] = (scene.multi_frame_paths[best_idx], scene.multi_frame_timestamps[best_idx])

        # 3. Apply Temporal Guidance
        ret_cfg = self.config.get("retrieval", {})
        use_temporal = ret_cfg.get("use_temporal_guidance", True) and use_timestamp_hint
        sim_matrix = self.apply_temporal_prior(
            sim_matrix, summary, manifest,
            use_temporal=use_temporal,
            beta=ret_cfg.get("temporal_weight", 0.3),
            sigma=ret_cfg.get("temporal_sigma", 30.0)
        )

        # 4. Matching Algorithm
        matching_algo = ret_cfg.get("matching_algorithm", "dp")
        if matching_algo == "greedy":
            assignment = self.greedy_assign(sim_matrix, allow_reuse=(num_scenes < num_sentences))
        elif matching_algo == "hungarian":
            reuse_p = ret_cfg.get("hungarian_reuse_penalty", 0.2)
            assignment = self.hungarian_align(sim_matrix, reuse_penalty=reuse_p)
        elif matching_algo == "dp":
            video_dur = max(s.end_seconds for s in manifest.scenes)
            jump_p = ret_cfg.get("dp_jump_penalty", 0.3)
            reuse_b = ret_cfg.get("dp_reuse_bonus", 0.3)
            back_p = ret_cfg.get("dp_backward_penalty", 0.5)
            assignment = self.dp_sequence_align(sim_matrix, manifest.scenes, video_dur, jump_penalty=jump_p, reuse_bonus=reuse_b, backward_penalty=back_p)
        elif matching_algo == "cv_align":
            video_dur = max(s.end_seconds for s in manifest.scenes)
            jump_p = ret_cfg.get("dp_jump_penalty", 0.01)
            reuse_b = ret_cfg.get("dp_reuse_bonus", 0.01)
            back_p = ret_cfg.get("dp_backward_penalty", 0.5)
            k_max = ret_cfg.get("cv_align_k_max", 3)
            lam = ret_cfg.get("cv_align_lambda", 0.1)
            assignment = self.cv_align_sequence(
                sim_matrix, manifest.scenes, video_dur,
                jump_penalty=jump_p, reuse_bonus=reuse_b, backward_penalty=back_p,
                k_max=k_max, lam=lam
            )
        elif matching_algo == "ccma":
            video_dur = max(s.end_seconds for s in manifest.scenes)
            c_max = ret_cfg.get("ccma_c_max", 3)
            reuse_p = ret_cfg.get("ccma_reuse_penalty", 0.2)
            jump_p = ret_cfg.get("dp_jump_penalty", 0.01)  # share with DP
            back_p = ret_cfg.get("dp_backward_penalty", 0.5)  # share with DP
            assignment = self.ccma_align_sequence(
                sim_matrix, manifest.scenes, video_dur,
                c_max=c_max, reuse_penalty=reuse_p, 
                jump_penalty=jump_p, backward_penalty=back_p
            )

        else:
            assignment = self.greedy_assign(sim_matrix)

        # 5. Convert to SceneMatch
        matches = []
        for i, scene_idx in enumerate(assignment):
            best_path, best_ts = best_frames.get((i, scene_idx), ("", 0.0))
            top_indices = np.argsort(-sim_matrix[i])[:5]
            alternatives = [AlternativeMatch(scene_id=int(idx), score=float(sim_matrix[i, idx])) for idx in top_indices]
            matches.append(SceneMatch(
                sentence_id=i, matched_scene_id=int(scene_idx), score=float(sim_matrix[i, scene_idx]),
                best_frame_path=best_path, best_frame_timestamp=best_ts, alternatives=alternatives
            ))
            
        return RetrievalOutput(
            video_id=summary.video_id,
            retrieval_method=method_name or ("caption_temporal" if use_timestamp_hint else "caption_cosine"),
            matches=matches
        )

class Phase4Retrieval:
    """Orchestrator for Phase 4: Semantic Visual Retrieval."""
    
    def __init__(self, config: Dict[str, Any], vram_manager: VRAMManager):
        self.config = config
        self.vram_manager = vram_manager
        self.extractor = KeyframeExtractor()
        
    def run(self, video_path: Path, summary: SummaryScript, language: str = "en", method: str = "siglip_temporal", progress_callback: Any = None) -> Dict[str, RetrievalOutput]:
        video_id = video_path.stem
        intermediate_root = self.config.get("paths", {}).get("intermediate_dir", "data/intermediate")
        output_dir = Path(intermediate_root) / video_id
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
        
        # Define the 6 ablation arms
        # format: (base_method, use_temporal, matching_algo)
        ARM_CONFIGS = {
            "random": ("random", False, "greedy"),
            "caption_direct": ("caption_temporal", False, "greedy"),
            "caption_temporal": ("caption_temporal", True, "greedy"),
            "caption_temporal_dp": ("caption_temporal", True, "dp"),
            "siglip_direct": ("siglip_temporal", False, "greedy"),
            "siglip_temporal": ("siglip_temporal", True, "greedy"),
            "siglip_temporal_hungarian": ("siglip_temporal", True, "hungarian"),
            "siglip_temporal_dp": ("siglip_temporal", True, "dp"),
            "caption_temporal_cvalign": ("caption_temporal", True, "cv_align"),
            "siglip_temporal_cvalign": ("siglip_temporal", True, "cv_align"),
            "caption_temporal_ccma": ("caption_temporal", True, "ccma"),
            "siglip_temporal_ccma": ("siglip_temporal", True, "ccma"),

        }

        if method == "all":
            arms_to_run = list(ARM_CONFIGS.keys())
        elif method in ARM_CONFIGS:
            arms_to_run = [method]
        else:
            # Fallback for old method names
            arms_to_run = [method]

        total_arms = len(arms_to_run)
        for i, arm_name in enumerate(arms_to_run):
            if progress_callback:
                progress_callback.update(4, "Visual Retrieval", 40 + int((i/total_arms)*50), f"Running retrieval arm: {arm_name}")
            
            # Temporary config override for this arm
            old_ret_cfg = self.config.get("retrieval", {}).copy()
            
            if arm_name in ARM_CONFIGS:
                base_method, use_temporal, matching = ARM_CONFIGS[arm_name]
                self.config.setdefault("retrieval", {})["use_temporal_guidance"] = use_temporal
                self.config["retrieval"]["matching_algorithm"] = matching
                
                if base_method == "random":
                    arm = RandomRetrieval(self.config, self.vram_manager)
                    results[arm_name] = arm.retrieve(summary, manifest, method_name=arm_name)
                elif "siglip" in base_method:
                    arm = SigLIP2DirectRetrieval(self.config, self.vram_manager)
                    results[arm_name] = arm.retrieve(summary, manifest, use_timestamp_hint=use_temporal, progress_callback=progress_callback, method_name=arm_name)
                elif "caption" in base_method:
                    arm = CaptionCosineRetrieval(self.config, self.vram_manager)
                    results[arm_name] = arm.retrieve(summary, manifest, language=language, use_timestamp_hint=use_temporal, progress_callback=progress_callback, method_name=arm_name)
            else:
                # Fallback for direct method calls
                if arm_name == "random":
                    arm = RandomRetrieval(self.config, self.vram_manager)
                    results[arm_name] = arm.retrieve(summary, manifest)
                elif "siglip" in arm_name:
                    use_t = "temporal" in arm_name
                    arm = SigLIP2DirectRetrieval(self.config, self.vram_manager)
                    results[arm_name] = arm.retrieve(summary, manifest, use_timestamp_hint=use_t, progress_callback=progress_callback)
                elif "caption" in arm_name:
                    use_t = "temporal" in arm_name
                    arm = CaptionCosineRetrieval(self.config, self.vram_manager)
                    results[arm_name] = arm.retrieve(summary, manifest, language=language, use_timestamp_hint=use_t, progress_callback=progress_callback)

            # Restore config
            self.config["retrieval"] = old_ret_cfg

        # Save results
        for m, output in results.items():
            out_file = output_dir / f"scene_matches_{m}.json"
            with open(out_file, "w") as f:
                f.write(output.model_dump_json(indent=2))
                
        if progress_callback:
            progress_callback.update(4, "Visual Retrieval", 100, "Phase 4 complete")
                
        return results


# ---------------------------------------------------------------------------
# Restored from stash@{0} (RetrievalGate & Helpers)
# ---------------------------------------------------------------------------


class TextEncoder(Protocol):
    """Minimal interface a text encoder must satisfy.

    The implementation is expected to share the same embedding space as the
    scene encoder used during preprocessing. For SigLIP this means using the
    SigLIP text tower with the same model id as the image tower that produced
    the scene embeddings.
    """

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class Sentence:
    """One narration sentence from Phase 2 output."""

    id: int
    text: str
    timestamp_hint: Tuple[float, float]


@dataclass
class Scene:
    """One scene from source video preprocessing.

    ``embedding`` must be in the same space as the text encoder output and is
    expected to be L2-normalised. If not normalised, cosine similarity is still
    computed correctly here but downstream metrics may behave differently.
    """

    id: int
    start: float
    end: float
    embedding: np.ndarray


@dataclass
class Assignment:
    """One group of sentences assigned to a single scene or to generation."""

    sentence_ids: List[int]
    scene_id: int
    best_similarity: float          # weighted similarity (cosine * temporal_weight)
    raw_cosine: float               # raw cosine before temporal weighting
    temporal_weight: float          # the weight applied to the locked scene
    action: str                     # "retrieve" or "generate"
    timestamp_hint_merged: Tuple[float, float]
    # Per-step weighted similarity trail, kept for debugging and threshold tuning.
    similarity_trail: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Safe against zero vectors."""

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cosine_to_all(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of ``query`` against every row of ``matrix``.

    ``matrix`` shape: ``(num_scenes, dim)``. Returns shape ``(num_scenes,)``.
    """

    q_norm = float(np.linalg.norm(query))
    if q_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    return (matrix @ query) / (row_norms * q_norm)


def _stack_scene_embeddings(scenes: Sequence[Scene]) -> np.ndarray:
    return np.stack([s.embedding for s in scenes], axis=0)


def _scene_centers(scenes: Sequence[Scene]) -> np.ndarray:
    """Time midpoint of each scene in seconds."""

    return np.array([(s.start + s.end) / 2.0 for s in scenes], dtype=np.float32)


def _gaussian_temporal_weights(
    scene_centers: np.ndarray,
    hint_center: float,
    sigma: float,
) -> np.ndarray:
    """Gaussian weight in [0, 1] for every scene given a hint center time.

    A scene whose centre matches ``hint_center`` gets weight 1.0; scenes that
    are ``sigma`` seconds away get weight ~0.61; ``2*sigma`` away ~0.14;
    ``3*sigma`` away ~0.01. The weight never reaches zero, so distant scenes
    can still be selected if their visual similarity is overwhelmingly strong,
    but they are heavily suppressed.
    """

    if sigma <= 0.0:
        # Degenerate: no temporal prior. Return all-ones.
        return np.ones_like(scene_centers, dtype=np.float32)
    delta = scene_centers - float(hint_center)
    return np.exp(-(delta ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


# ---------------------------------------------------------------------------
# Core: RetrievalGate
# ---------------------------------------------------------------------------


@dataclass
class RetrievalGateConfig:
    gate_threshold: float = 0.13       # tuned for SigLIP 2 raw cosine + temporal prior
    extend_epsilon: float = 0.03
    max_group_size: int = 5
    join_sep: str = " "
    temporal_sigma: float = 30.0       # seconds; controls Gaussian decay width
    enable_temporal_prior: bool = True
    enable_cascade_verification: bool = False  # SOTA Cascade Gating thesis innovation


class RetrievalGate:
    """Greedy forward-walk grouping with retrieval/generation gating.

    Each candidate group is scored against every scene as
    ``weighted_sim = cosine(text_emb, scene_emb) * gaussian_weight(scene_time, hint_time)``
    where ``gaussian_weight`` decays with the distance between the scene's
    centre time and the centre of the merged ``source_timestamp_hint`` of the
    current group. The decision gate compares the final weighted similarity
    of the locked scene to ``gate_threshold``.

    Algorithm summary:
      i = 0
      while i < N:
          form a group starting at i, anchored to the locked scene S_locked
              S_locked = argmax_scene weighted_sim(encode(text_i), scene_emb,
                                                  hint_center_i)
          try to extend the group by sentence i+1, i+2, ...
              extension is accepted if and only if:
                  (a) the new best scene for the extended group is still
                      S_locked, and
                  (b) weighted similarity to S_locked did not drop by more than
                      extend_epsilon below the previous similarity
          on rejection: close the group
          decision gate on the final group weighted similarity:
              >= gate_threshold -> action = "retrieve"
              < gate_threshold  -> action = "generate"
          i = i + len(group)
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        config: Optional[RetrievalGateConfig] = None,
        vram_manager: Optional[VRAMManager] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        manifest: Optional[KeyframesManifest] = None,
    ) -> None:
        self.encoder = text_encoder
        self.config = config or RetrievalGateConfig()
        self.vram_manager = vram_manager
        self.pipeline_config = pipeline_config
        self.manifest = manifest

    def run(
        self,
        sentences: Sequence[Sentence],
        scenes: Sequence[Scene],
    ) -> List[Assignment]:
        if not sentences:
            return []
        if not scenes:
            raise ValueError("At least one scene is required.")

        scene_matrix = _stack_scene_embeddings(scenes)
        scene_centers = _scene_centers(scenes)
        n = len(sentences)
        assignments: List[Assignment] = []

        i = 0
        while i < n:
            assignment = self._build_group(
                i, sentences, scenes, scene_matrix, scene_centers
            )
            assignments.append(assignment)
            consumed = len(assignment.sentence_ids)
            # Defensive: must always advance to prevent infinite loops.
            if consumed < 1:
                raise RuntimeError(
                    f"Group at index {i} consumed zero sentences; refusing to loop."
                )
            i += consumed

        # Cascade Entity Verification Gate (Qwen-VL-guided validation)
        if self.config.enable_cascade_verification and self.manifest and self.vram_manager:
            logger.info("Executing SOTA Cascade Entity Verification Gating using Qwen-VL...")
            
            # Identify model name
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
            if self.pipeline_config:
                model_name = self.pipeline_config.get("models", {}).get("qwen_vl", {}).get("model_name", model_name)
            
            def load_qwen():
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                if "AWQ" in model_name:
                    from awq import AutoAWQForCausalLM
                    model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, trust_remote_code=True, device_map="auto")
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_name)
                return model, processor

            model, processor = self.vram_manager.load_model(f"Qwen2.5-VL ({model_name})", load_qwen)
            
            from PIL import Image
            from qwen_vl_utils import process_vision_info
            import json
            import re
            
            # Map scene_id to manifest scene object
            scene_map = {sc.id: sc for sc in self.manifest.scenes}
            
            for a_idx, assignment in enumerate(assignments):
                if assignment.action == "retrieve":
                    scene_obj = scene_map.get(assignment.scene_id)
                    if not scene_obj:
                        continue
                    
                    img_path = Path("data/intermediate") / self.manifest.video_id / scene_obj.keyframe_path
                    if not img_path.exists():
                        # Try fallback path or skip
                        continue
                    
                    # Joined sentence texts for this group
                    joined_text = " ".join([sentences[sid].text for sid in assignment.sentence_ids])
                    
                    # Build verification prompt
                    prompt = (
                        f"Target description: '{joined_text}'\n"
                        "Verify if the key objects, actions, or setting mentioned in the target description are physically present in the image.\n"
                        "Answer strictly in JSON format: {\"verified\": true} or {\"verified\": false}."
                    )
                    
                    try:
                        messages = [{"role": "user", "content": [{"type": "image", "image": str(img_path)}, {"type": "text", "text": prompt}]}]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
                        
                        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False, temperature=0.0)
                        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        
                        # Parse JSON verified flag
                        match = re.search(r"\{.*?\}", output_text, re.DOTALL)
                        verified = True
                        if match:
                            try:
                                parsed = json.loads(match.group(0))
                                verified = parsed.get("verified", True)
                            except:
                                verified = "true" in output_text.lower()
                        else:
                            verified = "true" in output_text.lower()
                            
                        if not verified:
                            logger.info(f"Cascade Gating REJECTED scene {assignment.scene_id} for text '{joined_text}'. Overriding action to 'generate'.")
                            assignment.action = "generate"
                        else:
                            logger.info(f"Cascade Gating APPROVED scene {assignment.scene_id} for text '{joined_text}'.")
                    except Exception as ex:
                        logger.error(f"Error in cascade entity verification for assignment {a_idx}: {ex}")
            
            # Clean up VRAM
            self.vram_manager.load_model("None (Cleanup)", lambda: None)

        return assignments

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hint_center(self, sentences: Sequence[Sentence], ids: Sequence[int]) -> float:
        """Centre of the merged timestamp hint across the group.
        
        Sentences within a group may not be temporally ordered in the source video
        (LLM summary reorders by narrative/topic), so we take min/max across all
        sentences in the group rather than assuming first/last bound the range.
        """
        starts = [sentences[sid].timestamp_hint[0] for sid in ids]
        ends = [sentences[sid].timestamp_hint[1] for sid in ids]
        lo = min(starts)
        hi = max(ends)
        return (float(lo) + float(hi)) / 2.0

    def _weighted_sims(
        self,
        text_emb: np.ndarray,
        scene_matrix: np.ndarray,
        scene_centers: np.ndarray,
        hint_center: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (weighted, raw_cosine, weights) for every scene."""

        raw = _cosine_to_all(text_emb, scene_matrix)
        if self.config.enable_temporal_prior:
            weights = _gaussian_temporal_weights(
                scene_centers, hint_center, self.config.temporal_sigma
            )
        else:
            weights = np.ones_like(raw, dtype=np.float32)
        weighted = raw * weights
        return weighted, raw, weights

    def _build_group(
        self,
        start: int,
        sentences: Sequence[Sentence],
        scenes: Sequence[Scene],
        scene_matrix: np.ndarray,
        scene_centers: np.ndarray,
    ) -> Assignment:
        cfg = self.config
        sep = cfg.join_sep

        # Seed: group of one sentence, lock to its best (weighted) scene.
        group_ids: List[int] = [start]
        joined_text = sentences[start].text
        joined_emb = self.encoder.encode(joined_text)

        hint_center = self._hint_center(sentences, group_ids)
        weighted, raw, weights = self._weighted_sims(
            joined_emb, scene_matrix, scene_centers, hint_center
        )
        locked_idx = int(np.argmax(weighted))
        best_weighted = float(weighted[locked_idx])
        best_raw = float(raw[locked_idx])
        best_weight = float(weights[locked_idx])
        sim_trail: List[float] = [best_weighted]

        # Try to extend.
        n = len(sentences)
        while (
            start + len(group_ids) < n
            and len(group_ids) < cfg.max_group_size
        ):
            next_idx = start + len(group_ids)
            candidate_text = joined_text + sep + sentences[next_idx].text
            candidate_emb = self.encoder.encode(candidate_text)

            candidate_ids = group_ids + [next_idx]
            candidate_hint_center = self._hint_center(sentences, candidate_ids)
            cand_weighted, cand_raw, cand_weights = self._weighted_sims(
                candidate_emb, scene_matrix, scene_centers, candidate_hint_center
            )
            candidate_best_idx = int(np.argmax(cand_weighted))
            candidate_locked_weighted = float(cand_weighted[locked_idx])

            # Extension accepted only if:
            #   - the candidate group still maps best to the locked scene
            #   - weighted similarity to the locked scene did not drop too far
            same_scene = candidate_best_idx == locked_idx
            tolerable_drop = (
                candidate_locked_weighted >= best_weighted - cfg.extend_epsilon
            )
            if not (same_scene and tolerable_drop):
                break

            group_ids.append(next_idx)
            joined_text = candidate_text
            joined_emb = candidate_emb
            best_weighted = candidate_locked_weighted
            best_raw = float(cand_raw[locked_idx])
            best_weight = float(cand_weights[locked_idx])
            sim_trail.append(best_weighted)

        # Decision gate is on the weighted similarity.
        action = "retrieve" if best_weighted >= cfg.gate_threshold else "generate"

        # Sentences within a group may not be temporally ordered (LLM reorders).
        # Take min/max across all sentences to get true bounding range.
        all_starts = [sentences[sid].timestamp_hint[0] for sid in group_ids]
        all_ends = [sentences[sid].timestamp_hint[1] for sid in group_ids]
        hint_start = min(all_starts)
        hint_end = max(all_ends)

        return Assignment(
            sentence_ids=group_ids,
            scene_id=scenes[locked_idx].id,
            best_similarity=best_weighted,
            raw_cosine=best_raw,
            temporal_weight=best_weight,
            action=action,
            timestamp_hint_merged=(float(hint_start), float(hint_end)),
            similarity_trail=sim_trail,
        )


# ---------------------------------------------------------------------------
# FrameSelector (used by Phase 5 for generation conditioning)
# ---------------------------------------------------------------------------


@dataclass
class FrameSelectorConfig:
    # Strategy for picking a representative frame from the locked scene.
    # "middle"      -> the frame nearest the midpoint of the merged hint range
    # "best_clip"   -> the frame with the highest CLIP/SigLIP similarity to the
    #                  joined sentence text (requires per-frame embeddings)
    strategy: str = "middle"


class FrameSelector:
    """Pick a representative frame from the locked scene for Phase 5.

    Two strategies are supported. ``"middle"`` is dependency-free and always
    available. ``"best_clip"`` requires per-frame embeddings and a text encoder;
    if either is missing, the selector falls back to ``"middle"``.

    The frame chosen here becomes the image-conditioning input for the
    image-to-video diffusion model in Phase 5.
    """

    def __init__(
        self,
        config: Optional[FrameSelectorConfig] = None,
        text_encoder: Optional[TextEncoder] = None,
    ) -> None:
        self.config = config or FrameSelectorConfig()
        self.encoder = text_encoder

    def select(
        self,
        assignment: Assignment,
        scene: Scene,
        frames: Sequence["FrameRef"],
        joined_sentence_text: Optional[str] = None,
    ) -> "FrameRef":
        """Return the chosen frame.

        ``frames`` is the sequence of available frames inside ``scene``, each
        carrying its timestamp and an optional embedding. The selector narrows
        to frames inside ``assignment.timestamp_hint_merged`` first; if the
        narrowed window is empty (hints can lie outside the scene if Phase 2
        timestamps drift), it falls back to all scene frames.
        """

        if not frames:
            raise ValueError(f"Scene {scene.id} has no frames available.")

        lo, hi = assignment.timestamp_hint_merged
        in_window = [f for f in frames if lo <= f.timestamp <= hi]
        candidates = in_window if in_window else list(frames)

        strategy = self.config.strategy
        if strategy == "best_clip":
            if (
                self.encoder is not None
                and joined_sentence_text is not None
                and all(f.embedding is not None for f in candidates)
            ):
                text_emb = self.encoder.encode(joined_sentence_text)
                frame_matrix = np.stack(
                    [f.embedding for f in candidates], axis=0  # type: ignore[arg-type]
                )
                sims = _cosine_to_all(text_emb, frame_matrix)
                return candidates[int(np.argmax(sims))]
            # Fall through to middle if dependencies missing.

        # Default / fallback: middle of window (or middle of all frames).
        midpoint = (candidates[0].timestamp + candidates[-1].timestamp) / 2.0
        return min(candidates, key=lambda f: abs(f.timestamp - midpoint))


@dataclass
class FrameRef:
    """Reference to one frame, optionally with an embedding."""

    timestamp: float
    path: str  # path on disk or whatever the rest of the pipeline expects
    embedding: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Convenience: run end-to-end and summarise
# ---------------------------------------------------------------------------


def summarise_assignments(assignments: Sequence[Assignment]) -> dict:
    """Quick stats useful for smoke tests and threshold tuning."""

    if not assignments:
        return {"num_groups": 0}

    group_sizes = [len(a.sentence_ids) for a in assignments]
    weighted_sims = [a.best_similarity for a in assignments]
    raw_sims = [a.raw_cosine for a in assignments]
    weights = [a.temporal_weight for a in assignments]
    actions = [a.action for a in assignments]

    return {
        "num_groups": len(assignments),
        "num_sentences": sum(group_sizes),
        "group_size_min": min(group_sizes),
        "group_size_max": max(group_sizes),
        "group_size_mean": sum(group_sizes) / len(group_sizes),
        "num_singletons": sum(1 for s in group_sizes if s == 1),
        "num_multi": sum(1 for s in group_sizes if s > 1),
        "weighted_sim_min": min(weighted_sims),
        "weighted_sim_max": max(weighted_sims),
        "weighted_sim_mean": sum(weighted_sims) / len(weighted_sims),
        "raw_cosine_min": min(raw_sims),
        "raw_cosine_max": max(raw_sims),
        "raw_cosine_mean": sum(raw_sims) / len(raw_sims),
        "temporal_weight_min": min(weights),
        "temporal_weight_max": max(weights),
        "temporal_weight_mean": sum(weights) / len(weights),
        "num_retrieve": sum(1 for a in actions if a == "retrieve"),
        "num_generate": sum(1 for a in actions if a == "generate"),
    }

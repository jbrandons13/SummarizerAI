import logging
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_func
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    # These should be installed based on pyproject.toml
    pass

logger = logging.getLogger(__name__)

def compute_rouge(summary_text: str, reference_text: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, summary_text)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rouge_l": scores['rougeL'].fmeasure
    }

def compute_bertscore(summary_text: str, reference_text: str, lang: str = "en") -> float:
    """Compute BERTScore F1."""
    P, R, F1 = bert_score_func([summary_text], [reference_text], lang=lang, verbose=False)
    return float(F1.mean())

class CLIPScoreCalculator:
    """Calculates CLIPScore between images and text using transformers CLIPModel."""
    
    _model = None
    _processor = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            model_id = "openai/clip-vit-large-patch14"
            cls._model = CLIPModel.from_pretrained(model_id).to(cls._device)
            cls._processor = CLIPProcessor.from_pretrained(model_id)
            cls._model.eval()

    def compute(self, image_path: str, text: str) -> float:
        """Compute CLIPScore for a single image-text pair."""
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(text=[text], images=image, return_tensors="pt", padding=True).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # cosine similarity is already computed in CLIP's logits_per_image or text
            # but let's do it manually from features for standard CLIPScore formula
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
            # CLIPScore is typically 2.5 * max(cosine_sim, 0)
            score = 2.5 * max(similarity, 0)
            
        return score

def compute_clipscore_batch(image_paths: List[str], texts: List[str]) -> Dict[str, float]:
    """Compute mean and std of CLIPScore across a batch of matches."""
    if not image_paths or not texts or len(image_paths) != len(texts):
        return {"clipscore_mean": 0.0, "clipscore_std": 0.0}
        
    calculator = CLIPScoreCalculator()
    scores = []
    for img_path, txt in zip(image_paths, texts):
        try:
            scores.append(calculator.compute(img_path, txt))
        except Exception as e:
            logger.error(f"Failed to compute CLIPScore for {img_path}: {e}")
            
    if not scores:
        return {"clipscore_mean": 0.0, "clipscore_std": 0.0}
        
    import numpy as np
    return {
        "clipscore_mean": float(np.mean(scores)),
        "clipscore_std": float(np.std(scores))
    }

def compute_retrieval_recall_at_k(predicted_matches: List[int], ground_truth_matches: List[int], k: int = 3) -> float:
    """
    Compute Recall@K for retrieval.
    predicted_matches: List of predicted scene IDs for each sentence.
    ground_truth_matches: List of correct scene IDs for each sentence.
    """
    if not predicted_matches or not ground_truth_matches:
        return 0.0
        
    hits = 0
    total = len(ground_truth_matches)
    
    for pred, gt in zip(predicted_matches, ground_truth_matches):
        if pred == gt:
            hits += 1
            
    return hits / total

def get_val(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def temporal_alignment_score(matches, summary, manifest, thresholds=(5, 15, 30, 60)):
    """
    Measures how close retrieved scenes are to source content location.
    """
    import numpy as np
    errors = []
    if not manifest.scenes:
        return {"mean_temporal_error": -1, "n_evaluated": 0}
    video_duration = max(s.end_seconds for s in manifest.scenes)
    within_counts = {t: 0 for t in thresholds}

    for match in matches:
        match_sent_id = get_val(match, "sentence_id")
        sentence = next((s for s in summary.sentences if s.id == match_sent_id), None)
        if not sentence:
            if match_sent_id is not None and 0 <= match_sent_id < len(summary.sentences):
                sentence = summary.sentences[match_sent_id]
            else:
                continue

        scene_id = get_val(match, "matched_scene_id", get_val(match, "source_scene_id", None))
        scene = next((s for s in manifest.scenes if s.id == scene_id), None)
        if not scene:
            continue

        hint = sentence.source_timestamp_hint
        if not hint or len(hint) < 2:
            continue

        # Use the matched frame's timestamp if available, else scene midpoint
        retrieved_ts = get_val(match, "best_frame_timestamp", 0.0) or scene.keyframe_timestamp

        if hint[0] <= retrieved_ts <= hint[1]:
            error = 0.0
        else:
            error = min(abs(retrieved_ts - hint[0]), abs(retrieved_ts - hint[1]))

        errors.append(error)
        for t in thresholds:
            if error <= t:
                within_counts[t] += 1

    if not errors:
        return {"mean_temporal_error": -1, "n_evaluated": 0}

    result = {
        "n_evaluated": len(errors),
        "mean_temporal_error_seconds": float(np.mean(errors)),
        "median_temporal_error_seconds": float(np.median(errors)),
        "normalized_temporal_error": float(np.mean(errors) / video_duration),
    }
    for t in thresholds:
        result[f"temporal_accuracy_within_{t}s"] = within_counts[t] / len(errors)
    return result

def visual_coherence_score(matches, frame_embeddings):
    """
    Average cosine similarity between consecutive matched FRAMES.
    """
    import numpy as np
    consecutive_sims = []

    def find_nearest_embedding(scene_id, ts, frame_embeddings):
        best_key = None
        min_dist = float('inf')
        for k in frame_embeddings.keys():
            if k[0] == scene_id:
                dist = abs(k[1] - ts)
                if dist < min_dist:
                    min_dist = dist
                    best_key = k
        if best_key:
            return frame_embeddings[best_key]
        return None

    for i in range(len(matches) - 1):
        scene_a = get_val(matches[i], "matched_scene_id", get_val(matches[i], "source_scene_id", None))
        ts_a = get_val(matches[i], "best_frame_timestamp", 0.0)
        scene_b = get_val(matches[i + 1], "matched_scene_id", get_val(matches[i + 1], "source_scene_id", None))
        ts_b = get_val(matches[i + 1], "best_frame_timestamp", 0.0)

        key_a = (scene_a, ts_a)
        key_b = (scene_b, ts_b)

        emb_a = frame_embeddings.get(key_a)
        if emb_a is None:
            emb_a = find_nearest_embedding(scene_a, ts_a, frame_embeddings)
            
        emb_b = frame_embeddings.get(key_b)
        if emb_b is None:
            emb_b = find_nearest_embedding(scene_b, ts_b, frame_embeddings)

        if emb_a is None or emb_b is None:
            continue

        norm_a, norm_b = np.linalg.norm(emb_a), np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            continue
        consecutive_sims.append(float(np.dot(emb_a, emb_b) / (norm_a * norm_b)))

    if not consecutive_sims:
        return {"visual_coherence_mean": 0.0, "visual_coherence_std": 0.0, "n_pairs": 0}

    return {
        "visual_coherence_mean": float(np.mean(consecutive_sims)),
        "visual_coherence_std": float(np.std(consecutive_sims)),
        "n_pairs": len(consecutive_sims),
    }

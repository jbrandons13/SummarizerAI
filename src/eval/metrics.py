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

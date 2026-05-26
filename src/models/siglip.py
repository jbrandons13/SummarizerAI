import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
from PIL import Image
import joblib
import logging
from transformers import AutoProcessor, Siglip2Model

logger = logging.getLogger(__name__)

class SigLIPEncoder:
    def __init__(self, vram_manager, model_name: str = "google/siglip2-so400m-patch16-naflex"):
        self.vram_manager = vram_manager
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._dim = None

    def get_embedding_dim(self) -> int:
        if self._dim is None:
            self._load()
            self._dim = self.model.config.hidden_size
        return self._dim

    def _load(self):
        if self.model is None:
            def loader():
                processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                model = Siglip2Model.from_pretrained(
                    self.model_name, ignore_mismatched_sizes=True
                ).to("cuda")
                return model, processor
            self.model, self.processor = self.vram_manager.load_model(f"SigLIP2 ({self.model_name})", loader)

    def encode(self, text: str) -> np.ndarray:
        self._load()
        text_inputs = self.processor(text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(text_features, "pooler_output", text_features[0])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze(0).cpu().numpy()

    def embed_scenes(self, video_id: str, manifest: Any, progress_callback: Any = None, intermediate_dir: Optional[Union[str, Path]] = None) -> Dict[Tuple[int, float], np.ndarray]:
        model_slug = self.model_name.replace("/", "_").replace("-", "_")
        base_dir = Path(intermediate_dir) if intermediate_dir is not None else Path("data/intermediate")
        video_dir = base_dir / video_id
        cache_path = video_dir / f"embeddings_{model_slug}.joblib"
        
        if cache_path.exists():
            logger.info(f"Loaded SigLIP embeddings from cache: {cache_path}")
            return joblib.load(cache_path)

        self._load()
        frame_embeddings = {}
        all_frames = []
        for scene in manifest.scenes:
            for path, ts in zip(scene.multi_frame_paths, scene.multi_frame_timestamps):
                all_frames.append((scene.id, ts, video_dir / path))
        
        for i, (scene_id, ts, img_path) in enumerate(all_frames):
            image = Image.open(img_path).convert("RGB")
            img_inputs = self.processor(images=image, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**img_inputs)
                if not isinstance(image_features, torch.Tensor):
                    image_features = getattr(image_features, "pooler_output", image_features[0])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                frame_embeddings[(scene_id, ts)] = image_features.squeeze(0).cpu().numpy()
            
            if progress_callback and i % 5 == 0:
                pct = int(30 + (i / len(all_frames)) * 50)
                progress_callback.update(4, "Visual Retrieval", pct, f"SigLIP encoding frame {i+1}/{len(all_frames)}")

        joblib.dump(frame_embeddings, cache_path)
        return frame_embeddings

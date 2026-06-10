import torch
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from PIL import Image
import os
import json

class ScoringWrap:
    def __init__(self, clip_model="openai/clip-vit-large-patch14", dino_model="facebook/dinov2-base"):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[ScoringWrap] Loading DINOv2: {dino_model}")
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_model)
        self.dino_model = AutoModel.from_pretrained(dino_model).eval().to(self.dev)
        
        print(f"[ScoringWrap] Loading CLIP: {clip_model}")
        self.clip_model = CLIPModel.from_pretrained(clip_model).eval().to(self.dev)
        self.clip_proc = CLIPProcessor.from_pretrained(clip_model)
        
        self.emb_cache = {}
        
    @torch.no_grad()
    def embed_dino(self, image_path):
        if image_path not in self.emb_cache:
            img = Image.open(image_path).convert("RGB")
            inp = self.dino_proc(images=img, return_tensors="pt").to(self.dev)
            feat = self.dino_model(**inp).last_hidden_state[:, 0]
            v = torch.nn.functional.normalize(feat, dim=-1).cpu().squeeze(0)
            self.emb_cache[image_path] = v
        return self.emb_cache[image_path]
        
    @torch.no_grad()
    def get_clip_t(self, image_path, text):
        im = Image.open(image_path).convert("RGB")
        ii = self.clip_proc(images=im, return_tensors="pt").to(self.dev)
        f = self.clip_model.get_image_features(**ii)
        f = f / f.norm(dim=-1, keepdim=True)
        
        ti = self.clip_proc(text=[text], return_tensors="pt", padding=True).to(self.dev)
        t = self.clip_model.get_text_features(**ti)
        t = t / t.norm(dim=-1, keepdim=True)
        
        return float((f * t).sum().item())
        
    def score_shot(self, image_path, w0_path, ref_path, text_prompt):
        res = {}
        if w0_path and os.path.exists(w0_path):
            emb1 = self.embed_dino(image_path)
            emb2 = self.embed_dino(w0_path)
            res["c_s"] = float((emb1 * emb2).sum().item())
        else:
            res["c_s"] = 1.0 # If w0 is itself or missing
            
        if ref_path and os.path.exists(ref_path):
            emb1 = self.embed_dino(image_path)
            emb2 = self.embed_dino(ref_path)
            res["ref_sim"] = float((emb1 * emb2).sum().item())
        else:
            res["ref_sim"] = float("nan")
            
        res["clip_t"] = self.get_clip_t(image_path, text_prompt)
        return res

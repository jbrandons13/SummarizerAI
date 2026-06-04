#!/usr/bin/env python3
"""
collapse_metrics.py -- turn the side-by-side picture into a measured result.

From the weight-sweep manifest (weight_sweep.py output), compute the numbers that
prove the reward-collapse finding:

  (1) sim-to-REFERENCE   -- DINOv2 sim(image, reference). RISES with weight, because the
                            image literally becomes the reference. This is the "usual"
                            score that *rewards* the collapse.
  (2) sim-to-OWN-W0      -- DINOv2 sim(image, this shot's own weight=0 image). FALLS with
                            weight, because the shot abandons its own scene. This is the
                            content-preservation side ("did it keep itself, or just copy?").
  (3) inter-shot sim     -- mean pairwise sim AMONG the shots at each weight. RISES toward
                            1.0 as all shots collapse into the same reference.

The story: (1) goes UP while (2) goes DOWN -> a higher score is bought by destroying
content. The two curves crossing IS the reward-collapse, quantified.

Outputs:
  <out>/collapse_metrics.csv   per shot x weight + per-weight means
  <out>/collapse_curve.png     line plot of (1) rising vs (2) falling [+ (3)]

NOTE on matching thesis numbers: this uses a fallback DINOv2 (torch.hub). The TREND
(up vs down) is what proves the point and is robust to the exact variant. If you want
the absolute numbers to match your section 4, recompute (1)/(2) with your own
concept_consistency.py and keep the same up/down relationship.

Usage:
  python collapse_metrics.py --manifest .../collapse_evidence/manifest.json --out .../collapse_evidence
"""
import argparse
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--dino-model", default="dinov2_vitb14")
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(dev)

    cache = {}

    def emb(path):
        if path not in cache:
            img = Image.open(path).convert("RGB")
            inp = proc(images=img, return_tensors="pt").to(dev)
            with torch.no_grad():
                feat = model(**inp).last_hidden_state[:, 0]  # CLS token
            v = torch.nn.functional.normalize(feat, dim=-1).cpu().squeeze(0)
            cache[path] = v
        return cache[path]

    def sim(a, b):
        return float((emb(a) * emb(b)).sum().item())

    man = json.load(open(args.manifest))
    ref = args.reference
    
    # Convert list of dicts to rows
    shots = defaultdict(list)
    for entry in man:
        shots[entry["shot_id"]].append(entry)
        
    rows = []
    for shot_id, cells in shots.items():
        rows.append({
            "label": shot_id,
            "cells": [{"weight": c["weight"], "image": c["path"]} for c in cells]
        })
        
    weights = sorted({c["weight"] for r in rows for c in r["cells"]})
    w0 = min(weights)
    os.makedirs(args.out, exist_ok=True)

    csv = ["shot,weight,sim_to_reference,sim_to_own_w0"]
    by_ref = defaultdict(list)
    by_self = defaultdict(list)
    for r in rows:
        label = r.get("label", "shot")
        cells = {c["weight"]: c["image"] for c in r["cells"]}
        w0img = cells.get(w0)
        for w in weights:
            img = cells.get(w)
            if not (img and os.path.exists(img)):
                continue
            s_ref = sim(img, ref)
            s_self = sim(img, w0img) if (w0img and os.path.exists(w0img)) else float("nan")
            csv.append(f"\"{label}\",{w:g},{s_ref:.4f},{s_self:.4f}")
            by_ref[w].append(s_ref)
            by_self[w].append(s_self)

    # inter-shot similarity per weight (mean pairwise among the shots)
    inter = {}
    for w in weights:
        imgs = [{c["weight"]: c["image"] for c in r["cells"]}.get(w) for r in rows]
        imgs = [p for p in imgs if p and os.path.exists(p)]
        pair = [sim(imgs[i], imgs[j]) for i in range(len(imgs)) for j in range(i + 1, len(imgs))]
        inter[w] = sum(pair) / len(pair) if pair else float("nan")

    csv.append("")
    csv.append("weight,mean_sim_to_reference,mean_sim_to_own_w0,mean_inter_shot_sim")
    m_ref, m_self, m_inter = [], [], []
    for w in weights:
        mr = sum(by_ref[w]) / len(by_ref[w]) if by_ref[w] else float("nan")
        ms = sum(by_self[w]) / len(by_self[w]) if by_self[w] else float("nan")
        mi = inter[w]
        m_ref.append(mr); m_self.append(ms); m_inter.append(mi)
        csv.append(f"{w:g},{mr:.4f},{ms:.4f},{mi:.4f}")

    cpath = os.path.join(args.out, "collapse_metrics.csv")
    open(cpath, "w").write("\n".join(csv) + "\n")
    print(f"[ok] {cpath}\n")
    print("weight | sim->reference (UP) | sim->own w0 (DOWN) | inter-shot (UP)")
    for i, w in enumerate(weights):
        print(f"  {w:g}   |     {m_ref[i]:.3f}        |     {m_self[i]:.3f}        |   {m_inter[i]:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(weights, m_ref, "o-", color="#c0392b",
                 label="similarity to reference  (the score that rewards copying)  \u2191")
        plt.plot(weights, m_self, "s-", color="#2471a3",
                 label="similarity to the shot's own original scene  (content kept)  \u2193")
        plt.plot(weights, m_inter, "^--", color="#7d8c00",
                 label="similarity among the shots  (they converge to one)  \u2191")
        plt.xlabel("anchoring weight")
        plt.ylabel("DINOv2 cosine similarity")
        plt.title("Reward collapse: the score rises while the content is destroyed")
        plt.ylim(0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, loc="center left")
        plt.tight_layout()
        ppath = os.path.join(args.out, "collapse_curve.png")
        plt.savefig(ppath, dpi=150)
        print(f"\n[ok] {ppath}")
    except Exception as e:
        print(f"\n[warn] plot skipped ({e}); `pip install matplotlib` to get collapse_curve.png")


if __name__ == "__main__":
    main()

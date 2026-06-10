#!/usr/bin/env python3
"""
adaptive_anchor.py -- THE METHOD: per-shot content-preserving concept anchoring.

The reward-collapse showed scalar anchoring trades concept-consistency against content, and
block-wise (Idea-1) could NOT escape that trade-off -- it is intrinsic to image anchoring;
re-routing the injection only moves you ALONG the frontier. So instead of trying to ELIMINATE
the trade-off, this method NAVIGATES it optimally, per shot, using the content-preservation
counter-metric (sim-to-own) as a controller:

  for each shot, pick the LARGEST anchoring weight whose content-kept (sim-to-own) is still
  >= tau -- i.e. give each shot as much concept as it can take *before* it collapses.

Shots whose own scene is far from the concept can take more anchoring before collapsing;
shots already near it need little. A single global weight cannot do this; an adaptive
per-shot weight can -> more concept presence at a guaranteed content floor.

Reuses the EXISTING weight-sweep images and the EXISTING collapse_metrics.csv (sim-to-own
per shot per weight). NO regeneration. Concept is scored with CLIP text (does NOT reward
copying), consistent with concept_eval.py; content is the DINOv2 sim-to-own from the CSV
(the same counter-metric as the finding).

Outputs:
  <out>/adaptive_anchor.csv   per-shot chosen weight w* + per-scheme means
  <out>/adaptive_anchor.png   fixed-weight curve vs the adaptive point (fair plane)
  <out>/adaptive_grid.png     per shot: own | a fixed weight | adaptive

Usage:
  python adaptive_anchor.py \
    --manifest    .../collapse_evidence/manifest.json \
    --metrics-csv .../collapse_evidence/collapse_metrics.csv \
    --tau 0.70 --baselines 0.2,0.4,0.6 \
    --concept "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones" \
    --out .../collapse_evidence
"""
import argparse
import csv as _csv
import json
import os
from collections import defaultdict


def read_content(csv_path):
    """content[(label, weight)] = sim_to_own_w0, from collapse_metrics.csv first section."""
    out = {}
    with open(csv_path) as fh:
        rows = list(_csv.reader(fh))
    hdr = None
    for i, r in enumerate(rows):
        rr = [c.strip() for c in r]
        if rr[:2] == ["shot", "weight"] and "sim_to_own_w0" in rr:
            hdr = i
            wi = rr.index("weight")
            oi = rr.index("sim_to_own_w0")
            break
    if hdr is None:
        raise SystemExit("[err] could not find 'shot,weight,...,sim_to_own_w0' header in metrics csv")
    for r in rows[hdr + 1:]:
        if not r or not r[0].strip():
            break
        label = r[0].strip().strip('"')
        try:
            out[(label, round(float(r[wi]), 4))] = float(r[oi])
        except (ValueError, IndexError):
            pass
    return out


def nearest(weights, target):
    return min(weights, key=lambda w: abs(w - target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="weight_sweep manifest.json")
    ap.add_argument("--metrics-csv", required=True, help="collapse_metrics.csv (per-shot sim_to_own)")
    ap.add_argument("--tau", type=float, default=0.70, help="content-kept floor (sim_to_own >= tau)")
    ap.add_argument("--baselines", default="0.2,0.4,0.6", help="fixed weights to compare against")
    ap.add_argument("--concept", default="a colorful cartoon illustration of rocks, rocky terrain, boulders and stones")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    content = read_content(args.metrics_csv)
    man = json.load(open(args.manifest))
    if isinstance(man, list):
        grouped = defaultdict(list)
        for item in man:
            grouped[item.get("shot_id", "shot")].append({"weight": item["weight"], "image": item["path"]})
        rows = [{"label": shot_id, "cells": cells} for shot_id, cells in grouped.items()]
    else:
        rows = man["rows"]

    os.makedirs(args.out, exist_ok=True)

    # img[(label, weight)] = path ; labels in order ; available weights
    img, labels = {}, []
    for r in rows:
        label = r.get("label", "shot")
        labels.append(label)
        for c in r["cells"]:
            img[(label, round(float(c["weight"]), 4))] = c["image"]
    weights = sorted({round(float(c["weight"]), 4) for r in rows for c in r["cells"]})
    w0 = min(weights)

    # ---- THE METHOD: per-shot w* = max weight with content >= tau ----
    wstar = {}
    for label in labels:
        ok = [w for w in weights if content.get((label, w), -1) >= args.tau]
        wstar[label] = max(ok) if ok else w0

    # schemes to compare: adaptive + each fixed baseline (snapped to an available weight)
    base_ws = [nearest(weights, float(x)) for x in args.baselines.split(",") if x.strip()]
    schemes = {"adaptive": {label: wstar[label] for label in labels}}
    for bw in base_ws:
        schemes[f"fixed_w{bw:g}"] = {label: bw for label in labels}

    # ---- CLIP concept scoring (does NOT reward copying) ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] loading CLIP: {args.clip_model}")
    model = CLIPModel.from_pretrained(args.clip_model).eval().to(dev)
    proc = CLIPProcessor.from_pretrained(args.clip_model)
    ti = proc(text=[args.concept], return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        t = model.get_text_features(**ti); t = t / t.norm(dim=-1, keepdim=True)
    _cs = {}

    def concept_score(path):
        if path not in _cs:
            ii = proc(images=Image.open(path).convert("RGB"), return_tensors="pt").to(dev)
            with torch.no_grad():
                f = model.get_image_features(**ii); f = f / f.norm(dim=-1, keepdim=True)
            _cs[path] = float((f * t).sum().item())
        return _cs[path]

    # ---- per-scheme means + per-shot table ----
    out_csv = [f'concept_text,"{args.concept}"', f"tau,{args.tau}", "",
               "shot,adaptive_w*,content_at_w*(sim_to_own),concept_at_w*(CLIP)"]
    for label in labels:
        w = wstar[label]
        p = img.get((label, w))
        sid = label.split(":")[0].strip()
        out_csv.append(f'"{sid}",{w:g},{content.get((label,w),float("nan")):.4f},'
                       f'{concept_score(p) if p else float("nan"):.4f}')

    out_csv += ["", "scheme,mean_concept(CLIP),mean_content(sim_to_own)"]
    means = {}
    for name, sel in schemes.items():
        cs, ct = [], []
        for label in labels:
            w = sel[label]
            p = img.get((label, w))
            if p and os.path.exists(p):
                cs.append(concept_score(p))
                ct.append(content.get((label, w), float("nan")))
        mc = sum(cs) / len(cs) if cs else float("nan")
        mt = sum([x for x in ct if x == x]) / max(1, len([x for x in ct if x == x]))
        means[name] = (mc, mt)
        out_csv.append(f'"{name}",{mc:.4f},{mt:.4f}')

    cpath = os.path.join(args.out, "adaptive_anchor.csv")
    open(cpath, "w").write("\n".join(out_csv) + "\n")
    print(f"\n[ok] {cpath}\n")
    print("per-shot chosen weight (adaptive):")
    for label in labels:
        sid = label.split(":")[0].strip()
        print(f"  {sid:<12} w* = {wstar[label]:g}   (content {content.get((label,wstar[label]),float('nan')):.3f})")
    print("\nscheme           concept(CLIP)   content(sim->own)")
    for name in schemes:
        mc, mt = means[name]
        print(f"  {name:<14} {mc:>10.4f}     {mt:>10.4f}")

    # ---- figure: fair plane, fixed curve vs adaptive ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fixed = sorted([n for n in schemes if n.startswith("fixed")], key=lambda n: means[n][0])
        fig, ax = plt.subplots(figsize=(7.6, 6.0))
        if len(fixed) >= 2:
            fx = [means[n][0] for n in fixed]; fy = [means[n][1] for n in fixed]
            ax.plot(fx, fy, "o-", color="#c0392b", lw=2, ms=8, zorder=3, label="fixed global weight")
            ax.fill_between(fx, fy, 1.02, color="#e8f5e9", zorder=0)
            for n in fixed:
                ax.annotate(n.replace("fixed_", ""), (means[n][0], means[n][1]),
                            textcoords="offset points", xytext=(6, -11), fontsize=8, color="#c0392b")
        mc, mt = means["adaptive"]
        ax.scatter([mc], [mt], marker="*", s=520, color="#1565c0", edgecolor="white", lw=1.3,
                   zorder=4, label="adaptive per-shot (ours)")
        ax.annotate("adaptive", (mc, mt), textcoords="offset points", xytext=(9, 7),
                    fontsize=10, fontweight="bold", color="#1565c0")
        ax.axhline(args.tau, color="#999", ls=":", lw=1.2)
        ax.text(ax.get_xlim()[0], args.tau, f" content floor \u03c4={args.tau:g}", fontsize=7.5,
                color="#666", va="bottom")
        ax.set_xlabel("concept present  =  CLIP text\u2192concept score  (does NOT reward copying)  \u2192")
        ax.set_ylabel("content kept  =  similarity to the shot's own scene  \u2192")
        ax.set_title("The method: adaptive per-shot anchoring vs a fixed global weight")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.text(0.99, 0.99, "above the red line = more concept\nat the same content = ours wins",
                ha="right", va="top", fontsize=8, color="#2e7d32", transform=ax.transAxes)
        fig.tight_layout()
        fpath = os.path.join(args.out, "adaptive_anchor.png")
        fig.savefig(fpath, dpi=150)
        print(f"\n[ok] {fpath}")
    except Exception as e:
        print(f"\n[warn] plot skipped ({e})")

    # ---- grid: own | one fixed | adaptive ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
        midw = base_ws[len(base_ws) // 2] if base_ws else w0
        cols = [("own (w%g)" % w0, {label: w0 for label in labels}),
                ("fixed w%g" % midw, {label: midw for label in labels}),
                ("adaptive (per-shot)", {label: wstar[label] for label in labels})]
        nrow, ncol = len(labels), len(cols)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 1.9 * nrow))
        if nrow == 1:
            axes = axes.reshape(1, -1)
        for ri, label in enumerate(labels):
            sid = label.split(":")[0].strip()
            for ci, (title, sel) in enumerate(cols):
                ax = axes[ri][ci]; ax.axis("off")
                w = sel[label]; p = img.get((label, w))
                if p and os.path.exists(p):
                    ax.imshow(PILImage.open(p).convert("RGB"))
                    sub = f"w={w:g} | own {content.get((label,w),float('nan')):.2f} | cpt {concept_score(p):.2f}"
                    ax.text(0.5, -0.09, sub, ha="center", va="top", fontsize=7, transform=ax.transAxes)
                    if ci == 2:  # frame the adaptive (ours) column
                        for sp in ax.spines.values():
                            sp.set_visible(True); sp.set_color("#1565c0"); sp.set_linewidth(3)
                        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(title, fontsize=9, fontweight="bold")
            axes[ri][0].text(-0.08, 0.5, sid, ha="right", va="center", fontsize=9, rotation=90,
                             transform=axes[ri][0].transAxes)
        fig.suptitle("Adaptive per-shot anchoring (blue) keeps each shot below its collapse point", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        gpath = os.path.join(args.out, "adaptive_grid.png")
        fig.savefig(gpath, dpi=130)
        print(f"[ok] {gpath}")
    except Exception as e:
        print(f"[warn] grid skipped ({e})")


if __name__ == "__main__":
    main()

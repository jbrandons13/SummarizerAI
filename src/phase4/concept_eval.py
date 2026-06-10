#!/usr/bin/env python3
"""
concept_eval.py -- re-evaluate Idea-1 with a concept measure that does NOT reward copying.

WHY THIS EXISTS
  The structured trade-off used sim-to-reference (DINOv2) as the "concept" axis. But
  sim-to-reference is exactly the score the thesis shows REWARDS copying, and it is
  inversely coupled with sim-to-own -- so on the (sim-ref, sim-own) plane the upper-right
  is unreachable by construction. That makes it an unfair judge of a method that aims to
  keep the concept recognizable WITHOUT copying.

  This replaces the concept axis with a CLIP TEXT-to-concept score: how strongly each image
  depicts the recurring concept (e.g. "rocks / rocky terrain"), measured against TEXT, not
  against the reference image. A copy and a non-copy that both show the concept score about
  the same -> it does not reward copying. Paired with content-kept (sim-to-own, the
  counter-metric, read from structured_metrics.csv), the upper-right -- concept PRESENT and
  own content KEPT -- is now reachable, so we can fairly ask whether block-wise gets there.

  Reuses the EXISTING images (no regeneration).

HONEST LIMITS
  CLIP text-to-concept is coarse. If every config shows the concept (all "rocky"), the
  x-axis can come out flat -> then this is inconclusive too, and the visual audit decides.
  This is a *fairer* test, not a perfect one.

OUTPUTS
  <out>/concept_eval.csv     per (shot x config) + per-config means
  <out>/concept_eval.png     concept-present (x, CLIP-text) vs content-kept (y); above the
                             scalar line = better than scalar at equal concept = a real win.

USAGE
  python concept_eval.py \
    --manifest    .../ide1_structured/manifest.json \
    --metrics-csv .../ide1_structured/structured_metrics.csv \
    --concept "colorful cartoon rocks, rocky terrain, boulders, stones" \
    --out .../ide1_structured
"""
import argparse
import csv
import json
import os
from collections import defaultdict


def read_content_kept(csv_path):
    """Pull per-config mean_sim_to_own out of structured_metrics.csv (the means section)."""
    if not (csv_path and os.path.exists(csv_path)):
        return {}
    out = {}
    with open(csv_path) as fh:
        rows = list(csv.reader(fh))
    # find the means header: config,kind,mean_sim_to_reference,mean_sim_to_own,mean_inter_shot
    hdr_i = None
    for i, r in enumerate(rows):
        if r and r[0].strip() == "config" and "mean_sim_to_own" in [c.strip() for c in r]:
            hdr_i = i
            col = [c.strip() for c in r].index("mean_sim_to_own")
            break
    if hdr_i is None:
        return {}
    for r in rows[hdr_i + 1:]:
        if not r or not r[0].strip():
            break
        cfg = r[0].strip().strip('"')
        try:
            out[cfg] = float(r[col])
        except (ValueError, IndexError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--metrics-csv", default="", help="structured_metrics.csv (for content-kept). default: sibling of manifest")
    ap.add_argument("--concept", default="a colorful cartoon illustration of rocks, rocky terrain, boulders and stones",
                    help="TEXT describing the recurring concept (match your canonical anchor)")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    man = json.load(open(args.manifest))
    configs = man["configs"]
    kind = man.get("config_kind", {})
    shots = man["shots"]
    ref = man["reference"]
    img = {(it["shot"], it["config"]): it["image"] for it in man["items"]}
    os.makedirs(args.out, exist_ok=True)

    csv_path = args.metrics_csv or os.path.join(os.path.dirname(args.manifest), "structured_metrics.csv")
    content_kept = read_content_kept(csv_path)
    if not content_kept:
        print(f"[warn] could not read content-kept from {csv_path}; y-axis will be missing.\n"
              f"       (run structured_compare.py first, or pass --metrics-csv)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] loading CLIP: {args.clip_model}")
    model = CLIPModel.from_pretrained(args.clip_model).eval().to(dev)
    proc = CLIPProcessor.from_pretrained(args.clip_model)

    # concept text embedding
    ti = proc(text=[args.concept], return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        t = model.get_text_features(**ti)
        t = t / t.norm(dim=-1, keepdim=True)

    def concept_score(path):
        im = Image.open(path).convert("RGB")
        ii = proc(images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            f = model.get_image_features(**ii)
            f = f / f.norm(dim=-1, keepdim=True)
        return float((f * t).sum().item())

    # sanity: the reference itself should score high on the concept
    ref_cs = concept_score(ref)
    print(f"[sanity] concept score of the REFERENCE image = {ref_cs:.4f} "
          f"(should be among the highest)")

    rows_csv = ['concept_text,"' + args.concept.replace('"', "'") + '"',
                "shot,config,kind,clip_concept"]
    by_cfg = defaultdict(list)
    for s in shots:
        for c in configs:
            p = img.get((s, c))
            if not (p and os.path.exists(p)):
                continue
            cs = concept_score(p)
            rows_csv.append(f'"{s}","{c}",{kind.get(c,"")},{cs:.4f}')
            by_cfg[c].append(cs)

    rows_csv += ["", "config,kind,mean_clip_concept,content_kept(sim_to_own)"]
    cc, ck = {}, {}
    for c in configs:
        m = sum(by_cfg[c]) / len(by_cfg[c]) if by_cfg[c] else float("nan")
        cc[c] = m
        ck[c] = content_kept.get(c, float("nan"))
        rows_csv.append(f'"{c}",{kind.get(c,"")},{m:.4f},{ck[c]:.4f}')

    cpath = os.path.join(args.out, "concept_eval.csv")
    open(cpath, "w").write("\n".join(rows_csv) + "\n")
    print(f"\n[ok] {cpath}\n")
    print("config            kind    concept(CLIP-text)   content-kept(sim->own)")
    for c in configs:
        print(f"  {c:<15} {kind.get(c,''):<7} {cc[c]:>12.4f}        {ck[c]:>10.4f}")

    # ---- figure: concept-present (x, does NOT reward copying) vs content-kept (y) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [cc[c] for c in configs if cc[c] == cc[c]]
        ys = [ck[c] for c in configs if ck[c] == ck[c]]
        if not xs or not ys:
            print("[warn] not enough data to plot")
            return
        xpad = max(0.01, (max(xs) - min(xs)) * 0.15)
        ypad = max(0.02, (max(ys) - min(ys)) * 0.15)
        xlo, xhi = min(xs) - xpad, max(max(xs), ref_cs) + xpad
        ylo, yhi = min(ys) - ypad, 1.02

        fig, ax = plt.subplots(figsize=(7.6, 6.0))

        # scalar frontier (sorted by concept), then shade ABOVE it = better content at equal concept
        scal = sorted([c for c in configs if kind.get(c) == "scalar" and cc[c] == cc[c] and ck[c] == ck[c]],
                      key=lambda c: cc[c])
        if len(scal) >= 2:
            sx = [cc[c] for c in scal]
            sy = [ck[c] for c in scal]
            ax.plot(sx, sy, "o-", color="#c0392b", lw=2, ms=8, zorder=3,
                    label="scalar anchoring")
            ax.fill_between(sx, sy, yhi, color="#e8f5e9", zorder=0)
            for c in scal:
                ax.annotate(c.replace("scalar_", ""), (cc[c], ck[c]),
                            textcoords="offset points", xytext=(6, -11), fontsize=8, color="#c0392b")

        blk = [c for c in configs if kind.get(c) == "block" and cc[c] == cc[c] and ck[c] == ck[c]]
        if blk:
            ax.scatter([cc[c] for c in blk], [ck[c] for c in blk], marker="*", s=440,
                       color="#1565c0", edgecolor="white", lw=1.2, zorder=4,
                       label="block-wise (Idea-1)")
        for c in blk:
            ax.annotate(c, (cc[c], ck[c]), textcoords="offset points",
                        xytext=(8, 6), fontsize=9, fontweight="bold", color="#1565c0")

        # reference's own concept score, as a vertical guide (how "on-concept" a full copy is)
        ax.axvline(ref_cs, color="#888", ls=":", lw=1.2, zorder=1)
        ax.text(ref_cs, ylo, " reference's\n concept level", fontsize=7.5, color="#666",
                ha="left", va="bottom")

        ax.set_xlabel("concept present  =  CLIP text\u2192concept score  (does NOT reward copying)  \u2192")
        ax.set_ylabel("content kept  =  similarity to the shot's own scene  \u2192")
        ax.set_title("Idea-1 re-eval on a fair plane:\nis the concept present AND the content kept?")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.text(0.99, 0.99, "better (green): more content kept\nat the same concept level",
                ha="right", va="top", fontsize=8, color="#2e7d32", transform=ax.transAxes)
        fig.tight_layout()
        fpath = os.path.join(args.out, "concept_eval.png")
        fig.savefig(fpath, dpi=150)
        print(f"\n[ok] {fpath}")
        print("[read] a block star ABOVE the red scalar line = keeps more content at the same")
        print("       concept level than scalar can = Idea-1 genuinely helps. On/below = it does not.")
    except Exception as e:
        print(f"\n[warn] plot skipped ({e})")


if __name__ == "__main__":
    main()

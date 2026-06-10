#!/usr/bin/env python3
"""
structured_compare.py -- Idea-1 evaluator: does block-wise anchoring BREAK the collapse?

Reads the structured_sweep.py manifest and, per config, measures the SAME two axes the
reward-collapse finding used, plus inter-shot:

  (1) sim-to-REFERENCE   DINOv2 sim(image, canonical reference).  = concept consistency.
                         (the "score" that, under scalar anchoring, rewards copying.)
  (2) sim-to-OWN         DINOv2 sim(image, this shot's OWN no-anchor scene).
                         = content / context preserved. HIGHER is better.
  (3) inter-shot sim     mean pairwise sim AMONG the shots in that config (1.0 = all identical).

The reward-collapse is the inverse coupling under SCALAR anchoring: pushing (1) up drags
(2) down. Idea-1 WINS if a block-wise config reaches concept consistency (1) comparable to
a collapsing scalar config, but with (2) staying HIGH -- i.e. it sits ABOVE/RIGHT of the
scalar trade-off frontier in the (concept-consistency, content-kept) plane.

Outputs:
  <out>/structured_metrics.csv     per (shot x config) + per-config means
  <out>/structured_tradeoff.png    the money figure: content-kept (y) vs concept-consistency (x)
  <out>/structured_grid.png        visual grid: rows = shots, cols = [reference] + configs

Usage:
  python structured_compare.py --manifest .../ide1_structured/manifest.json --out .../ide1_structured

NOTE: uses a fallback DINOv2 (torch.hub), identical to collapse_metrics.py, so the numbers
are directly comparable to the collapse curves. For thesis-final absolute numbers, recompute
(1)/(2) with your own concept_consistency.py keeping the same relationships.
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
                f = model(**inp).pooler_output
            cache[path] = (f / f.norm(dim=-1, keepdim=True)).cpu()
        return cache[path]

    def sim(a, b):
        return float((emb(a) * emb(b)).sum().item())

    man = json.load(open(args.manifest))
    ref = man["reference"]
    own_cfg = man["own_baseline"]
    configs = man["configs"]
    kind = man.get("config_kind", {})
    shots = man["shots"]
    os.makedirs(args.out, exist_ok=True)

    # index images by (shot, config)
    img = {(it["shot"], it["config"]): it["image"] for it in man["items"]}
    own = {s: img.get((s, own_cfg)) for s in shots}  # each shot's own no-anchor scene

    # ---- per (shot, config) metrics ----
    csv = ["shot,config,kind,sim_to_reference,sim_to_own"]
    by_ref = defaultdict(list)
    by_own = defaultdict(list)
    for s in shots:
        for c in configs:
            p = img.get((s, c))
            if not (p and os.path.exists(p)):
                continue
            s_ref = sim(p, ref)
            s_own = sim(p, own[s]) if (own[s] and os.path.exists(own[s])) else float("nan")
            csv.append(f'"{s}","{c}",{kind.get(c,"")},{s_ref:.4f},{s_own:.4f}')
            by_ref[c].append(s_ref)
            by_own[c].append(s_own)

    # ---- inter-shot per config ----
    inter = {}
    for c in configs:
        ps = [img.get((s, c)) for s in shots]
        ps = [p for p in ps if p and os.path.exists(p)]
        pair = [sim(ps[i], ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps))]
        inter[c] = sum(pair) / len(pair) if pair else float("nan")

    # ---- per-config means ----
    csv.append("")
    csv.append("config,kind,mean_sim_to_reference,mean_sim_to_own,mean_inter_shot")
    means = {}
    for c in configs:
        mr = sum(by_ref[c]) / len(by_ref[c]) if by_ref[c] else float("nan")
        mo = sum(by_own[c]) / len(by_own[c]) if by_own[c] else float("nan")
        means[c] = (mr, mo, inter[c])
        csv.append(f'"{c}",{kind.get(c,"")},{mr:.4f},{mo:.4f},{inter[c]:.4f}')

    cpath = os.path.join(args.out, "structured_metrics.csv")
    open(cpath, "w").write("\n".join(csv) + "\n")
    print(f"[ok] {cpath}\n")
    print("config            kind    concept(sim->ref)  content-kept(sim->own)  inter-shot")
    for c in configs:
        mr, mo, mi = means[c]
        print(f"  {c:<15} {kind.get(c,''):<7} {mr:>10.3f}        {mo:>10.3f}          {mi:>8.3f}")

    # ---- the money figure: trade-off scatter ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.4, 6.0))
        # shade the "good" region: high concept consistency AND high content kept
        ax.axhspan(0.78, 1.0, xmin=0.0, xmax=1.0, color="#e8f5e9", zorder=0)

        # scalar frontier (the collapse trade-off): connect scalar configs sorted by concept
        scal = [c for c in configs if kind.get(c) == "scalar"]
        scal = sorted(scal, key=lambda c: means[c][0])
        if len(scal) >= 2:
            ax.plot([means[c][0] for c in scal], [means[c][1] for c in scal],
                    "o-", color="#c0392b", lw=2, ms=8, zorder=3,
                    label="scalar anchoring (the collapse trade-off)")
        for c in scal:
            mr, mo, _ = means[c]
            ax.annotate(c.replace("scalar_", ""), (mr, mo), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color="#c0392b")

        # block-wise points (Idea-1): big stars
        blk = [c for c in configs if kind.get(c) == "block"]
        if blk:
            ax.scatter([means[c][0] for c in blk], [means[c][1] for c in blk],
                       marker="*", s=420, color="#1565c0", edgecolor="white", lw=1.2,
                       zorder=4, label="block-wise anchoring (Idea-1)")
        for c in blk:
            mr, mo, _ = means[c]
            ax.annotate(c, (mr, mo), textcoords="offset points",
                        xytext=(8, 6), fontsize=9, fontweight="bold", color="#1565c0")

        ax.set_xlabel("concept consistency  =  similarity to reference  \u2192")
        ax.set_ylabel("content kept  =  similarity to the shot's own scene  \u2192")
        ax.set_title("Idea-1: can block-wise anchoring keep the concept\nwhile preserving each shot's content?")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.text(0.99, 0.99, "better\n(keep concept + keep content)", ha="right", va="top",
                fontsize=8, color="#2e7d32", transform=ax.transAxes)
        fig.tight_layout()
        tpath = os.path.join(args.out, "structured_tradeoff.png")
        fig.savefig(tpath, dpi=150)
        print(f"\n[ok] {tpath}")
    except Exception as e:
        print(f"\n[warn] trade-off plot skipped ({e})")

    # ---- visual grid: rows = shots, cols = [reference] + configs ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        cols = ["__reference__"] + configs
        nrow, ncol = len(shots), len(cols)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.0 * nrow))
        if nrow == 1:
            axes = axes.reshape(1, -1)
        for r, s in enumerate(shots):
            for cidx, c in enumerate(cols):
                ax = axes[r][cidx]
                ax.axis("off")
                if c == "__reference__":
                    path, title = ref, ("REFERENCE" if r == 0 else "")
                    sub = ""
                else:
                    path = img.get((s, c))
                    title = c if r == 0 else ""
                    if path and os.path.exists(path):
                        s_ref = sim(path, ref)
                        s_own = sim(path, own[s]) if own[s] else float("nan")
                        sub = f"ref {s_ref:.2f} | own {s_own:.2f}"
                    else:
                        sub = ""
                if path and os.path.exists(path):
                    ax.imshow(PILImage.open(path).convert("RGB"))
                    if kind.get(c) == "block":  # frame Idea-1 columns
                        for sp in ax.spines.values():
                            sp.set_visible(True); sp.set_color("#1565c0"); sp.set_linewidth(3)
                        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
                if title:
                    ax.set_title(title, fontsize=9, fontweight="bold")
                if sub:
                    ax.text(0.5, -0.08, sub, ha="center", va="top", fontsize=7.5,
                            transform=ax.transAxes, color="#333")
            axes[r][0].text(-0.1, 0.5, s, ha="right", va="center", fontsize=9,
                            rotation=90, transform=axes[r][0].transAxes)
        fig.suptitle("Block-wise (blue) vs scalar anchoring \u2014 per shot", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        gpath = os.path.join(args.out, "structured_grid.png")
        fig.savefig(gpath, dpi=130)
        print(f"[ok] {gpath}")
    except Exception as e:
        print(f"[warn] grid skipped ({e})")


if __name__ == "__main__":
    main()

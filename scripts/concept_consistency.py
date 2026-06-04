#!/usr/bin/env python3
"""
Concept-consistency metric  (THESIS EVALUATION CORE)
====================================================
Measures whether a RECURRING CONCEPT looks visually consistent across the shots
where it appears. This is the non-tautological evidence: it is an OUTCOME of the
rendered pixels, not a function of the policy's decision counts.

Primary embedder: DINOv2 (field standard for subject/object appearance similarity;
CLIP captures global semantics but misses object-detail consistency -> our §7 fix).
Secondary: CLIP (sanity / matches the original eval plan).

Calibration (addresses §7: cosine is poorly scaled for "same thing"):
  - same_concept  = mean pairwise sim over shots sharing a concept
  - cross_concept = mean pairwise sim over shots sharing NO concept (the FLOOR)
  Headline gap = same_concept - cross_concept. Anchoring should raise same_concept.

Run it TWICE for the before/after proof:
  baseline (no anchoring):  --images data/intermediate/<id>/phase4/never_chain/images
  after  (concept-anchored): --images data/intermediate/<id>/phase4/concept_anchor_canonical/images
"""
import argparse, json, re, glob, os, itertools, random
from collections import defaultdict

def normalize_concept(e):
    e = re.sub(r"[^a-z0-9 ]", "", e.lower().strip())
    return re.sub(r"\s+", " ", e)

# ---------------- embedders ----------------
def embed_dinov2(paths):
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    out = {}
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            inp = proc(images=img, return_tensors="pt").to(dev)
            feat = model(**inp).last_hidden_state[:, 0]          # CLS token
            v = torch.nn.functional.normalize(feat, dim=-1).cpu().squeeze(0)
            out[p] = v.numpy()
    return out

def embed_clip(paths):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    out = {}
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            inp = proc(images=img, return_tensors="pt").to(dev)   # default center-crop; see note
            feat = model.get_image_features(**inp)
            v = torch.nn.functional.normalize(feat, dim=-1).cpu().squeeze(0)
            out[p] = v.numpy()
    return out


def clip_faithfulness(path_prompt_pairs):
    """CLIP image<->text alignment per shot: does the rendered image still match ITS OWN
    prompt? Drops if anchoring over-rides the prompt (collapse toward the reference).
    Counter-metric to directed fidelity. Always uses CLIP (DINOv2 has no text encoder)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    out = {}
    with torch.no_grad():
        for sid, path, prompt in path_prompt_pairs:
            img = Image.open(path).convert("RGB")
            ii = proc(images=img, return_tensors="pt").to(dev)
            ti = proc(text=[prompt[:300]], return_tensors="pt", padding=True, truncation=True).to(dev)
            vi = torch.nn.functional.normalize(model.get_image_features(**ii), dim=-1)
            vt = torch.nn.functional.normalize(model.get_text_features(**ti), dim=-1)
            out[sid] = float((vi * vt).sum())
    return out

def embed_mock(paths):
    """Deterministic pseudo-embeddings from filename hash (numpy). PLUMBING TEST ONLY."""
    import numpy as np, hashlib
    out = {}
    for p in paths:
        h = int(hashlib.sha256(os.path.basename(p).encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(h).standard_normal(64)
        out[p] = v / np.linalg.norm(v)
    return out

EMBED = {"dinov2": embed_dinov2, "clip": embed_clip, "mock": embed_mock}

def cos(a, b):
    import numpy as np
    return float(np.dot(a, b))

def mean_pairwise(emb, shots):
    sims = [cos(emb[a], emb[b]) for a, b in itertools.combinations(shots, 2)]
    return sum(sims) / len(sims) if sims else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboard", required=True)
    ap.add_argument("--images", required=True, help="dir with shot_NNN.png")
    ap.add_argument("--backend", default="dinov2", choices=list(EMBED))
    ap.add_argument("--ext", default="png")
    ap.add_argument("--floor-pairs", type=int, default=300)
    ap.add_argument("--anchors", default=None, help="storyboard_with_anchors.json -> enables DIRECTED anchor fidelity")
    ap.add_argument("--faithfulness", action="store_true", help="(secondary) CLIP image<->own-prompt alignment")
    ap.add_argument("--baseline-images", default=None, help="baseline dir (e.g. never_chain) -> CONTENT PRESERVATION = sim(this shot, its own baseline)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    story = json.load(open(args.storyboard))
    shots = story["shots"]
    concepts_by_shot = {s["shot_id"]: set(c for c in (normalize_concept(e)
                        for e in s.get("key_entities", [])) if c) for s in shots}

    # resolve image path per shot; skip missing
    path_by_shot = {}
    for s in shots:
        p = os.path.join(args.images, f"{s['shot_id']}.{args.ext}")
        if os.path.exists(p):
            path_by_shot[s["shot_id"]] = p
    missing = [s["shot_id"] for s in shots if s["shot_id"] not in path_by_shot]
    present = list(path_by_shot)
    if len(present) < 2:
        print("ERROR: <2 images found in", args.images); return

    emb = EMBED[args.backend](list(path_by_shot.values()))
    emb = {sid: emb[path_by_shot[sid]] for sid in present}


    # CONTENT PRESERVATION (primary collapse check, DINOv2-sensitive):
    # sim(anchored shot, its OWN baseline render). High = kept own content;
    # low = drifted toward the reference (copying). Directly catches the drift
    # that coarse CLIP image<->prompt misses.
    content_pres = None
    if args.baseline_images:
        base_paths = {}
        for sid in present:
            bp = os.path.join(args.baseline_images, f"{sid}.{args.ext}")
            if os.path.exists(bp):
                base_paths[sid] = bp
        if base_paths:
            base_emb_raw = EMBED[args.backend](list(base_paths.values()))
            base_emb = {sid: base_emb_raw[base_paths[sid]] for sid in base_paths}
            anchored_ids = set()
            if args.anchors:
                _ad = {a["shot_id"]: a for a in json.load(open(args.anchors))["shots"]}
                anchored_ids = {sid for sid in base_emb if _ad.get(sid, {}).get("anchor_source")}
            cp = {sid: cos(emb[sid], base_emb[sid]) for sid in base_emb}
            anc = [cp[s] for s in cp if s in anchored_ids]
            content_pres = {
                "overall_mean": sum(cp.values()) / len(cp),
                "anchored_mean": (sum(anc) / len(anc)) if anc else None,
                "per_shot": {k: round(v, 4) for k, v in sorted(cp.items(), key=lambda kv: kv[1])},
            }

    # recurring concepts present in the rendered set
    occ = defaultdict(list)
    for sid in present:
        for c in concepts_by_shot[sid]:
            occ[c].append(sid)
    recurring = {c: v for c, v in occ.items() if len(v) >= 2}

    per_concept = {}
    for c, sids in recurring.items():
        per_concept[c] = {"n": len(sids), "sim": mean_pairwise(emb, sids)}

    # aggregate same-concept consistency
    vals = [d["sim"] for d in per_concept.values() if d["sim"] is not None]
    same_unweighted = sum(vals) / len(vals) if vals else None
    wsum = sum(per_concept[c]["sim"] * per_concept[c]["n"] for c in per_concept if per_concept[c]["sim"] is not None)
    wden = sum(per_concept[c]["n"] for c in per_concept if per_concept[c]["sim"] is not None)
    same_weighted = wsum / wden if wden else None

    # cross-concept FLOOR: random pairs sharing NO concept
    random.seed(42)
    cross = []
    tries = 0
    while len(cross) < args.floor_pairs and tries < args.floor_pairs * 50:
        a, b = random.sample(present, 2); tries += 1
        if concepts_by_shot[a] & concepts_by_shot[b]:
            continue
        cross.append(cos(emb[a], emb[b]))
    floor = sum(cross) / len(cross) if cross else None

    # DIRECTED anchor fidelity: for each anchored shot, sim(shot, its anchor_source).
    # This is the clean diagnostic (does the shot actually move toward its reference?),
    # unlike the aggregate which mixes in concepts a shot was NOT anchored to.
    directed = None
    if args.anchors:
        adoc = {a["shot_id"]: a for a in json.load(open(args.anchors))["shots"]}
        pairs = []
        for sid in present:
            a = adoc.get(sid, {})
            src = a.get("anchor_source")
            if src and src in emb:
                pairs.append((sid, src, cos(emb[sid], emb[src])))
        if pairs:
            directed = {
                "n_anchored_pairs": len(pairs),
                "mean_sim_to_anchor": sum(p[2] for p in pairs) / len(pairs),
                "pairs": [{"shot": s, "anchor": a, "sim": round(v, 4)} for s, a, v in
                          sorted(pairs, key=lambda p: p[2])],
            }

    # FAITHFULNESS (collapse check): image vs its OWN prompt, focused on anchored shots.
    faithfulness = None
    if args.faithfulness:
        prompt_by_shot = {s["shot_id"]: s.get("image_prompt", "") for s in shots}
        pairs = [(sid, path_by_shot[sid], prompt_by_shot.get(sid, "")) for sid in present
                 if prompt_by_shot.get(sid)]
        fmap = clip_faithfulness(pairs)
        anchored_ids = set()
        if args.anchors:
            adoc = {a["shot_id"]: a for a in json.load(open(args.anchors))["shots"]}
            anchored_ids = {sid for sid in present if adoc.get(sid, {}).get("anchor_source")}
        anc = [fmap[s] for s in fmap if s in anchored_ids]
        faithfulness = {
            "overall_mean": sum(fmap.values()) / len(fmap) if fmap else None,
            "anchored_mean": (sum(anc) / len(anc)) if anc else None,
            "per_shot": {k: round(v, 4) for k, v in sorted(fmap.items(), key=lambda kv: kv[1])},
        }

    report = {
        "images_dir": args.images, "backend": args.backend,
        "n_images": len(present), "n_missing": len(missing), "missing": missing,
        "n_recurring_concepts": len(recurring),
        "same_concept_consistency_unweighted": same_unweighted,
        "same_concept_consistency_weighted": same_weighted,
        "cross_concept_floor": floor,
        "calibrated_gap": (same_weighted - floor) if (same_weighted is not None and floor is not None) else None,
        "per_concept": dict(sorted(per_concept.items(), key=lambda kv: (kv[1]["sim"] if kv[1]["sim"] is not None else 9))),
        "directed_anchor_fidelity": directed,
        "faithfulness": faithfulness,
        "content_preservation": content_pres,
    }

    print("="*72)
    print(f"CONCEPT CONSISTENCY  [{args.backend}]  dir={args.images}")
    print(f"images={len(present)} (missing {len(missing)})  recurring concepts={len(recurring)}")
    print(f"  same-concept (weighted)  = {same_weighted:.4f}" if same_weighted is not None else "  n/a")
    print(f"  cross-concept FLOOR      = {floor:.4f}" if floor is not None else "  floor n/a")
    if report["calibrated_gap"] is not None:
        print(f"  CALIBRATED GAP           = {report['calibrated_gap']:+.4f}   (higher = concepts more consistent than chance)")
    if content_pres is not None:
        am = content_pres["anchored_mean"]
        print(f"  CONTENT PRESERVATION (anchored) = {am:.4f}   (1.0=kept own content; low=copied reference)"
              if am is not None else "  content preservation n/a")
        print("    --- most-drifted anchored shots (lowest = most likely over-anchored/collapsed):")
        drifted = [(k, v) for k, v in content_pres["per_shot"].items()]
        for k, v in drifted[:6]:
            print(f"       {k}  content_pres={v:.4f}")
    if directed is not None:
        print(f"  DIRECTED anchor fidelity = {directed['mean_sim_to_anchor']:.4f}  "
              f"(mean sim of each anchored shot to its reference; n={directed['n_anchored_pairs']})")
        print("    --- weakest-pulled anchored shots:")
        for p in directed["pairs"][:6]:
            print(f"       {p['shot']} -> {p['anchor']}  sim={p['sim']:.4f}")
    if faithfulness is not None:
        am = faithfulness["anchored_mean"]; om = faithfulness["overall_mean"]
        print(f"  FAITHFULNESS (CLIP img<->own-prompt):  anchored={am:.4f}" if am is not None else "  faithfulness n/a",
              f" overall={om:.4f}" if om is not None else "")
        print("    (compare baseline vs anchored: a DROP = over-anchoring / collapse)")
    print("  --- least-consistent recurring concepts (most headroom for anchoring):")
    for c, d in list(report["per_concept"].items())[:8]:
        if d["sim"] is not None:
            print(f"     {c:32s} n={d['n']}  sim={d['sim']:.4f}")
    print("="*72)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("wrote", args.out)

if __name__ == "__main__":
    main()

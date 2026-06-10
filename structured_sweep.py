#!/usr/bin/env python3
"""
structured_sweep.py -- Idea-1 generator: STRUCTURED (block-wise) anchoring vs scalar.

This is the "solution" experiment. The reward-collapse finding showed that SCALAR
anchoring couples two things inversely: to raise similarity-to-reference (concept
consistency) you must destroy the shot's own content (context). Idea-1 tests whether
injecting the IP-Adapter reference into only SPECIFIC SDXL transformer blocks
(InstantStyle-style) breaks that coupling -- keeping the concept recognizable while
each shot keeps its OWN scene.

It re-generates the SAME shots under several NAMED configs, holding everything else
fixed (same prompt, same seed per shot, same reference, same model/LoRA/steps/CFG,
forced CONCEPT_ANCHOR). The ONLY thing that changes per config is the IP-Adapter
*scale*, which can be:
  - a float  -> uniform scalar anchoring (the baseline we already studied), or
  - a dict   -> per-block scale, e.g. {"up": {"block_0": [0,1,0]}} (block-wise).

WHY THIS PLUGS IN WITH NO PATCH:
  image_gen.generate_image() reads config["phase4"]["image_gen"]["concept_anchor_ref_weight"]
  and passes it STRAIGHT to pipe.set_ip_adapter_scale(...) with no numeric op. diffusers
  0.36 accepts a dict there natively (verified by probe), so we just set that field to a
  float OR a dict per config and reuse the exact same generation path as the thesis.

BLOCK MAP (SDXL, confirmed by attn-processor probe; lengths = #attentions in that block):
  up_blocks.0.attentions.1   -> STYLE / look      ->  {"up":   {"block_0": [0.0, 1.0, 0.0]}}
  down_blocks.2.attentions.1 -> LAYOUT / structure ->  {"down": {"block_2": [0.0, 1.0]}}
  (these two are the InstantStyle "style" and "layout" injection points.)

Outputs:
  <out>/<shot_id>__<config>.png    for each (shot x config)
  <out>/manifest.json              feed this to structured_compare.py

Example:
  python structured_sweep.py \
    --config config.yaml \
    --storyboard data/intermediate/<vid>/phase4/storyboard.json \
    --reference  data/intermediate/<vid>/phase4/concept_anchor_canonical_w02/images/<canonical>.png \
    --shots shot_005,shot_011,shot_013 \
    --out data/intermediate/<vid>/phase4/ide1_structured
"""
import argparse
import json
import os
import sys

import transformers.activations
if not hasattr(transformers.activations, 'PytorchGELUTanh'):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUActivation

sys.modules['gptqmodel'] = None
sys.modules['awq'] = None
sys.modules['bitsandbytes'] = None
import peft.import_utils
peft.import_utils.is_auto_awq_available = lambda: False
peft.import_utils.is_gptqmodel_available = lambda: False
peft.import_utils.is_auto_gptq_available = lambda: False


# ----- the configs we compare. Edit here, or pass --configs <file.json> to override. -----
# Scale is a float (uniform scalar) OR a dict (per-block, InstantStyle-style).
DEFAULT_CONFIGS = {
    # --- scalar baselines: these reproduce the collapse trade-off, the thing to beat ---
    "scalar_w0.0":  0.0,   # no anchor  -> the shot's OWN scene (content-preservation baseline)
    "scalar_w0.4":  0.4,   # scalar, moderate
    "scalar_w0.6":  0.6,   # scalar, where content has clearly collapsed
    # --- block-wise (Idea-1): inject only specific blocks, full strength on the chosen block ---
    "style_only":   {"up":   {"block_0": [0.0, 1.0, 0.0]}},                 # keep the look, free the layout
    "layout_only":  {"down": {"block_2": [0.0, 1.0]}},                      # keep the structure, free the look
    "style+layout": {"down": {"block_2": [0.0, 1.0]},
                     "up":   {"block_0": [0.0, 1.0, 0.0]}},                 # InstantStyle full
}
# which config is each shot's "own scene" reference for content-preservation:
OWN_BASELINE = "scalar_w0.0"


def _kind(scale):
    return "scalar" if isinstance(scale, (int, float)) else "block"


def load_config(path):
    with open(path) as fh:
        text = fh.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def load_storyboard_shots(path):
    with open(path) as fh:
        sb = json.load(fh)
    shots = sb.get("shots", sb) if isinstance(sb, dict) else sb
    out = {}
    for s in shots:
        sid = s.get("shot_id") or s.get("id")
        if sid:
            out[sid] = s.get("image_prompt") or s.get("visual_description", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the SAME config your pipeline uses")
    ap.add_argument("--storyboard", required=True)
    ap.add_argument("--reference", required=True, help="concept-anchor reference image (the canonical one)")
    ap.add_argument("--shots", default="", help="comma list of shot_ids; default = first --n")
    ap.add_argument("--n", type=int, default=3, help="how many shots if --shots not given")
    ap.add_argument("--configs", default="", help="optional JSON file: {name: scale} overriding the defaults")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())  # so `import image_gen` finds your module
    from PIL import Image
    from src.phase4 import image_gen  # your real generator

    configs = DEFAULT_CONFIGS
    if args.configs.strip():
        with open(args.configs) as fh:
            configs = json.load(fh)
    if OWN_BASELINE not in configs:
        sys.exit(f"[err] configs must include the own-scene baseline '{OWN_BASELINE}' (a 0.0 scalar)")

    cfg = load_config(args.config)
    prompts = load_storyboard_shots(args.storyboard)
    if not prompts:
        sys.exit("[err] no shots found in storyboard")

    if args.shots.strip():
        shot_ids = [s.strip() for s in args.shots.split(",") if s.strip()]
    else:
        shot_ids = list(prompts.keys())[: args.n]
    missing = [s for s in shot_ids if s not in prompts]
    if missing:
        sys.exit(f"[err] shot_ids not in storyboard: {missing}\n      available: {list(prompts.keys())}")

    os.makedirs(args.out, exist_ok=True)
    ref_img = Image.open(args.reference).convert("RGB")

    print(f"[info] shots={shot_ids}")
    print(f"[info] configs={list(configs.keys())}")
    print("[info] loading pipeline (SDXL + LoRA + IP-Adapter, once) ...")
    pipe, _ = image_gen.load_pipeline(cfg)

    ig = cfg.setdefault("phase4", {}).setdefault("image_gen", {})

    items = []
    try:
        for sid in shot_ids:
            prompt = prompts[sid]
            for name, scale in configs.items():
                # the ONLY thing we change: scalar OR dict -> goes straight to set_ip_adapter_scale
                ig["concept_anchor_ref_weight"] = scale
                shot = {"id": sid, "image_prompt": prompt}  # seed derived from id -> constant across configs
                print(f"[gen] {sid}  config={name}  scale={scale}")
                img = image_gen.generate_image(
                    pipe, shot, decision="CONCEPT_ANCHOR",
                    config=cfg, ref_image=ref_img, seed=None,
                )
                fn = os.path.join(args.out, f"{sid}__{name}.png")
                img.save(fn)
                items.append({"shot": sid, "config": name, "kind": _kind(scale),
                              "scale": scale, "image": os.path.abspath(fn)})
    finally:
        try:
            image_gen.unload_pipeline(pipe)
        except Exception:
            pass

    manifest = {
        "reference": os.path.abspath(args.reference),
        "own_baseline": OWN_BASELINE,
        "configs": list(configs.keys()),
        "config_kind": {n: _kind(s) for n, s in configs.items()},
        "shots": shot_ids,
        "items": items,
    }
    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[ok] wrote {len(items)} images to {args.out}")
    print(f"[ok] wrote {mpath}")
    print(f"[next] python structured_compare.py --manifest {mpath} --out {args.out}")


if __name__ == "__main__":
    main()

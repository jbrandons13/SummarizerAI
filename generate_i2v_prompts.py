#!/usr/bin/env python3
"""
generate_i2v_prompts.py  --  Phase 4.5: per-shot MOTION prompts for I2V.

WHY THIS EXISTS
  The still (T2I) prompt describes a static scene. Feeding that same prompt to
  the image-to-video model under-specifies motion -> bland or warped output.
  This step generates a SEPARATE, motion-specific prompt per shot (what moves,
  how, plus camera), grounded in the still's own content so Wan I2V animates the
  existing frame instead of hallucinating new objects.

PIPELINE POSITION
  phase2 (summary_script.json) -> storyboard.py (storyboard.json: image_prompt)
  -> [THIS] adds `i2v_prompt` per shot -> render_summary_video.py uses it for I2V.
  (render now reads `i2v_prompt` if present, else falls back to `image_prompt`.)

DESIGN NOTE
  Uses ONE LLM call for ALL shots (output: JSON {id: motion_prompt}). This matters
  for the LOCAL backend: LocalBackend.generate() loads then unloads the model on
  every call, so a per-shot loop would reload the 14B model once per shot. One
  batched call = one load. Parsing is robust with a safe per-shot fallback.

USAGE (run from repo root so `src/` imports resolve)
  # Local Qwen (default, same LLM as phase2) -- preview first, writes nothing:
  python generate_i2v_prompts.py \
    --storyboard data/.../phase4/storyboard.json \
    --model <your-exact-qwen-model-from-config> --dry-run

  # when prompts look good, drop --dry-run to write storyboard_i2v.json:
  python generate_i2v_prompts.py --storyboard data/.../phase4/storyboard.json \
    --model <your-exact-qwen-model-from-config>

OUTPUT
  Writes storyboard_i2v.json next to the input (each shot gains `i2v_prompt`),
  and prints each still-prompt -> motion-prompt pair for review.
  Then render with:  render_summary_video.py --storyboard storyboard_i2v.json ...
  (or pass --in-place to overwrite the original; a .bak backup is written.)
"""
import argparse, json, os, sys, re

SYSTEM_PROMPT = """You write a short, SCENE-SPECIFIC motion prompt for an image-to-video model (Wan I2V) that animates an EXISTING flat 2D illustrated still. The still is already drawn; you only describe how it moves. Hit TWO goals at once:
(A) NATURAL, non-destructive motion -- the drawing must never morph, melt, or distort.
(B) Motion specific to THIS shot -- never a generic template reused across shots.

FIRST read the shot's scene and pick what would realistically move IN IT. Match movers to the actual setting (this is a mapping to reason with, NOT phrases to copy):
- volcano / magma / fire -> lava surface glow and ripple, rising smoke, heat haze
- river / lake / ocean / coast -> water ripples, gentle waves, light glinting on the surface
- mountains / cliffs / canyon -> clouds drifting past, rolling haze or mist
- desert / sand / sediment / dry ground -> blowing sand, drifting dust, heat shimmer over the ground
- forest / grassland / plants -> grass and leaves swaying, slow cloud shadows moving across
- sky / globe / map of Earth / space -> clouds drifting across, faint atmospheric glow or shimmer
Name the SPECIFIC element from this shot. Never fall back to "rock layers" or "the landscape" as a catch-all mover.

HARD RULES:
1. Keep every solid structure perfectly still: rocks, mountains, cliffs, ground, terrain, buildings, the subject's outline. They must NOT move, grow, melt, crack, pile up, or change shape.
2. NO transformation or process: nothing forms, becomes, cools into, builds up, erodes, transforms, or changes identity -- these warp the image. Describe the present instant, never a process over time.
3. Animate ONLY ambient / fluid / light elements (smoke, clouds, haze, mist, water, dust, glow, shimmer, sway). NEVER add new discrete objects (no birds, people, vehicles) that are not already drawn.
4. Each prompt = 1-2 fitting movers + ONE gentle camera move. VARY the camera move across shots: slow push-in, slow pan left, slow pan right, or slow tilt -- pick what suits the composition; do NOT always push-in.
5. Subtle and slow. Max ~20 words, present-tense, concrete.

VARIETY IS MANDATORY: every shot must read clearly differently -- different movers and/or camera move, different wording. Never repeat a motion sentence or its opening words across shots. If two shots show a similar place, still differentiate them.
Only if a scene genuinely has nothing that can move: "slow camera push-in, faint ambient light shimmer".

OUTPUT: one valid JSON object mapping each shot id (exactly as given) to its motion prompt string. JSON only -- no markdown, no preamble, no commentary.
Format example (do NOT copy this wording): {"shot_A": "clouds drift past the still peaks, light glints on the river below, slow pan right", "shot_B": "heat haze shimmers over the lava field, thin smoke rises, slow push-in"}"""

def build_user_prompt(shots, id_of):
    lines = ["Write a motion prompt for each of these shots:", ""]
    for s in shots:
        img = (s.get("image_prompt") or "").strip()
        extra = (s.get("text") or s.get("caption") or "").strip()
        line = f"- {id_of(s)}: {img}"
        if extra:
            line += f"  [concept: {extra}]"
        lines.append(line)
    lines.append("")
    lines.append("Return ONLY the JSON object {id: motion_prompt}.")
    return "\n".join(lines)

def extract_json(resp):
    resp = (resp or "").strip()
    m = re.search(r"```json\s*(.*?)\s*```", resp, re.DOTALL) or re.search(r"```\s*(.*?)\s*```", resp, re.DOTALL)
    if m:
        block = m.group(1)
    else:
        s, e = resp.find("{"), resp.rfind("}")
        block = resp[s:e + 1] if (s != -1 and e != -1) else "{}"
    try:
        return json.loads(block)
    except Exception:
        out = {}
        for k, v in re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', block):
            out[k] = v
        return out

def clean(v):
    if not isinstance(v, str):
        return ""
    return v.strip().strip('"').strip("'").strip()

def build_backend(args):
    if args.backend == "groq":
        from src.models.llm_wrapper import GroqBackend
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            sys.exit("GROQ_API_KEY not set. Export it or use --backend local.")
        return GroqBackend(api_key=key, model_name=args.model)
    # local Qwen on GPU -- same LLM as phase2
    from src.models.llm_wrapper import LocalBackend
    from src.utils.vram import VRAMManager
    try:
        vram = VRAMManager()
    except TypeError as e:
        sys.exit("VRAMManager() needs constructor args in your project (e.g. VRAMManager(device_id=0)). "
                 "Edit build_backend() in this script to match how your pipeline builds it. (%s)" % e)
    return LocalBackend(vram, args.model)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboard", required=True)
    ap.add_argument("--out", default=None, help="output path (default: storyboard_i2v.json next to input)")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input storyboard (a .bak backup is written)")
    ap.add_argument("--backend", choices=["local", "groq"], default="local")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ",
                    help="use YOUR exact model string/path from phase2 config")
    ap.add_argument("--field", default="i2v_prompt", help="field name to write the motion prompt into")
    ap.add_argument("--max-new-tokens", type=int, default=1536, help="raise if you have many shots")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, write nothing")
    ap.add_argument("--only", nargs="*", help="limit to these shot ids (for a quick smoke test)")
    a = ap.parse_args()

    sb = json.load(open(a.storyboard))
    shots_all = sb["shots"] if isinstance(sb, dict) and "shots" in sb else sb
    id_of = lambda s: str(s.get("shot_id") or s.get("id"))
    only = [str(x) for x in a.only] if a.only else None
    shots = [s for s in shots_all if (not only or id_of(s) in only)]
    if not shots:
        sys.exit("no shots matched.")

    backend = build_backend(a)
    resp = backend.generate(SYSTEM_PROMPT, build_user_prompt(shots, id_of), max_new_tokens=a.max_new_tokens)
    mapping = extract_json(resp)

    n = 0
    for s in shots:
        sid = id_of(s)
        mp = clean(mapping.get(sid) or mapping.get(sid.lstrip("0")) or "")
        if not mp:
            mp = "slow camera push-in, subtle ambient motion, minimal movement"
            print("[WARN] no motion prompt parsed for %s; using fallback" % sid, file=sys.stderr)
        s[a.field] = mp
        n += 1
        print("\n[%s]" % sid)
        print("  still : %s" % ((s.get("image_prompt") or "")[:100]))
        print("  motion: %s" % mp)

    print("\nGenerated motion prompts for %d shots (single LLM call)." % n)
    if a.dry_run:
        print("(dry-run: no file written -- review the pairs above, then re-run without --dry-run)")
        return

    if a.in_place:
        bak = a.storyboard + ".bak"
        if not os.path.exists(bak):
            json.dump(sb, open(bak, "w"), indent=2, ensure_ascii=False)
            print("backup written: %s" % bak)
        out = a.storyboard
    else:
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.storyboard)), "storyboard_i2v.json")
    json.dump(sb, open(out, "w"), indent=2, ensure_ascii=False)
    print("written: %s" % out)
    if not a.in_place:
        print("-> render with: python render_summary_video.py --storyboard %s ..." % out)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Phase 5 orchestrator: turn the w0.2 stills into the final narrated summary VIDEO.

Per shot (in storyboard order), drive the content-aware animation policy:
  - method == "i2v"       -> Wan 2.1 I2V (DiT) via ComfyUI API  (dynamic-concept shots)
  - method == "ken_burns" -> ffmpeg zoom (ken_burns.py)         (static-object shots)
Then mux each shot's voiceover wav, and concat all clips -> one final mp4.

Design notes:
  * Wan length must be 4n+1 -> snapped UP so video >= audio; -shortest trims to audio.
  * Deterministic seed per shot (reproducible).
  * --resume skips shots whose *_av.mp4 already exists (I2V is hours; survive interrupts).
  * ComfyUI model stays loaded across /prompt calls, so only the 1st I2V pays load.

Usage (smoke first!):
  python render_summary_video.py --only shot_004 shot_016 --dry-run     # inspect plan + patch
  python render_summary_video.py --only shot_004 shot_016               # 1 KB + 1 I2V, real
  python render_summary_video.py                                        # full 44 -> final.mp4
"""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time, urllib.request, wave, contextlib

ID = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
BASE = f"data/intermediate/{ID}/phase4"

def seed_of(sid): return int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % (2**31)
def wav_seconds(p):
    with contextlib.closing(wave.open(p, "r")) as w: return w.getnframes()/float(w.getframerate())
def snap_len_up(frames):  # Wan needs length = 4n+1; round UP so video >= audio
    return max(5, int(math.ceil((frames - 1)/4.0))*4 + 1)
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# ---------- ComfyUI ----------
def comfy_submit(workflow, url):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(url + "/prompt", data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]

def comfy_wait(pid, url, timeout, poll=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(urllib.request.urlopen(url + f"/history/{pid}", timeout=30).read())
        except Exception:
            h = {}
        if pid in h and h[pid].get("outputs"):
            return h[pid]["outputs"]
        time.sleep(poll)
    raise TimeoutError(f"ComfyUI prompt {pid} timed out after {timeout}s")

def collect_frames(outputs):
    """Gather the PNG frames THIS prompt produced (from this prompt's history only,
    so stale frames from earlier attempts are never picked up). Sorted by filename:
    the zero-padded counter makes lexicographic order == frame order."""
    frames = []
    for node in outputs.values():
        for img in node.get("images", []):
            if img.get("filename", "").lower().endswith(".png"):
                frames.append((img["filename"], img.get("subfolder", "")))
    if not frames:
        raise RuntimeError(f"no png frames in outputs: {outputs}")
    frames.sort()
    return frames

def render_i2v(sid, still, prompt, frames, wf_template, args):
    wf = json.loads(json.dumps(wf_template))  # deep copy
    shutil.copy(still, os.path.join(args.comfy_input, f"{sid}.png"))
    wf["5"]["inputs"]["image"] = f"{sid}.png"
    wf["9"]["inputs"]["text"] = prompt
    wf["11"]["inputs"]["length"] = frames
    wf["11"]["inputs"]["width"], wf["11"]["inputs"]["height"] = args.w, args.h
    wf["12"]["inputs"]["seed"] = seed_of(sid)
    if args.steps: wf["12"]["inputs"]["steps"] = args.steps
    # ffmpeg 4.4 can't decode animated WEBP -> override output node to SaveImage
    # (built-in; no custom node) so ComfyUI emits a PNG sequence we assemble.
    wf["14"] = {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"clip_{sid}"}}
    pid = comfy_submit(wf, args.comfy_url)
    outputs = comfy_wait(pid, args.comfy_url, args.i2v_timeout)
    frame_files = collect_frames(outputs)
    tmp = os.path.join(args.work, f"_frames_{sid}")
    if os.path.isdir(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp)
    for i, (fn, sub) in enumerate(frame_files):
        shutil.copy(os.path.join(args.comfy_output, sub, fn), os.path.join(tmp, f"{i:05d}.png"))
    out = os.path.join(args.work, f"{sid}.mp4")
    run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(tmp, "%05d.png"),
         "-vf", f"scale={args.w}:{args.h}", "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
    shutil.rmtree(tmp, ignore_errors=True)
    return out

def render_ken_burns(sid, still, dur, motion, args):
    out = os.path.join(args.work, f"{sid}.mp4")
    run(["python", "ken_burns.py", "--image", still, "--duration", f"{dur}",
         "--fps", str(args.fps), "--size", f"{args.w}x{args.h}", "--motion", motion, "--out", out])
    return out

def mux(video, wav, out, args):
    run(["ffmpeg", "-y", "-i", video, "-i", wav, "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", out])

def burn_subtitles(av, wav, text, out, args):
    """Burn modern line-by-line subtitles synced to the shot's audio.
    On ANY failure, fall back to the un-subtitled clip so the video still renders."""
    try:
        run(["python", "subtitle_gen.py", "--text", text, "--audio", wav,
             "--in-video", av, "--out-video", out, "--size", f"{args.w}x{args.h}"])
        return out
    except Exception as e:
        print(f"  [subtitle skipped: {e}]")
        shutil.copy(av, out)
        return out

def load_narration(script_path, order):
    """Map shot_id -> spoken narration text from summary_script.json.
    Defensive: match by shot_id/id if they line up, else positional (storyboard is 1:1 with the script)."""
    texts = {o: "" for o in order}
    if not (script_path and os.path.exists(script_path)):
        print(f"  [no script at {script_path}; subtitles will be empty]")
        return texts
    data = json.load(open(script_path))
    segs = data.get("sentences") or data.get("segments") or data.get("shots") or (data if isinstance(data, list) else [])
    by_id = {}
    for s in segs:
        k = f"shot_{s.get('id'):03d}" if 'id' in s else s.get("shot_id")
        if k is not None:
            by_id[str(k)] = s.get("text", "")
    if any(o in by_id for o in order):            # shot_ids line up directly
        for o in order:
            texts[o] = by_id.get(o, "")
    else:                                          # positional fallback
        for i, o in enumerate(order):
            if i < len(segs):
                texts[o] = segs[i].get("text", "")
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="animation_plan.json")
    ap.add_argument("--storyboard", default=f"{BASE}/storyboard.json")
    ap.add_argument("--script", default=f"{BASE}/summary_script.json", help="narration source for subtitles")
    ap.add_argument("--no-subtitles", action="store_true", help="disable burned subtitles")
    ap.add_argument("--all-i2v", action="store_true", help="force every shot to I2V (full generative, no Ken Burns)")
    ap.add_argument("--images-dir", default=f"{BASE}/concept_anchor_canonical_w02/images")
    ap.add_argument("--audio-dir", default=f"{BASE}/audio")
    ap.add_argument("--work", default=f"{BASE}/phase5_clips")
    ap.add_argument("--final", default=f"{BASE}/summary_video.mp4")
    ap.add_argument("--workflow", default="wan_i2v_workflow.json")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-input", default=os.path.expanduser("~/comfyui/ComfyUI/input"))
    ap.add_argument("--comfy-output", default=os.path.expanduser("~/comfyui/ComfyUI/output"))
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--w", type=int, default=832)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--i2v-timeout", type=int, default=7200)
    ap.add_argument("--only", nargs="*", default=None, help="subset of shot_ids (smoke)")
    ap.add_argument("--force", action="store_true", help="re-render even if clip exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sb = json.load(open(args.storyboard))
    shots = sb.get("shots") or sb.get("segments") or sb.get("sentences") or (sb if isinstance(sb, list) else [])
    order = [f"shot_{s.get('id'):03d}" if 'id' in s else s.get("shot_id") for s in shots]
    prompt_by = { (f"shot_{s.get('id'):03d}" if 'id' in s else s.get("shot_id")): (s.get("i2v_prompt") or s.get("image_prompt") or (", ".join(s.get("keywords", [])) if s.get("keywords") else s.get("text", ""))) for s in shots }
    if args.only: order = [s for s in order if s in args.only]
    if args.all_i2v:                       # full generative: every shot animated, no Ken Burns
        method_by = {s: "i2v" for s in order}
    else:
        method_by = json.load(open(args.plan))["method_by_shot"]
    os.makedirs(args.work, exist_ok=True)
    if not args.dry_run: os.makedirs(args.comfy_input, exist_ok=True)
    wf_template = json.load(open(args.workflow)) if "i2v" in method_by.values() else {}
    # subtitles: per-shot narration text. shots.json (segmenter) holds shot_id -> the exact spoken
    # fragment for each shot; summary_script.json is sentence-level (wrong ids), so prefer shots.json.
    shots_json = os.path.join(os.path.dirname(os.path.abspath(args.storyboard)), "shots.json")
    sub_src = shots_json if os.path.exists(shots_json) else args.script
    narration = {} if args.no_subtitles else load_narration(sub_src, order)

    av_clips = []
    MOTIONS = ["zoom_in", "pan_right", "zoom_out", "pan_left", "pan_up", "pan_down", "drift"]
    for idx, sid in enumerate(order):
        method = method_by.get(sid, "ken_burns")
        still = os.path.join(args.images_dir, f"{sid}.png")
        wav = os.path.join(args.audio_dir, f"{sid}.wav")
        if not os.path.exists(wav):
            wav = os.path.join(args.audio_dir, f"sentence_{int(sid.split('_')[1]):03d}.wav")
        dur = wav_seconds(wav) if (os.path.exists(wav) and not args.dry_run) else None
        frames_i2v = snap_len_up(round((dur or 3.0) * args.fps))
        motion = MOTIONS[idx % len(MOTIONS)]
        av = os.path.join(args.work, f"{sid}_av.mp4")
        sub_text = narration.get(sid, "")
        final_clip = os.path.join(args.work, f"{sid}_final.mp4") if sub_text else av
        av_clips.append(final_clip)

        if args.dry_run:
            print(f"{sid:>9} | {method:<9} | dur={'?' if dur is None else round(dur,2)} | "
                  f"i2v_frames={frames_i2v} (4n+1) | motion={motion} | "
                  f"subs={'Y' if sub_text else 'N'} | prompt='{prompt_by.get(sid,'')[:40]}'")
            continue

        t0 = time.time()
        # expensive I2V: reuse an existing clip across runs; ken_burns is cheap -> always re-render
        if method == "i2v" and os.path.exists(av) and not args.force:
            print(f"{sid}: reuse existing i2v clip")
        else:
            if method == "i2v":
                vid = render_i2v(sid, still, prompt_by.get(sid, ""), frames_i2v, wf_template, args)
            else:
                vid = render_ken_burns(sid, still, dur, motion, args)
            mux(vid, wav, av, args)
        # subtitles: cheap, always (re)apply when narration text exists
        if sub_text:
            burn_subtitles(av, wav, sub_text, final_clip, args)
        print(f"{sid}: {method}{' +subs' if sub_text else ''} done in {time.time()-t0:.0f}s -> {final_clip}")

    if args.dry_run:
        print(f"\n[dry-run] {len(order)} shots; "
              f"i2v={sum(1 for s in order if method_by.get(s)=='i2v')}, "
              f"ken_burns={sum(1 for s in order if method_by.get(s)!='i2v')}")
        # verify workflow patch on a sample
        s0 = next((s for s in order if method_by.get(s)=='i2v'), None)
        if s0:
            wf = json.loads(json.dumps(wf_template))
            wf["5"]["inputs"]["image"]=f"{s0}.png"; wf["11"]["inputs"]["length"]=99; wf["12"]["inputs"]["seed"]=seed_of(s0)
            print(f"[patch check] node5.image={wf['5']['inputs']['image']} "
                  f"node11.length={wf['11']['inputs']['length']} node12.seed={wf['12']['inputs']['seed']}")
        return

    # concat (re-encode = robust to any param drift)
    lst = os.path.join(args.work, "concat.txt")
    with open(lst, "w") as f:
        for c in av_clips:
            if os.path.exists(c): f.write(f"file '{os.path.abspath(c)}'\n")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(args.fps),
         "-c:a", "aac", "-b:a", "192k", args.final])
    print(f"\nFINAL VIDEO -> {args.final}")

if __name__ == "__main__":
    main()
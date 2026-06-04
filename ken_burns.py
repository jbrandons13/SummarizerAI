#!/usr/bin/env python3
"""
Smooth, JITTER-FREE, VARIED camera motion on a still -> mp4.
Rewritten to use PIL sub-pixel cropping (float box in Image.resize) instead of
ffmpeg zoompan (which jitters). Each shot can get a different motion for variety.

Motions: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, drift (zoom+pan).
Duration from --duration or --audio (wav length).

Usage:
  python ken_burns.py --image s.png --audio s.wav --motion pan_right --out s.mp4
  python ken_burns.py --image s.png --duration 4.0 --motion drift --out s.mp4 --mux-audio --audio s.wav
"""
import argparse, contextlib, math, os, shutil, subprocess, sys, tempfile, wave
from PIL import Image

def wav_seconds(p):
    with contextlib.closing(wave.open(p, "r")) as w: return w.getnframes()/float(w.getframerate())

def _clamp_box(left, top, cw, chh, iw, ih):
    left = max(0.0, min(left, iw - cw)); top = max(0.0, min(top, ih - chh))
    return (left, top, left + cw, top + chh)

def render_frames(img_path, out_dir, frames, motion, w, h, zmax=1.18):
    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size
    aspect = w / h
    # largest centered crop of the target aspect that fits the source
    if iw / ih > aspect: bh = float(ih); bw = bh * aspect
    else:                bw = float(iw); bh = bw / aspect
    for n in range(frames):
        p = n / (frames - 1) if frames > 1 else 0.0
        z = 1.0
        if motion == "zoom_in":   z = 1 + (zmax - 1) * p
        elif motion == "zoom_out":z = zmax - (zmax - 1) * p
        elif motion == "drift":   z = 1 + (zmax - 1) * p
        else:                     z = 1.07  # gentle zoom so pans have headroom
        cw, chh = bw / z, bh / z
        cx, cy = iw / 2.0, ih / 2.0  # default centered
        rx, ry = (iw - cw), (ih - chh)
        if motion == "pan_right": cx = cw/2 + rx * p
        elif motion == "pan_left":cx = iw - cw/2 - rx * p
        elif motion == "pan_up":  cy = ih - chh/2 - ry * p
        elif motion == "pan_down":cy = chh/2 + ry * p
        elif motion == "drift":   cx = cw/2 + rx * p  # zoom + slow pan-right
        box = _clamp_box(cx - cw/2, cy - chh/2, cw, chh, iw, ih)
        img.resize((w, h), Image.LANCZOS, box=box).save(os.path.join(out_dir, f"{n:05d}.png"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--size", default="832x480")
    ap.add_argument("--motion", default="zoom_in",
                    choices=["zoom_in","zoom_out","pan_left","pan_right","pan_up","pan_down","drift"])
    ap.add_argument("--mux-audio", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dur = a.duration if a.duration is not None else (wav_seconds(a.audio) if a.audio else None)
    if dur is None: sys.exit("need --duration or --audio")
    w, h = (int(x) for x in a.size.lower().split("x"))
    frames = max(2, round(dur * a.fps))
    print(f"{os.path.basename(a.image)} | {a.motion} | {round(dur,2)}s -> {frames} frames @ {a.fps}fps (PIL sub-pixel, no jitter)")
    if a.dry_run: return
    tmp = tempfile.mkdtemp(prefix="kb_")
    try:
        render_frames(a.image, tmp, frames, a.motion, w, h)
        cmd = ["ffmpeg","-y","-framerate",str(a.fps),"-i",os.path.join(tmp,"%05d.png")]
        if a.mux_audio and a.audio: cmd += ["-i",a.audio,"-c:a","aac","-b:a","192k","-shortest"]
        cmd += ["-c:v","libx264","-pix_fmt","yuv420p",a.out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", a.out)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Modern, line-by-line subtitles synced to the narration, burned into a clip.
Shows ONE short line at a time (not a long block), timed across the shot's audio.

Style: clean bold sans, white text on a semi-transparent rounded-ish dark bar
(BorderStyle=3 box), bottom-center. Swap --font to Montserrat/Inter for extra polish.

Timing: lines are distributed proportional to word count across the audio duration.
(Hook: pass --word-timestamps later for exact forced-aligned timing.)

Usage:
  python subtitle_gen.py --text "Magma cools to form igneous rock." \
     --audio shot_016.wav --in-video shot_016.mp4 --out-video shot_016_sub.mp4
"""
import argparse, contextlib, os, re, subprocess, sys, tempfile, wave

def wav_seconds(p):
    with contextlib.closing(wave.open(p, "r")) as w: return w.getnframes()/float(w.getframerate())

def get_whisper_words(audio_path):
    try:
        from faster_whisper import WhisperModel
        # Use tiny.en for fast, accurate enough English word timestamps
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, word_timestamps=True)
        words = []
        for s in segments:
            for w in s.words:
                words.append((w.word, w.start, w.end))
        return words
    except Exception as e:
        print("Whisper error:", e)
        return None

def split_lines_whisper(words, max_words=6):
    lines, cur = [], []
    for w_text, start, end in words:
        cur.append((w_text, start, end))
        if len(cur) >= max_words or re.search(r"[.!?,;:]$", w_text):
            lines.append((" ".join([x[0].strip() for x in cur]), cur[0][1], cur[-1][2]))
            cur = []
    if cur:
        lines.append((" ".join([x[0].strip() for x in cur]), cur[0][1], cur[-1][2]))
    
    if len(lines) >= 2 and len(lines[-1][0].split()) < 3:
        lines[-2] = (lines[-2][0] + " " + lines[-1][0], lines[-2][1], lines[-1][2])
        lines.pop()
    return lines

def split_lines(text, max_words=6):
    words = text.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?,;:]$", w):
            lines.append(" ".join(cur)); cur = []
    if cur: lines.append(" ".join(cur))
    lines = [l.strip() for l in lines if l.strip()]
    if len(lines) >= 2 and len(lines[-1].split()) < 3:
        lines[-2] = lines[-2] + " " + lines[-1]; lines.pop()
    return lines

def ass_time(t):
    h = int(t//3600); m = int((t%3600)//60); s = t%60
    return f"{h}:{m:02d}:{s:05.2f}"

def build_ass(lines, dur, w, h, font, fontsize):
    total_words = sum(len(l.split()) for l in lines) or 1
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Modern,{font},{fontsize},&H00FFFFFF,&H00FFFFFF,&H00202020,&H80101010,-1,0,0,0,100,100,0,0,3,6,0,2,60,60,{int(h*0.07)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events, t = [], 0.0
    for l in lines:
        seg = dur * (len(l.split()) / total_words)
        start, end = t, min(dur, t + seg)
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Modern,,0,0,0,,{l}")
        t = end
    return head + "\n".join(events) + "\n"

def build_ass_whisper(lines_with_times, w, h, font, fontsize):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Modern,{font},{fontsize},&H00FFFFFF,&H00FFFFFF,&H00202020,&H80101010,-1,0,0,0,100,100,0,0,3,6,0,2,60,60,{int(h*0.07)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for text, start, end in lines_with_times:
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Modern,,0,0,0,,{text}")
    return head + "\n".join(events) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--in-video", required=True)
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--font", default="DejaVu Sans")
    ap.add_argument("--fontsize", type=int, default=None)
    ap.add_argument("--size", default="832x480")
    ap.add_argument("--max-words", type=int, default=6)
    ap.add_argument("--fontsdir", default=None, help="dir of custom font files")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    
    dur = a.duration if a.duration is not None else (wav_seconds(a.audio) if a.audio else None)
    if dur is None: sys.exit("need --duration or --audio")
    
    w, h = (int(x) for x in a.size.lower().split("x"))
    # Reduced font size from 0.085 to 0.05
    fontsize = a.fontsize or max(20, round(h * 0.05))
    
    words = get_whisper_words(a.audio) if a.audio else None
    
    if words:
        lines_with_times = split_lines_whisper(words, a.max_words)
        ass = build_ass_whisper(lines_with_times, w, h, a.font, fontsize)
        print(f"Whisper sync: {len(lines_with_times)} lines over {round(dur,2)}s | font={a.font} size={fontsize}")
        for l, s, e in lines_with_times: print(f"   • [{s:.2f}-{e:.2f}] {l}")
    else:
        lines = split_lines(a.text, a.max_words)
        ass = build_ass(lines, dur, w, h, a.font, fontsize)
        print(f"Fallback sync: {len(lines)} lines over {round(dur,2)}s | font={a.font} size={fontsize}")
        for l in lines: print(f"   • {l}")
        
    if a.dry_run:
        print("\n--- ASS preview ---\n" + ass); return
        
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False); tmp.write(ass); tmp.close()
    vf = f"ass={tmp.name}" + (f":fontsdir={a.fontsdir}" if a.fontsdir else "")
    subprocess.run(["ffmpeg","-y","-i",a.in_video,"-vf",vf,"-c:a","copy",a.out_video],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.unlink(tmp.name); print("wrote", a.out_video)

if __name__ == "__main__":
    main()

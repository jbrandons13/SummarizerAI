import subprocess
from pathlib import Path

def add_debug_annotation(
    clean_video_path: Path | str,
    annotated_video_path: Path | str,
    segments: list[dict],  # [{"start", "end", "action", "score", "group_id"}]
    bar_height: int = 40,
) -> None:
    """Add top-bar debug annotation to video. Subtitle in original remains untouched."""
    
    # Build drawtext filters: one per segment, gated by enable='between(t,start,end)'
    drawtext_parts = []
    
    # Background bar (semi-transparent black)
    bg_filter = (
        f"drawbox=x=0:y=0:w=iw:h={bar_height}:"
        f"color=black@0.6:t=fill"
    )
    drawtext_parts.append(bg_filter)
    
    for seg in segments:
        action = seg["action"]
        score = seg["score"]
        group_id = seg["group_id"]
        
        # Color: retrieve = cyan-ish, generate = orange
        color = "0x4FC3F7" if action == "RETRIEVE" else "0xFFB74D"
        
        text = f"[{action}] | Score\\: {score:.3f} | Group {group_id}"
        # Escape special chars for ffmpeg drawtext
        text_escaped = text.replace(":", "\\:").replace("'", "\\'")
        
        drawtext = (
            f"drawtext=text='{text_escaped}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf:"
            f"fontsize=18:"
            f"fontcolor={color}:"
            f"x=20:y=10:"
            f"enable='between(t,{seg['start']:.3f},{seg['end']:.3f})'"
        )
        drawtext_parts.append(drawtext)
    
    vf = ",".join(drawtext_parts)
    
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(clean_video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "copy",  # passthrough audio
        str(annotated_video_path),
    ]
    print("Running command:", " ".join(cmd))
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Error!")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
    else:
        print("Success!")

if __name__ == "__main__":
    clean = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/data/output/review_1/summary_grouping_gate.mp4"
    out = "/home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/scratch/test_annotated.mp4"
    segments = [
        {"start": 0.0, "end": 2.5, "action": "RETRIEVE", "score": 0.166, "group_id": 0},
        {"start": 2.5, "end": 5.0, "action": "GENERATE", "score": 0.084, "group_id": 1},
    ]
    add_debug_annotation(clean, out, segments)

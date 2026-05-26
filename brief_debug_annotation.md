# Brief: Add Debug-Annotated Output Variant

**Task type:** Execution (feature addition).
**Goal:** Tambahkan output variant dengan overlay annotation untuk debugging purposes. Output existing (clean) tetap dipertahankan. Generate **dua file output per video**:
1. `summary_grouping_gate.mp4` — clean (existing, no change)
2. `summary_grouping_gate_debug.mp4` — annotated dengan action + score di top bar

User locked specs:
- Purpose: debugging only (not for thesis demo)
- Score to show: `best_similarity` dari Phase 4 (weighted score yang dipake gate decision)
- Layout: top bar dengan action + score; subtitle existing tetap di bawah (jangan diganggu)
- Format: separate file (2 outputs per video), bukan replace

## Context

- Existing output: `data/output/{video_id}/summary_grouping_gate.mp4` dengan subtitle di bottom
- Phase 5 assembler udah punya logic concat + subtitle muxing
- `p4_assignments.json` punya field: `sentence_ids`, `scene_id`, `best_similarity`, `action`, `timestamp_hint_merged`, `similarity_trail`
- Per-segment timing diketahui dari audio durations (per group)

## Implementation

### Approach

Tambahkan logic di assembler (`src/phase5_assemble.py`) untuk generate annotated variant setelah clean output selesai. Annotated version = clean output + drawtext overlay di atas.

**Tidak perlu modify pipeline upstream.** Cuma tambah post-processing step di assembler.

### Top bar specification

Format: `[ACTION] | Score: X.XXX | Group N`

Contoh:
- `[RETRIEVE] | Score: 0.166 | Group 2`
- `[GENERATE] | Score: 0.084 | Group 6`

Style:
- Background: semi-transparent black bar di top (height ~40px)
- Text: white, mono font, ukuran ~20px
- Color hint: bisa color-code action (e.g. retrieve = light blue text, generate = orange text), atau simple white-on-black supaya minimal styling complexity
- Position: top-left padding, or center, sesuai readability

### Per-segment timing

Top bar text **berubah per segment** berdasarkan group yang sedang diputar. Timing tiap segment = audio duration tiap group, akumulasi mulai dari 0.

Compute timeline:
```python
segments = []  # list of {start_time, end_time, action, score, group_id}
current_time = 0.0
for i, group in enumerate(p4_assignments):
    duration = sum_audio_duration(group)  # already computed di assembler
    segments.append({
        "start": current_time,
        "end": current_time + duration,
        "action": group["action"].upper(),
        "score": group["best_similarity"],
        "group_id": i,
    })
    current_time += duration
```

### FFmpeg drawtext implementation

Pakai ffmpeg `drawtext` filter dengan `enable` expression untuk per-segment text switching.

Karena ada banyak segments per video, **pakai multiple drawtext filters dengan enable** atau pakai single drawtext dengan dynamic text from file. Multiple drawtext lebih simple debug.

Helper function:

```python
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
    
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(clean_video_path),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "copy",  # passthrough audio
            str(annotated_video_path),
        ],
        check=True, capture_output=True,
    )
```

**Important notes:**
- Font path `/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf` — verify exists. Kalau ga ada, cari font monospace alternatif (e.g. `fc-list | grep -i mono`) dan adjust path. Kalau ga ada mono font, fall back ke regular DejaVu.
- ffmpeg drawtext text escaping itu finicky. Test dengan 1 segment dulu sebelum loop semua.
- `enable='between(t,start,end)'` ngasih drawtext active cuma di rentang waktu itu.

### Integration in assembler

Di `src/phase5_assemble.py`, setelah generate clean output:

```python
# After successful clean output generation
clean_path = output_dir / f"summary_{method}.mp4"
debug_path = output_dir / f"summary_{method}_debug.mp4"

# Build segment timeline from p4_assignments + audio durations
segments_timeline = build_debug_segments_timeline(assignments, audio_manifest)

# Generate debug annotated variant
add_debug_annotation(clean_path, debug_path, segments_timeline)

logger.info(f"Debug-annotated output: {debug_path}")
```

`build_debug_segments_timeline` constructs the timeline dict array as specified above.

### Testing on review_1

Run pipeline on review_1 dengan changes. Verify:

1. **Clean output** `summary_grouping_gate.mp4` masih ada dan unchanged from before
2. **Debug output** `summary_grouping_gate_debug.mp4` exist
3. Compare ffprobe:
   - Same duration (within 0.1s tolerance)
   - Same audio stream
   - Video stream: same resolution + fps
4. Tonton debug output, verify:
   - Top bar visible at all times (background black bar)
   - Text changes per segment (action + score + group_id)
   - Color differs between RETRIEVE (cyan) and GENERATE (orange)
   - Subtitle existing tetap di bottom, ga terganggu

## Hard rules

- **Clean output (existing path) JANGAN diganggu.** Annotated is separate file.
- **Don't modify subtitle logic.** Existing subtitles stay where they are.
- **Don't claim quality.** Just report ffprobe metrics + paths.
- **If drawtext fails** (font missing, escape issues), report verbatim stderr.
- **Test on review_1 only first.** Don't auto-run on all 10 videos.

## What to report back

```
## Implementation
- Files modified: <list>
- Functions added: <list>
- Font used: <path>

## Verification on review_1
- Clean output path: <path> (size: X MB, duration: Y.YYs)
- Debug output path: <path> (size: X MB, duration: Y.YYs)
- Duration match: yes/no (delta: X.XXs)
- Audio stream identical: yes/no

## Segment timeline (for debug overlay)
[Table: group_id, start, end, action, score]

## User visual review needed
"Please review debug output: <path>"
```

## Anti-hallucination

- Verify font exists before claiming it works
- Quote ffprobe values verbatim
- If any drawtext segment fails to render, report which segment + reason
- If timeline computation produces negative durations or overlap, flag it

## Out of scope

- Modifying clean output
- Running on multiple videos (review_1 only first)
- Custom font installation
- Animation / fade transitions in overlay
- Adding more fields beyond action/score/group_id
- Removing existing subtitle

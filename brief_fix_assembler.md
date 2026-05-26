# Brief: Fix Phase 5 Assembler — Resolution Unification + Duration Handling

**Task type:** Execution (integrate provided code + verify).
**Goal:** Fix dua bugs di assembler yang menyebabkan freeze 50 detik di output review_1:
1. Resolution mismatch (LTX 768x512 vs source 640x320) → decoder crash di concat
2. Clip duration < audio duration → no padding logic

User locked decisions:
- **Resolution target:** unify ke source video resolution (640x320 untuk review_1, dynamic per video) dengan letterbox pad untuk preserve aspect ratio LTX content
- **Duration handling:** Hybrid strategy — slow playback dengan cap 0.6x minimum speed + freeze last frame untuk remainder

## Context

Per `brief_diagnose_output.md`:
- `src/phase5_assemble.py`: assembler ga handle clip < audio case, ga handle resolution mismatch
- `src/utils/ffmpeg_ops.py`: `cut_video_segment` ga ada scale logic, `concat_videos` pake `-c copy` (no re-encode)
- Generated LTX clips: hardcoded 768x512 di `src/phase5_ltx_runner.py` lines 229-230
- Symptom: freeze frame 50 detik di output post-LTX

**Important:** code dalam brief ini adalah **starting points**, mungkin perlu adjust ke conventions yang ada di codebase. JANGAN copy verbatim tanpa cek import & API consistency.

## Implementation

### Step 1: Add utility function untuk probe video duration

File: `src/utils/ffmpeg_ops.py` (atau tempat utility lain)

```python
import subprocess
import json
from pathlib import Path

def probe_video_duration(video_path: Path | str) -> float:
    """Probe actual video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def probe_video_resolution(video_path: Path | str) -> tuple[int, int]:
    """Probe video resolution (width, height) using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(video_path)
        ],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])
```

### Step 2: Add resolution unification function

File: `src/utils/ffmpeg_ops.py`

```python
def scale_video_to_target(
    input_path: Path | str,
    output_path: Path | str,
    target_width: int,
    target_height: int,
    fps: int = 30,
) -> None:
    """
    Scale video to target resolution with letterbox padding to preserve aspect ratio.
    
    LTX outputs are 768x512 (1.5:1). Source videos may be 640x320 (2:1), 1920x1080 (16:9), etc.
    Letterbox pad with black bars to preserve LTX content aspect ratio.
    Also normalize fps for clean concat.
    """
    # Build filter: scale to fit within target, pad to exact target size
    vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},"
        f"setsar=1"
    )
    
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",  # remove audio (segments are silent at this stage)
            str(output_path)
        ],
        check=True,
        capture_output=True,
    )
```

### Step 3: Add hybrid duration handling

File: `src/utils/ffmpeg_ops.py`

```python
# Constants for hybrid duration strategy
MIN_SLOWDOWN_RATIO = 0.6  # don't slow below 0.6x speed (i.e. clip plays at 60% speed = 1.67x duration)

def extend_clip_to_duration(
    input_path: Path | str,
    output_path: Path | str,
    target_duration_s: float,
    width: int,
    height: int,
    fps: int = 30,
) -> None:
    """
    Extend a short video clip to target duration using hybrid strategy:
    1. If clip duration >= target: trim to exact target.
    2. If MIN_SLOWDOWN_RATIO * target <= clip < target: pure slow playback to match.
    3. If clip < MIN_SLOWDOWN_RATIO * target: slow to MIN_SLOWDOWN_RATIO, then freeze last frame for remainder.
    """
    clip_dur = probe_video_duration(input_path)
    
    if clip_dur >= target_duration_s:
        # Case 1: trim
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(input_path),
                "-t", f"{target_duration_s}",
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-an",
                str(output_path)
            ],
            check=True, capture_output=True
        )
        return
    
    ratio = clip_dur / target_duration_s
    
    if ratio >= MIN_SLOWDOWN_RATIO:
        # Case 2: pure slow playback
        # setpts=PTS/speed where speed = clip_dur/target_duration (slow down)
        speed_factor = clip_dur / target_duration_s  # < 1.0 means slowdown
        vf = f"setpts={1.0/speed_factor:.6f}*PTS,fps={fps}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(input_path),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-an",
                str(output_path)
            ],
            check=True, capture_output=True
        )
        return
    
    # Case 3: hybrid (slow to MIN_SLOWDOWN_RATIO, then freeze remainder)
    # Slowed clip duration = clip_dur / MIN_SLOWDOWN_RATIO
    # Freeze duration = target - slowed_duration
    slowed_dur = clip_dur / MIN_SLOWDOWN_RATIO
    freeze_dur = target_duration_s - slowed_dur
    
    # Step 3a: produce slowed version
    slowed_path = Path(output_path).with_suffix(".slowed.mp4")
    speed_factor = MIN_SLOWDOWN_RATIO
    vf_slow = f"setpts={1.0/speed_factor:.6f}*PTS,fps={fps}"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_path),
            "-vf", vf_slow,
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-an",
            str(slowed_path)
        ],
        check=True, capture_output=True
    )
    
    # Step 3b: extract last frame as image
    last_frame_path = Path(output_path).with_suffix(".lastframe.png")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-sseof", "-0.1",  # seek to ~0.1s from end
            "-i", str(slowed_path),
            "-vframes", "1",
            str(last_frame_path)
        ],
        check=True, capture_output=True
    )
    
    # Step 3c: create freeze clip from last frame
    freeze_path = Path(output_path).with_suffix(".freeze.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-loop", "1",
            "-i", str(last_frame_path),
            "-t", f"{freeze_dur}",
            "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-an",
            str(freeze_path)
        ],
        check=True, capture_output=True
    )
    
    # Step 3d: concat slowed + freeze
    concat_list_path = Path(output_path).with_suffix(".concat.txt")
    concat_list_path.write_text(f"file '{slowed_path.resolve()}'\nfile '{freeze_path.resolve()}'\n")
    
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-an",
            str(output_path)
        ],
        check=True, capture_output=True
    )
    
    # Cleanup temp files
    for p in [slowed_path, last_frame_path, freeze_path, concat_list_path]:
        p.unlink(missing_ok=True)
```

### Step 4: Modify `cut_video_segment` to support target resolution

File: `src/utils/ffmpeg_ops.py`. Locate existing `cut_video_segment` function (lines 49-83 per audit report) and add optional resolution parameters:

```python
def cut_video_segment(
    source_path: Path | str,
    start: float,
    end: float,
    output_path: Path | str,
    target_width: int | None = None,
    target_height: int | None = None,
    fps: int = 30,
) -> None:
    """
    Cut a segment from source video. If target_width/height provided, scale + letterbox.
    Output is always re-encoded (no -c copy) to ensure uniform codec & timestamps.
    """
    duration = end - start
    
    if target_width and target_height:
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={fps},"
            f"setsar=1"
        )
    else:
        vf = f"fps={fps},setsar=1"
    
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start}",
            "-i", str(source_path),
            "-t", f"{duration}",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path)
        ],
        check=True, capture_output=True
    )
```

**Note:** This changes `cut_video_segment` signature. Update all callers di codebase. Use `target_width=None, target_height=None` defaults supaya existing callers tetap work, tapi assembler harus eksplisit pass target.

### Step 5: Modify assembler to use target resolution per video

File: `src/phase5_assemble.py`

Logic changes:

1. **At start of assembly**, determine target resolution dari source video:
   ```python
   source_video_path = ...  # path to original video for this video_id
   target_width, target_height = probe_video_resolution(source_video_path)
   target_fps = 30  # or probe from source
   ```

2. **For each segment processing:**
   
   - **Action=retrieve:** call `cut_video_segment(..., target_width=target_width, target_height=target_height, fps=target_fps)` — segment will be cut & normalized.
   
   - **Action=generate:**
     ```python
     # Step a: scale LTX clip to target resolution (LTX is 768x512, source may differ)
     scaled_clip_path = video_seg_path.with_suffix(".scaled.mp4")
     scale_video_to_target(generated_path, scaled_clip_path, target_width, target_height, fps=target_fps)
     
     # Step b: extend to audio duration with hybrid strategy
     extend_clip_to_duration(
         scaled_clip_path, 
         video_seg_path, 
         target_duration_s=group_audio_duration,
         width=target_width, 
         height=target_height,
         fps=target_fps,
     )
     
     # Cleanup
     scaled_clip_path.unlink(missing_ok=True)
     ```

3. **Spacer handling:** existing spacer logic harus pakai resolution & fps yang sama dengan segments. Kalau spacer ada, generate dengan `target_width × target_height` resolution dan `target_fps`.

4. **Concat method:** keep concat demuxer, BUT karena semua segments sekarang **uniform resolution + fps + codec**, `-c copy` should work cleanly. Kalau masih ada artifact, fallback ke filter_complex concat (re-encode all).

### Step 6: Verify

Run pipeline pada review_1:
```bash
python scripts/run_pipeline.py --video review_1 --rebuild-clips
```

Wait, but `--rebuild-clips` regenerates LTX. Don't do that (waste time). Use cached LTX output:
```bash
python scripts/run_pipeline.py --video review_1
```

(Assuming pipeline skips LTX if clips exist, per resume logic in original brief.)

**Verification checks** (run after pipeline completes):

1. **ffprobe final output:**
   ```bash
   ffprobe -v error -show_format -show_streams data/output/review_1/summary_grouping_gate.mp4
   ```
   - Video duration ≈ audio duration (within 0.5s tolerance)
   - Resolution: 640x320 (or whatever source review_1 was)
   - No stream errors

2. **Frame uniqueness check** (same as Investigation 4 in diagnostic):
   ```bash
   mkdir -p /tmp/review_1_post_fix
   ffmpeg -i data/output/review_1/summary_grouping_gate.mp4 -vf fps=1 /tmp/review_1_post_fix/frame_%03d.png
   ```
   Then count unique frames. Expected: substantial reduction in freeze regions vs pre-fix (88.7% frozen → should drop to <30% in worst case).

3. **Per-segment check:**
   ```bash
   ls data/intermediate/review_1/  # find segment intermediate files
   # for each segment, ffprobe to confirm uniform resolution + duration matching audio
   ```

4. **Visual sanity check (user-driven, not Gemini):**
   - Report at the end: "Pipeline complete. Please review `data/output/review_1/summary_grouping_gate.mp4` and confirm freeze regions are eliminated."

## What to report back

Markdown report:

```
## Implementation
- Files modified: <list>
- New functions added: <list>
- Caller updates: <list of files where cut_video_segment signature changed>

## Verification

### ffprobe final output
- Duration (video stream): X.XXs
- Duration (audio stream): X.XXs  
- Sync delta: X.XXs (should be < 0.5s)
- Resolution: WxH
- Codec, fps: <values>

### Frame uniqueness
- Total frames sampled: N
- Unique frames: N (X%)
- Frozen frame %: X% (target: <30%)
- Freeze regions ≥0.5s: <list>

### Per-segment audit (post-fix)
[Same table as Investigation 1, with new "post_fix_duration", "post_fix_resolution" columns]

### Diff vs diagnostic baseline
- Pre-fix freeze %: 88.7%
- Post-fix freeze %: X%
- Improvement: yes/no

## Anomalies / issues
<list any errors, warnings, or unexpected behavior>

## Ready for user visual review
Output file: data/output/review_1/summary_grouping_gate.mp4
```

## Hard rules

- **Implement code as provided** with minor adjustments for codebase conventions (imports, logging, etc).
- **Don't redesign architecture.** Stick to provided functions.
- **Test only on review_1.** Don't scale.
- **Don't claim quality** ("artifact-free", "high-fidelity"). Report numbers only.
- **If any function fails** (e.g. ffmpeg command errors), report full stderr verbatim and STOP.
- **Don't regenerate LTX clips** unless explicitly necessary (use cached).

## Anti-hallucination

- All ffprobe values from actual command output
- Frame counts from actual file enumeration (`ls | wc -l`)
- If post-fix frozen % > 30%, report as failure — don't spin as success
- If sync delta > 0.5s, report and investigate further

## Out of scope

- Regenerate LTX clips
- Tune slowdown ratio constant (locked at 0.6)
- Run on videos other than review_1
- Visual quality judgment of generated clips themselves

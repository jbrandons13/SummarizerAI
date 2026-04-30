# TASK: Cap clip duration in Phase 5 Assembly

## Problem
Some matched scenes are too long (10-30 seconds) while the narration sentence is only 3-5 seconds. The output video has awkward long silent clips.

## Fix
In Phase 5 (src/phase5_assemble.py), when cutting clips from the source video, cap each clip's duration to match its corresponding audio sentence duration + a small buffer.

## Logic
```
For each sentence:
    audio_duration = sentence audio clip duration (from AudioManifest)
    max_clip_duration = audio_duration + 0.5  # 0.5s buffer
    
    scene_start = matched scene start_seconds
    scene_end = matched scene end_seconds
    scene_mid = (scene_start + scene_end) / 2
    
    # Crop centered around scene midpoint, capped to max_clip_duration
    clip_start = max(scene_start, scene_mid - max_clip_duration / 2)
    clip_end = min(scene_end, clip_start + max_clip_duration)
    
    # Use clip_start and clip_end for FFmpeg -ss and -t
```

## FFmpeg command change
```
# BEFORE:
ffmpeg -ss {scene_start} -to {scene_end} -i video.mp4 ...

# AFTER:
ffmpeg -ss {clip_start} -t {max_clip_duration} -i video.mp4 ...
```

## Config
Add to configs/default.yaml:
```yaml
assembly:
  clip_duration_mode: "match_audio"  # options: "full_scene", "match_audio"
  clip_buffer_seconds: 0.5
```

## Do NOT change
- Phase 1, 2, 3, 4 — unaffected
- Audio generation — stays the same
- Subtitle timing — should still sync correctly since clip = audio duration

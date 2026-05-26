# Brief: Phase 5 LTX Generation Module — 2-Stage Offline Implementation

**Task type:** Execution (implementation), bukan research.
**Goal:** Build Phase 5 LTX generation module yang baca `p4_assignments.json`, generate video clips untuk groups dengan `action="generate"`, dan save ke path yang assembler udah expect (`data/intermediate/{video_id}/generated/group_{group_id}.mp4`).

Architecture: **2-stage offline** untuk separation of concerns + inspectability + resume-friendly.

## Context (verified state)

- Phase 4 outputs `data/intermediate/{video_id}/p4_assignments.json` dengan list of Assignment per group. Each Assignment punya: `sentence_ids`, `scene_id`, `best_similarity`, `action` (retrieve/generate), `timestamp_hint_merged`.
- Phase 2 outputs `data/intermediate/{video_id}/summary_script.json` dengan `keywords: List[str]` per sentence.
- Phase 3 outputs `data/intermediate/{video_id}/audio/sentence_NNN.wav` per sentence.
- Phase 4 keyframes manifest di `data/intermediate/{video_id}/keyframes_manifest.json` dengan `multi_frame_paths` per scene.
- Assembler udah expect generate output di `data/intermediate/{video_id}/generated/group_{group_id}.mp4`, dengan fallback ke retrieval kalau file ga ada.
- LTX I2V locked: `Lightricks/LTX-Video-0.9.7-distilled`, 768x512, 8 steps custom timesteps, `enable_model_cpu_offload()` (atau sequential fallback kalau OOM).
- Qwen2.5-VL-3B available untuk prompt rewriter. Field `keywords` udah verified quality.
- VRAMManager di `src/utils/vram.py` bisa load/unload model (verified).

## Architecture: 2-stage offline

```
Stage A: Prompt Construction (per video)
  Input: p4_assignments.json + summary_script.json + keyframes_manifest.json
  Process: Qwen2.5-VL liat keyframe + narrasi + keywords → generate LTX prompt
  Output: data/intermediate/{video_id}/ltx_prompts.json
         [{group_id, action, num_frames, keyframe_path, prompt, audio_duration}]

Stage B: LTX Generation (per video, after Stage A)
  Input: ltx_prompts.json
  Process: Load LTX → for each generate group: generate clip, trim, save
  Output: data/intermediate/{video_id}/generated/group_{group_id}.mp4
```

Why 2-stage:
- Inspectable: review `ltx_prompts.json` before commit ~70 min ke LTX run
- Resume-friendly: Stage B crash, restart from clip yang belum jadi
- VRAM clean: 1 model di GPU at a time (Qwen-VL ~5GB, LTX ~17GB — ga akan fit barengan)
- Workflow phase-able di thesis writeup

## File structure to create

```
src/phase5_generate.py       # Main orchestrator: read p4_assignments, call Stage A & B
src/phase5_prompt_builder.py # Stage A: Qwen-VL prompt rewriter
src/phase5_ltx_runner.py     # Stage B: LTX I2V wrapper
```

Plus update `src/pipeline.py` untuk call `phase5_generate` antara Phase 4 dan Phase 5 assembler (existing `phase5_assemble.py`).

## Stage A: Prompt Builder

### Module: `src/phase5_prompt_builder.py`

```python
class PromptBuilder:
    """Stage A: construct LTX prompts via Qwen2.5-VL multimodal LLM."""
    
    def __init__(self, vram_manager, model_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"):
        ...
    
    def build_prompts(self, video_id: str) -> Path:
        """
        Read Phase 2/3/4 outputs, generate LTX prompts for all generate groups.
        Save to data/intermediate/{video_id}/ltx_prompts.json
        Return path to generated json.
        """
```

### Per-group prompt construction logic

For each group in `p4_assignments.json` dengan `action="generate"`:

1. **Aggregate narration**: concat text dari semua sentence di group (lookup di summary_script.json by sentence_ids).
2. **Aggregate keywords**: union dari keywords semua sentence di group (dedupe, preserve order).
3. **Aggregate audio duration**: sum durations dari `sentence_NNN.wav` files (use `librosa.get_duration` atau `soundfile.info`).
4. **Pick keyframe**: 
   - Group's `scene_id` → lookup di keyframes_manifest → pick `multi_frame_paths[0]` (first frame) OR frame closest to `best_frame_timestamp`
   - Resolve to absolute path
   - Preprocess: resize ke 768x512 (LTX requirement, divisible by 32), center crop kalau aspect ratio beda
5. **Decide num_frames** adaptive:
   - audio_duration ≤ 4.0 → num_frames = 121 (4.03s @ 30fps)
   - audio_duration > 4.0 → num_frames = 241 (8.03s @ 30fps)
6. **Construct prompt via Qwen2.5-VL**:
   - Input: keyframe image + narration + keywords
   - System prompt instructs Qwen to produce LTX-friendly description (lihat template below)
   - Output: text prompt string

### Qwen2.5-VL system prompt template

```
You are a visual prompt engineer for an image-to-video diffusion model (LTX-Video).
Your task: given a reference image (a frame from a tech review video), a narration script, 
and a list of visual keywords, write a single English prompt (max 80 words) that describes 
what the generated video clip should show.

Rules for the prompt:
1. Start with the main subject visible in the image (be specific: brand, color, shape, key features).
2. Describe natural camera motion (e.g. "slow camera pan", "smooth dolly forward", "static camera with subtle handheld feel"). For complex scenes with multiple objects, prefer "static camera, no zoom".
3. Include details from the narration that are visually depictable.
4. Use the keywords as anchors for specific visual elements.
5. End with style cues: "tech product review style, soft studio lighting, dark background, shallow depth of field".
6. Do NOT use negative phrasing ("no buttons", "without X"). Use positive description only.
7. Do NOT mention sound, music, or text overlays.
8. Single paragraph, no bullet points.

Output the prompt text directly, no preamble.
```

### User message format

```
Reference image: <image attachment>
Narration: <concatenated text from all sentences in group>
Keywords: <comma-separated keyword list>
```

### Output format: `ltx_prompts.json`

```json
{
  "video_id": "review_1",
  "groups": [
    {
      "group_id": 0,
      "sentence_ids": [0, 1],
      "action": "generate",
      "audio_duration_seconds": 10.704,
      "num_frames": 241,
      "keyframe_path": "data/intermediate/review_1/keyframes/scene_004_f00.jpg",
      "keyframe_preprocessed_path": "data/intermediate/review_1/keyframes_ltx/group_000_keyframe_768x512.jpg",
      "narration": "Samsung released the new Buds 4 and Buds 4 Pro earbuds...",
      "keywords": ["Samsung", "Buds 4", "earbuds", "cube-shaped case", "tinted clear lid"],
      "prompt": "<Qwen-VL generated prompt>"
    },
    ...
  ]
}
```

Include `action="retrieve"` groups in JSON with placeholder fields (null prompt) for completeness, but don't process them via Qwen-VL (skip to save time).

### Stage A behavior

- Skip groups with `action="retrieve"` (no prompt needed).
- Cache: if `ltx_prompts.json` already exists, ask user (CLI flag `--rebuild-prompts`) before overwrite.
- VRAM management:
  - Pause Ollama/orchestrator before loading Qwen-VL (use subprocess to send SIGSTOP to known PID; PID discovery via `pgrep` for "ollama" or known process name)
  - Load Qwen-VL via VRAMManager
  - Batch-process all generate groups
  - Unload Qwen-VL after all groups done
  - Resume Ollama (SIGCONT)
- Error handling: if Qwen-VL fails on one group (e.g. CUDA error), log group_id and continue with others. Mark prompt as null in JSON for that group.

## Stage B: LTX Runner

### Module: `src/phase5_ltx_runner.py`

```python
class LTXRunner:
    """Stage B: generate I2V clips using LTX-Video 0.9.7 distilled."""
    
    def __init__(self, vram_manager, model_path="~/models/ltx_video_distilled"):
        ...
    
    def generate_clips(self, video_id: str) -> List[Path]:
        """
        Read ltx_prompts.json, generate clips for all generate groups,
        save to data/intermediate/{video_id}/generated/group_{group_id}.mp4.
        Return list of generated clip paths.
        Skip groups that already have output file (resume support).
        """
```

### Per-clip generation logic

For each generate group in `ltx_prompts.json`:

1. **Skip if output already exists** (resume support): `data/intermediate/{video_id}/generated/group_{group_id}.mp4`.
2. **Load preprocessed keyframe** from `keyframe_preprocessed_path` (already 768x512).
3. **Generate clip via LTX I2V**:
   - `LTXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=bfloat16)`
   - `pipeline.enable_model_cpu_offload()`
   - Apply `retrieve_timesteps` monkeypatch (from previous smoke test session)
   - Custom timesteps: `[1000, 993, 987, 981, 975, 909, 725, 0.03]`
   - `guidance_scale=1.0`
   - `seed=42` (deterministic for reproducibility)
   - Call: `pipeline(prompt=..., image=..., num_frames=..., height=512, width=768, generator=...)`
4. **Trim output to audio_duration**:
   - LTX outputs 121f (~4.03s) or 241f (~8.03s) @ 30fps
   - Trim via ffmpeg to exact audio_duration (use `-t` flag, no re-encode if possible: `-c copy`)
   - Output: `data/intermediate/{video_id}/generated/group_{group_id}.mp4`
5. **Cleanup**: clear temp files, save inference time + peak VRAM in `data/intermediate/{video_id}/generation_metrics.json`.

### Stage B behavior

- VRAM management:
  - Pause Ollama (SIGSTOP) before loading LTX
  - Load LTX once at start, keep loaded throughout video
  - Process all generate groups in this video
  - Unload LTX after video done
  - Resume Ollama (SIGCONT)
- Error handling per-clip:
  - Try-except around generation call
  - On OOM: try `enable_sequential_cpu_offload()` fallback, retry once
  - On other error: log group_id + traceback, skip group (assembler will fallback to retrieval)
- Crash recovery hook:
  - Signal handler (SIGINT) yang ensure Ollama resumed before exit
  - Use try-finally pattern around Ollama pause/resume

## Pipeline integration

Update `src/pipeline.py`:

```python
# Existing flow:
phase1_transcript = run_phase1(...)
phase2_summary = run_phase2(...)
phase3_audio = run_phase3(...)
phase4_assignments = run_phase4_grouping_gate(...)  # writes p4_assignments.json

# NEW: Phase 5 generation
prompt_builder = PromptBuilder(vram_manager)
prompts_path = prompt_builder.build_prompts(video_id)  # writes ltx_prompts.json

ltx_runner = LTXRunner(vram_manager)
generated_clips = ltx_runner.generate_clips(video_id)  # writes generated/*.mp4

# Existing: Phase 5 assembler (consumes generated/ + falls back to retrieval if missing)
phase5_output = run_phase5_assemble(...)
```

CLI flags untuk skip Phase 5 generation kalau perlu (debug):
- `--skip-generation`: skip both stages, assembler will fallback semuanya
- `--rebuild-prompts`: regenerate ltx_prompts.json bahkan kalau exist
- `--rebuild-clips`: regenerate semua clips (overwrite existing)

## Testing on review_1

After implementation:

### Test 1: Stage A only

Run prompt builder on review_1. Inspect output `ltx_prompts.json`:
- All generate groups have non-null prompt?
- Prompts make sense (read 2-3 manually)?
- Aspect ratio preprocessing creates valid 768x512 keyframes?

### Test 2: Stage B on subset

Run LTX runner on review_1 BUT modify temporarily to only process **first 2 generate groups** (debug mode). Verify:
- Clips generated successfully?
- Output mp4 valid (use ffprobe to verify duration matches audio_duration)?
- VRAM peak reasonable (<20 GB)?
- Latency per clip reasonable (~40-70s)?

### Test 3: Full pipeline review_1

Run full pipeline (Phase 1-4 + Phase 5 generation + Phase 5 assembler). Verify:
- All generate groups in review_1 produced clips
- Final mp4 (`data/output/review_1/summary_grouping_gate.mp4`) uses generated clips where action=generate, retrieval clips where action=retrieve
- Total duration reasonable
- Visual sanity check: tonton output, apakah generated segments terlihat reasonable?

## What to report back

Markdown report dengan struktur:

```
## Implementation
- Files created: <list>
- Files modified: <list>
- CLI flags added: <list>

## Test 1: Stage A (prompt builder)
- Generate groups processed: <N>
- Sample 3 prompts (with group_id + narration + final prompt)
- Time: <total seconds>
- Issues: <list>

## Test 2: Stage B subset (LTX runner debug mode)
- Clips generated: <N>
- Per-clip: latency, peak VRAM, output duration vs audio_duration
- Visual sanity (subjective): "looks reasonable" / specific issues

## Test 3: Full pipeline review_1
- Total generate groups: <N>
- Successfully generated: <N>
- Fallback to retrieval (failures): <N>
- Final mp4 duration: <value>
- Visual notes per generated clip (1 line each)

## VRAM orchestration
- Ollama pause/resume working: yes/no
- Cleanup hook tested (Ctrl+C during generation): yes/no
- Peak VRAM across runs: <value>

## Summary
- Phase 5 LTX integration: success / partial / failed
- Remaining issues: <list>
- Ready to scale to all 10 videos: yes/no
```

## Hard rules

- **STAGE A AND B SEPARATE.** Don't combine into single script with both models loaded.
- **VRAM clean between stages.** Unload Qwen-VL before loading LTX.
- **Resume support mandatory.** Both stages skip already-completed work.
- **Ollama orchestration in code.** Use subprocess to SIGSTOP/SIGCONT. Identify PID via `pgrep -f ollama` or known PID from orchestrator.
- **Cleanup hook mandatory.** SIGCONT Ollama even if pipeline crashes (try-finally).
- **Don't modify Phase 4 code.** Just consume p4_assignments.json.
- **Don't modify Phase 5 assembler logic.** Just ensure clips land in expected path.
- **Test on review_1 only first.** Don't auto-run on all 10 videos. User decide after seeing review_1 result.

## Anti-hallucination

- Verify file paths exist before claim
- Quote actual values from output files (durations, scores)
- If Qwen-VL or LTX fail in unexpected way, log full traceback verbatim
- If output mp4 duration doesn't match audio_duration, report exact deltas
- Visual notes from actual mp4 inspection, jangan generalisasi

## Out of scope

- Auto-rerun on quality score threshold (no metric-based retry)
- Running on all 10 videos (do review_1 first, user decides scaling)
- Renaming phase5_assemble.py → phase6_assemble.py (separate cleanup task)
- Implementing SigLIP-based post-generation quality scoring (defer to evaluation phase)
- Tuning num_frames thresholds (4s cutoff locked: ≤4s → 121f, >4s → 241f)

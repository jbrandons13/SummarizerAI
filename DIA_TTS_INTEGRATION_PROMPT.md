# TASK: Integrate Dia 1.6B TTS as Alternative to Kokoro in Phase 3

## Context
We have a video summarization pipeline with 5 phases. Phase 3 currently uses Kokoro v1.0 ONNX for TTS. We want to add Dia 1.6B (by Nari Labs) as a second TTS option that can be selected via config, while keeping Kokoro as fallback.

The pipeline runs on a single RTX 3090 (24GB VRAM) with sequential model loading — only one large model in memory at a time.

## About Dia 1.6B
- 1.6B parameter TTS model by Nari Labs
- ~10GB VRAM
- English only (that's fine, our pipeline outputs English narration)
- Generates highly realistic speech from text
- Supports non-verbal cues like (laughs), (coughs) via tags
- Single speaker format: just plain text (no [S1] tags needed for our use case)
- Apache-like license
- GitHub: https://github.com/nari-labs/dia
- HuggingFace: nari-labs/Dia-1.6B
- Output: 44100 Hz WAV

## What to Do

### Step 1: Install Dia
```bash
pip install dia-tts
# or clone the repo if pip package doesn't work:
# git clone https://github.com/nari-labs/dia.git
# cd dia && pip install -e .
```

### Step 2: Create Dia TTS backend in the pipeline

In the existing TTS module (likely `src/phase3_voiceover.py`), add a Dia backend alongside Kokoro. The interface should be the same — takes text, returns audio file path.

```python
# Basic Dia usage:
import soundfile as sf
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B")
text = "Your narration sentence here."
output = model.generate(text)
sf.write("output.wav", output, 44100)
```

### Step 3: Integration requirements

1. **Same interface as Kokoro.** The rest of the pipeline should not need to change. Phase 3 takes a list of sentences, outputs audio files + AudioManifest.

2. **Config-driven selection.** In `configs/default.yaml`, add:
```yaml
tts:
  backend: "dia"  # options: "kokoro", "dia"
  # Kokoro settings (existing)
  kokoro_voice: "af_heart"
  kokoro_speed: 1.05
  # Dia settings (new)
  dia_model: "nari-labs/Dia-1.6B"
```

3. **Per-sentence generation.** Loop through each sentence in the SummaryScript, generate audio individually (same as Kokoro does now). This is important because each sentence maps to one visual segment in Phase 4.

4. **Maintain existing post-processing:**
   - Loudness normalization to -18 LUFS (use pyloudnorm or ffmpeg)
   - 150ms silence padding between sentences
   - clean_for_tts() preprocessing should still be applied before passing text to Dia

5. **Sample rate handling.** Dia outputs 44100 Hz. Check if Phase 5 (FFmpeg assembly) expects a specific sample rate. If the rest of the pipeline expects 24000 Hz (Kokoro's output rate), resample Dia's output to match:
```python
import librosa
audio_resampled = librosa.resample(output, orig_sr=44100, target_sr=24000)
```
Or adjust FFmpeg commands in Phase 5 to handle 44100 Hz.

6. **VRAM management.** Dia uses ~10GB. Make sure VRAMManager:
   - Unloads Phase 2 LLM before loading Dia
   - Unloads Dia after Phase 3 is done, before Phase 4 loads VLM/SigLIP
   - Clears CUDA cache between phases

### Step 4: Test

1. Run Phase 3 with `backend: "dia"` on a single sentence first
2. Check audio quality — does it sound natural?
3. Check audio file format — correct sample rate, mono channel
4. Run full pipeline end-to-end with Dia
5. Compare output video quality vs Kokoro version

### Step 5: Handle potential issues

- **Dia is English-only.** If input text contains non-English characters, clean_for_tts() should strip them.
- **Dia may be slower than Kokoro.** This is expected (~10x slower). For ablation with 8-10 videos, total added time might be 10-20 minutes. Acceptable.
- **If Dia fails to install or has dependency conflicts**, fall back to Kokoro and document the attempt. We already had this problem with Chatterbox and Orpheus before — don't spend more than 1-2 hours debugging install issues.
- **Long sentences.** If Dia struggles with very long sentences (>50 words), split them at natural pause points (commas, semicolons) and concatenate the audio clips.

## File Changes Expected

| File | Change |
|------|--------|
| src/phase3_voiceover.py | Add DiaTTS class alongside existing KokoroTTS |
| configs/default.yaml | Add tts.backend and dia settings |
| requirements.txt | Add dia-tts (or dia dependency) |
| src/pipeline.py | No change needed if Phase 3 interface stays the same |

## Do NOT Change
- Phase 1, 2, 4, 5 — they should not be affected
- The AudioManifest output schema — Dia should produce the same output format
- clean_for_tts() — keep using it, just apply before passing to Dia
- Loudness normalization and silence padding — keep these as post-processing

## Success Criteria
- `python -c "from dia.model import Dia; print('OK')"` works
- Phase 3 runs with backend="dia" and produces audio files
- Audio sounds natural and clear
- Full pipeline produces a valid output video with Dia narration
- Can switch between Kokoro and Dia via config change only

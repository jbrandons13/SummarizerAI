# Project Summary: Video Summarizer AI

### Files Created
- `video-summarizer/src/phase5_assemble.py`: Phase 5 implementation (Precise Assembly)
- `video-summarizer/src/pipeline.py`: Full pipeline orchestrator
- `video-summarizer/tests/test_phase5.py`: Unit tests for assembly logic
- `video-summarizer/tests/test_pipeline.py`: Integration test suite
- `video-summarizer/data/intermediate/{video_id}/`:
    - `keyframes/`: Extracted scene midpoint frames (JPG quality 90)
    - `keyframes_manifest.json`: List of scenes with start/end/midpoint
    - `keyframes_captions.json`: Cached Qwen-VL captions for Arm B
    - `scene_matches_{method}.json`: Match results for Arm A, B, or C
- `video-summarizer/data/output/`: Final summary videos and provenance metadata

### Key Component Signatures
- **KeyframeExtractor** (`phase4_retrieve.py`):
  - `extract(video_path: Path) -> Path`: PySceneDetect (threshold=27.0) + Uniform fallback. 
- **RetrievalArms** (`phase4_retrieve.py`):
  - `RandomRetrieval`: Baseline arm.
  - `CaptionCosineRetrieval`: `Qwen2.5-VL-3B-Instruct` + `SentenceTransformer`. VRAM Peak: ~8GB.
  - `SigLIP2DirectRetrieval`: `google/siglip2-so400m-patch16-naflex`. VRAM Peak: ~2GB.
- **Phase5Assembler** (`phase5_assemble.py`):
  - `run(...)`: Re-encodes video segments (H.264 CRF 20), interleaves 200ms silence, and burns in subtitles.
- **VideoSummarizerPipeline** (`pipeline.py`):
  - `run(video_path, method)`: Orchestrates Phases 1-5 with VRAM-safe transitions.

### Environment & Dependencies (Phase 4/5)
- **WAJIB**: `transformers` must be installed from source for SigLIP 2 support.
- **WAJIB**: `FFmpeg` must support `libx264` and `libass` for re-encoding and subtitle burn-in.
- **VRAM Manager**: Handles sequential loading/unloading to keep peak usage under 10GB for most steps.

### Assembly Logic
- **Re-encoding**: Mandatory re-encoding for all segments to ensure timestamp alignment.
- **Padding**: 200ms silence interleaving between segments for better audio-visual flow.
- **Subtitles**: Automatic `.srt` generation and hard-burning into the final output.
- **Provenance**: Generates `{video_id}_summary_{method}_metadata.json` with full segment lineage.

### Progress Status
- [x] Phase 1: Ingestion & Precise Transcription (WhisperX).
- [x] Phase 2: Semantic Highlight Detection & Scripting.
- [x] Phase 3: Neural Voiceover (Kokoro/F5-TTS).
- [x] Phase 4: Semantic Visual Retrieval (3 Retrieval Arms + VRAM Safe).
- [x] Phase 5: Video Assembly & Final Render.
- [x] End-to-End Verification on `tiny_video.mp4` (Success).
- [ ] Phase 6: Evaluation & Comparison.

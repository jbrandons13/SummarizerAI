# Project Summary: Video Summarizer AI

### Files Created
- `video-summarizer/pyproject.toml`: Pinned dependencies (Torch 2.4, WhisperX, etc.)
- `video-summarizer/configs/default.yaml`: Hyperparams and model names
- `video-summarizer/src/schemas.py`: Pydantic models for cross-phase validation
- `video-summarizer/src/phase1_transcribe.py`: Phase 1 implementation (Transcription)
- `video-summarizer/src/phase2_summarize.py`: Phase 2 implementation (LLM Scripting)
- `video-summarizer/src/phase3_voiceover.py`: Phase 3 implementation (Neural Voiceover)
- `video-summarizer/src/models/llm_wrapper.py`: LLM wrappers (Groq/Local)
- `video-summarizer/src/models/tts_wrapper.py`: TTS wrappers (Kokoro/F5-TTS)
- `video-summarizer/src/utils/{vram.py, ffmpeg_ops.py, io.py}`: Core utilities
- `video-summarizer/{.env, .gitignore, README.md}`: Project meta

### Key Component Signatures
- **TranscriptionPhase** (`phase1_transcribe.py`):
  - `run(video_path: Path) -> Path`: WhisperX large-v3 + alignment. Saves `transcript.json`.
- **Phase2Summarizer** (`phase2_summarize.py`):
  - `run(...) -> Path`: Orchestrates chunking + LLM calls. Saves `summary_script.json`.
- **Phase3Voiceover** (`phase3_voiceover.py`):
  - `run(...) -> Path`: Sentence TTS generation + Audio Manifest. Saves `audio_manifest.json`.
- **LLMBackend** (`llm_wrapper.py`):
  - `GroqBackend` / `LocalBackend` wrappers.
- **TTSBackend** (`tts_wrapper.py`):
  - `KokoroBackend`: Ultra-fast English TTS (ONNX).
  - `F5TTSBackend`: High-quality voice cloning (F5-TTS).
- **VRAMManager** (`vram.py`):
  - `load_model(name, loader_fn)`: Unload prev, clear cache, load new.
  - `log_peak_usage(phase_name: str)`: Track VRAM peak (Phase 1 Peak: 12.4 GB).
- **FFmpeg Ops** (`ffmpeg_ops.py`):
  - `extract_audio(v_path, a_path, sr=16k)`
  - `extract_frame_at(v_path, ts, img_path)`
  - `cut_video_segment(v_path, start, end, out_path)` (Re-encode)
  - `concat_videos(v_paths, out_path)`
- **IO Helpers** (`io.py`):
  - `load_json_as_model(path, pydantic_model)` / `save_model_as_json(instance, path)`

### Environment Setup (CUDA 12.1 Fix)
- `LD_LIBRARY_PATH` workaround is required for `ctranslate2` compatibility.
- Fixed via conda activation script: `$CONDA_PREFIX/etc/conda/activate.d/cuda_fix.sh`.
- Points to `nvidia/cudnn/lib` and `nvidia/cublas/lib` in site-packages.

### Progress Status
- [x] Phase 1: Ingestion & Precise Transcription (WhisperX + Alignment).
- [x] Phase 2: Semantic Highlight Detection & Scripting (Groq/Local Dual Backend).
- [x] Phase 3: Neural Voiceover (Kokoro/F5-TTS + Normalization).
- [ ] Phase 4: Multimodal Data Ingestion (B-Roll matching).
- [ ] Phase 5: Video Assembly & Final Render.
- [ ] `shot_detect.py` utility is currently a placeholder.

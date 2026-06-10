# Handoff Summary: Full Overnight Pipeline Run (v2)

**Timestamp:** 2026-06-09 12:15 WIB (Approx)

## 1. What Has Been Achieved
The Phase A processing for the newly introduced showcase videos (**Sun**, **Heart**, **iPhone Review**) is fully complete. Specifically:
- **Phase 1-3 (Transcribe, Summarize, TTS)**: Successfully executed using WhisperX (patched PyTorch 2.6 `weights_only` error), Qwen 14B AWQ, and Kokoro TTS.
- **Phase 4 (Segmentation & Storyboard)**: Successfully extracted scene boundaries (`shots.json`) via PySceneDetect and generated `storyboard.json`.
- **Dynamic Anchor Generation (Bug Fixed)**: Overcame a PIL error caused by empty `touch` dummy files. `weight_sweep.py` was monkey-patched to dynamically invoke SDXL and generate a valid `shot_001` concept anchor image with $w=0.0$ if the reference was a 1x1 dummy.
- **Sun (V3) Fine-Grid Sweep**: Successfully generated 144 images (16 shots x 9 weights). DINOv2 metrics and the adaptive anchor thresholds (`w^*`) have been accurately extracted and saved.

## 2. Current Execution State
The `run_overnight_v2.sh` script is currently running securely in the background via `nohup` (`run_overnight_v2.log`).
- **Heart & iPhone Sweeps**: The script is progressing through the T2I Fine-Grid Sweeps for the Heart and iPhone videos.
- **Caching Advantage**: Due to aggressive checkpoint caching, restarting the script bypassed all previously completed tasks instantly, picking up right at the metric evaluation stage.

## 3. Next Steps (Automated)
Once the Heart and iPhone sweeps conclude in the next ~30-40 minutes, the pipeline will automatically trigger **Phase B (Wan I2V Renders)**.
- **DACA First**: It will render the Adaptive Concept Anchoring (DACA) version for all 4 target videos (Sun, Heart, Geology, Ecology).
- **FIXED Second**: It will follow up with the `w=0.2` fixed anchor renders.
*(Note: Ecology and Geology bypass the sweeps completely, leveraging their already locked image sets and quantitative data to maintain thesis consistency).*

## 4. Known Bugs & Caveats Resolved
- **PyTorch 2.6 Unpickler Error**: Patched `src/phase1_transcribe.py` temporarily overriding `torch.load` to accept `weights_only=False`.
- **Adaptive Anchor Pathing**: Corrected a bug in the shell script where `adaptive_anchor.py` was being called from the global scope rather than `src/phase4/adaptive_anchor.py`, and fixed the `--out` argument for `collapse_metrics.py` which generates a directory.
- **Pre-flight yt-dlp 403 Forbidden**: YouTube blocked automated downloads, so we relied on the 3 manually downloaded raw `.mp4` files placed in `data/raw_videos`.

**To the Next Agent:**
Do NOT interrupt `run_overnight_v2.sh` unless an Out-Of-Memory (OOM) error surfaces in the logs. Monitor `run_overnight_v2.log` via `tail` to observe Phase B I2V generation rates. Phase B takes ~16-20 hours of total GPU time and is not expected to finish before tomorrow.

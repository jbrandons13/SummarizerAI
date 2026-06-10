#!/bin/bash
set -e

echo "=== PRE-FLIGHT SMOKE ==="
python -m py_compile *.py src/*.py src/**/*.py
echo "[OK] PyCompile passed."

mkdir -p runs/sun/audio runs/heart/audio runs/iphone/audio
mkdir -p runs/sun/sweep runs/heart/sweep runs/iphone/sweep
mkdir -p runs/sun/daca runs/heart/daca runs/iphone/daca
mkdir -p runs/sun/images_fixed_w02 runs/sun/images_daca runs/sun/clips_fixed runs/sun/clips_daca
mkdir -p runs/heart/images_fixed_w02 runs/heart/images_daca runs/heart/clips_fixed runs/heart/clips_daca
mkdir -p runs/geology/images_fixed_w02 runs/geology/images_daca runs/geology/clips_fixed runs/geology/clips_daca
mkdir -p runs/ecology/images_fixed_w02 runs/ecology/images_daca runs/ecology/clips_fixed runs/ecology/clips_daca

SUN_VIDEO="data/raw_videos/b22HKFMIfWo.mp4"
HEART_VIDEO="data/raw_videos/X9ZZ6tcxArI.mp4"
IPHONE_VIDEO="data/raw_videos/MRtg6A1f2Ko.mp4"

if [ ! -f "$SUN_VIDEO" ] || [ ! -f "$HEART_VIDEO" ] || [ ! -f "$IPHONE_VIDEO" ]; then
    echo "ERROR: Missing raw videos."
    exit 1
fi

echo "=== PHASE A: SUN ==="
if [ ! -f "runs/sun/storyboard.json" ]; then
    echo "Running Pipeline Phase 1-3 for Sun"
    python scripts/run_pipeline.py --phases 1,2,3 "$SUN_VIDEO"
    
    SUN_ID="b22HKFMIfWo"
    echo "Running Phase 4 for Sun"
    PYTHONPATH=. python src/phase4/segmenter.py --video-id ${SUN_ID}
    PYTHONPATH=. python src/phase4/storyboard.py --video-id ${SUN_ID}
    PYTHONPATH=. python src/phase4/anchor_policy.py --video-id ${SUN_ID}
    
    cp data/intermediate/${SUN_ID}/phase4/storyboard.json runs/sun/ 2>/dev/null || cp data/intermediate/${SUN_ID}/storyboard.json runs/sun/ 2>/dev/null || cp data/intermediate/${SUN_ID}/phase2/storyboard.json runs/sun/ 2>/dev/null
    cp data/intermediate/${SUN_ID}/summary_script.json runs/sun/ 2>/dev/null || cp data/intermediate/${SUN_ID}/phase2/summary_script.json runs/sun/ 2>/dev/null
    cp data/intermediate/${SUN_ID}/phase4/audio/*.wav runs/sun/audio/ 2>/dev/null || cp data/intermediate/${SUN_ID}/audio/*.wav runs/sun/audio/ 2>/dev/null
    cp data/intermediate/${SUN_ID}/phase4/concept_anchor_canonical_w02/images/*.png runs/sun/reference.png 2>/dev/null || true
fi

echo "=== PHASE A: HEART ==="
if [ ! -f "runs/heart/storyboard.json" ]; then
    echo "Running Pipeline Phase 1-3 for Heart"
    python scripts/run_pipeline.py --phases 1,2,3 "$HEART_VIDEO"
    
    HEART_ID="X9ZZ6tcxArI"
    echo "Running Phase 4 for Heart"
    PYTHONPATH=. python src/phase4/segmenter.py --video-id ${HEART_ID}
    PYTHONPATH=. python src/phase4/storyboard.py --video-id ${HEART_ID}
    PYTHONPATH=. python src/phase4/anchor_policy.py --video-id ${HEART_ID}
    
    cp data/intermediate/${HEART_ID}/phase4/storyboard.json runs/heart/ 2>/dev/null || cp data/intermediate/${HEART_ID}/storyboard.json runs/heart/ 2>/dev/null || cp data/intermediate/${HEART_ID}/phase2/storyboard.json runs/heart/ 2>/dev/null
    cp data/intermediate/${HEART_ID}/summary_script.json runs/heart/ 2>/dev/null || cp data/intermediate/${HEART_ID}/phase2/summary_script.json runs/heart/ 2>/dev/null
    cp data/intermediate/${HEART_ID}/phase4/audio/*.wav runs/heart/audio/ 2>/dev/null || cp data/intermediate/${HEART_ID}/audio/*.wav runs/heart/audio/ 2>/dev/null
    cp data/intermediate/${HEART_ID}/phase4/concept_anchor_canonical_w02/images/*.png runs/heart/reference.png 2>/dev/null || true
fi

echo "=== PHASE A: IPHONE REVIEW ==="
if [ ! -f "runs/iphone/storyboard.json" ]; then
    echo "Running Pipeline Phase 1-3 for iPhone"
    python scripts/run_pipeline.py --phases 1,2,3 "$IPHONE_VIDEO"
    
    IPHONE_ID="MRtg6A1f2Ko"
    echo "Running Phase 4 for iPhone"
    PYTHONPATH=. python src/phase4/segmenter.py --video-id ${IPHONE_ID}
    PYTHONPATH=. python src/phase4/storyboard.py --video-id ${IPHONE_ID}
    PYTHONPATH=. python src/phase4/anchor_policy.py --video-id ${IPHONE_ID}
    
    cp data/intermediate/${IPHONE_ID}/phase4/storyboard.json runs/iphone/ 2>/dev/null || cp data/intermediate/${IPHONE_ID}/storyboard.json runs/iphone/ 2>/dev/null || cp data/intermediate/${IPHONE_ID}/phase2/storyboard.json runs/iphone/ 2>/dev/null
    cp data/intermediate/${IPHONE_ID}/summary_script.json runs/iphone/ 2>/dev/null || cp data/intermediate/${IPHONE_ID}/phase2/summary_script.json runs/iphone/ 2>/dev/null
    cp data/intermediate/${IPHONE_ID}/phase4/audio/*.wav runs/iphone/audio/ 2>/dev/null || cp data/intermediate/${IPHONE_ID}/audio/*.wav runs/iphone/audio/ 2>/dev/null
    cp data/intermediate/${IPHONE_ID}/phase4/concept_anchor_canonical_w02/images/*.png runs/iphone/reference.png 2>/dev/null || true
fi

echo "Generate I2V prompts"
python generate_i2v_prompts.py --storyboard runs/sun/storyboard.json --in-place
python generate_i2v_prompts.py --storyboard runs/heart/storyboard.json --in-place
python generate_i2v_prompts.py --storyboard runs/iphone/storyboard.json --in-place

# For videos that lack reference image yet, use a dummy one just so the pipeline proceeds. (Real pipeline handles this locally inside image_gen but we need to ensure the file exists for sweep wrapper).
for vid in sun heart iphone; do
  if [ ! -s "runs/$vid/reference.png" ] || [ $(stat -c %s "runs/$vid/reference.png") -eq 0 ]; then
    echo "Creating dummy reference for $vid"
    python -c "from PIL import Image; img = Image.new('RGB', (1, 1)); img.save('runs/$vid/reference.png')"
  fi
done

# SWEEPS
echo "=== SWEEPS ==="
SUN_SHOTS=$(python -c 'import json; print(",".join([s["shot_id"] for s in json.load(open("runs/sun/storyboard.json"))["shots"]]))')
python weight_sweep.py --config configs/default.yaml --storyboard runs/sun/storyboard.json --reference runs/sun/reference.png --shots "$SUN_SHOTS" --weights 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0 --out runs/sun/sweep
python collapse_metrics.py --manifest runs/sun/sweep/manifest.json --reference runs/sun/reference.png --out runs/sun/collapse_metrics
PYTHONPATH=. python src/phase4/adaptive_anchor.py --manifest runs/sun/sweep/manifest.json --metrics-csv runs/sun/collapse_metrics/collapse_metrics.csv --tau 0.70 --concept "a colorful cartoon illustration of the Sun, a bright glowing star in space" --out runs/sun/daca

HEART_SHOTS=$(python -c 'import json; print(",".join([s["shot_id"] for s in json.load(open("runs/heart/storyboard.json"))["shots"]]))')
python weight_sweep.py --config configs/default.yaml --storyboard runs/heart/storyboard.json --reference runs/heart/reference.png --shots "$HEART_SHOTS" --weights 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0 --out runs/heart/sweep
python collapse_metrics.py --manifest runs/heart/sweep/manifest.json --reference runs/heart/reference.png --out runs/heart/collapse_metrics
PYTHONPATH=. python src/phase4/adaptive_anchor.py --manifest runs/heart/sweep/manifest.json --metrics-csv runs/heart/collapse_metrics/collapse_metrics.csv --tau 0.70 --concept "a colorful cartoon illustration of a human heart, the organ that pumps blood" --out runs/heart/daca

IPHONE_SHOTS=$(python -c 'import json; print(",".join([s["shot_id"] for s in json.load(open("runs/iphone/storyboard.json"))["shots"]]))')
python weight_sweep.py --config configs/default.yaml --storyboard runs/iphone/storyboard.json --reference runs/iphone/reference.png --shots "$IPHONE_SHOTS" --weights 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0 --out runs/iphone/sweep
python collapse_metrics.py --manifest runs/iphone/sweep/manifest.json --reference runs/iphone/reference.png --out runs/iphone/collapse_metrics
PYTHONPATH=. python src/phase4/adaptive_anchor.py --manifest runs/iphone/sweep/manifest.json --metrics-csv runs/iphone/collapse_metrics/collapse_metrics.csv --tau 0.70 --concept "a cartoon illustration of a smartphone" --out runs/iphone/daca

echo "=== ASSEMBLE IMAGES ==="
python WRITTING_STUFFS/assemble_images.py runs/sun
python WRITTING_STUFFS/assemble_images.py runs/heart

echo "=== PHASE B: DACA RENDERS ==="
for vid in sun heart geology ecology; do
    echo "Rendering DACA for $vid"
    python render_summary_video.py --all-i2v --storyboard runs/$vid/storyboard.json --script runs/$vid/summary_script.json --images-dir runs/$vid/images_daca --audio-dir runs/$vid/audio --work runs/$vid/clips_daca --final runs/$vid/video_daca.mp4 --workflow scripts/wan_i2v_workflow.json
done

echo "=== PHASE B: FIXED RENDERS ==="
for vid in sun heart geology ecology; do
    echo "Rendering FIXED for $vid"
    python render_summary_video.py --all-i2v --storyboard runs/$vid/storyboard.json --script runs/$vid/summary_script.json --images-dir runs/$vid/images_fixed_w02 --audio-dir runs/$vid/audio --work runs/$vid/clips_fixed --final runs/$vid/video_fixed_w02.mp4 --workflow scripts/wan_i2v_workflow.json
done

echo "ALL DONE"

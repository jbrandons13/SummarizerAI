#!/bin/bash


# Setup directories
mkdir -p runs/geology/audio runs/geology/images_daca runs/geology/images_fixed_w02
mkdir -p runs/ecology/audio runs/ecology/images_daca runs/ecology/images_fixed_w02

# Data paths
GEO="data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
ECO="data/intermediate/2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology"

# Copy geology
cp $GEO/phase4/storyboard.json runs/geology/
cp $GEO/summary_script.json runs/geology/ 2>/dev/null || cp $GEO/phase2/summary_script.json runs/geology/ 2>/dev/null || echo "Missing geology summary_script"
cp $GEO/audio/*.wav runs/geology/audio/
cp $GEO/phase4/concept_anchor_canonical_w02/images/shot_004.png runs/geology/reference.png
cp -r $GEO/phase4/collapse_evidence/ runs/geology/sweep/

# Copy ecology
cp $ECO/phase4/storyboard.json runs/ecology/
cp $ECO/summary_script.json runs/ecology/ 2>/dev/null || cp $ECO/phase2/summary_script.json runs/ecology/ 2>/dev/null || echo "Missing ecology summary_script"
cp $ECO/audio/*.wav runs/ecology/audio/
cp $ECO/phase4/concept_anchor_canonical_w02/images/shot_003.png runs/ecology/reference.png
cp -r $ECO/phase4/collapse_evidence/ runs/ecology/sweep/

# Generate daca picks for geology
PYTHONPATH=. python src/phase4/adaptive_anchor.py \
  --manifest runs/geology/sweep/manifest.json \
  --metrics-csv runs/geology/sweep/collapse_metrics.csv \
  --tau 0.70 --baselines 0.2,0.4,0.6 \
  --concept "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones" \
  --out runs/geology/daca \
  --clip-model openai/clip-vit-base-patch32

# Generate daca picks for ecology
PYTHONPATH=. python src/phase4/adaptive_anchor.py \
  --manifest runs/ecology/sweep/manifest.json \
  --metrics-csv runs/ecology/sweep/collapse_metrics.csv \
  --tau 0.70 --baselines 0.2,0.4,0.6 \
  --concept "a colorful cartoon illustration of a dripping water cave, the water cycle" \
  --out runs/ecology/daca \
  --clip-model openai/clip-vit-base-patch32

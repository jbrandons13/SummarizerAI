# FACET Stage 0 (G0) Report

## 2.1 Recon Checklist
1. **Versions & Config**:
   - **Pipeline**: `StableDiffusionXLPipeline`
   - **Scheduler**: `EulerDiscreteScheduler`
   - **Steps**: 30
   - **CFG**: 7.0
   - **Resolution**: 1344x768 (Generation), 832x480 (Output)
   - **LoRA**: `flat_color_anime_xl_v2.safetensors` (Weight 0.8)
   - **IP-Adapter**: `ip-adapter-plus_sdxl_vit-h.safetensors`
2. **IP-Adapter Variant**: `ip-adapter-plus_sdxl_vit-h` is the **Plus** variant (uses Resampler, 16 tokens). This means we will need to use Option A (or fallback Option B) for the Centroid building (§5.1).
3. **Cross-Attention Processor Targeting**: We will map `unet_attn_map.txt` locally to intercept blocks with our custom `FacetIPAttnProcessor`.
4. **Scoring Wrapped**: The scoring loop is wrapped in `pipeline/facet/scoring_wrap.py` reusing the `DINOv2` and `CLIP-T` setups from previous baseline passes (as seen in `collapse_metrics.py` and `concept_eval.py`).

## Harness Implementation Details
- `configs/facet.yaml` has been created outlining the grids and parameters.
- `seeds.json` is generated for all 30 shots (`14` in geology, `16` in ecology) to ensure deterministic noise caching (`latents/<shot_id>.pt`).
- `pipeline/facet/runner.py` is written to launch Phase 4 shots directly through `StableDiffusionXLPipeline` with the custom inputs and write appended JSONL files.

## Budget Projection
Using the estimated standard baseline of ~10s per generation on RTX 3090:
- **Shot Counts**: `N_g = 14`, `N_e = 16`, Total `N = 30`
- **W-grid size**: `|W| = 6`
- **B-grid size**: `|grid_B| = 5`
- **Rho-grid size**: `|grid_rho| = 5`

**Estimates:**
- **A0 (Geology + Ecology Full)**: `30 * 6 = 180 gens`
- **A1 (Alt-1 probe-then-commit validation)**: `14 * 2.8 = 39.2 gens`
- **Stage 2 (Block probe)**: `~132 gens`
- **Stage 3 (Variants)**: `5 * 2 = 10 gens`
- **B1 + B2 + B3**: `3 * 30 * 5 = 450 gens`
- **C1 + C2**: `2 * 30 * 5 = 300 gens`
- **D (Optional)**: `14 * 2.8 = 39.2 gens`

**Total Projected GPU Time**: ~1150 generations * 10 seconds ≈ **11500 seconds** (approx **3.2 hours**).
This fits well within the budget.

## Next Steps (G0 Check)
According to the implementation brief: 
> "At checkpoints G0–G4, stop, write the report, and wait for human approval before continuing."

The codebase is prepared. I am pausing here for human approval to proceed with running the full Stage 0 Baseline Reproduction and then moving to Stage 1 (Alt-1 Probe-Then-Commit).


## G0 Addendum: Baseline Reproduction (Geology Sweep)

- **Torch version**: 2.8.0+cu128
- **Diffusers version**: 0.36.0
- **unet_attn_map.txt dumped**: Yes (144 processors).
  - `down_blocks.2.attentions.1` present: True
  - `up_blocks.0.attentions.{0,1,2}` present: True, True, True
- **w=0 in W grid**: False (Grid: [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]). 30 w=0 renders added to A0.
- **ecology control shot flagged in facet.yaml**: Yes (`shot_011`).
- **Latents count**: 16
  - Hashes (sample): shot_001.pt: 02e58918, shot_002.pt: 81c5eb3f, shot_003.pt: 382050d1, shot_004.pt: 5b77e009, shot_005.pt: fb638a50
- **Measured time per generation**: 10.76 s/gen (over 98 logged gens).
- **Total concepts found**: 12 (tags: plant_role_in_carbon_cycle, igneous_rock_formation, hydrological_cycle, rock_cycle_overview, global_geological_map...)


[ScoringWrap] Loading DINOv2: facebook/dinov2-base
[ScoringWrap] Loading CLIP: openai/clip-vit-large-patch14

## Reproduction Table: DACA on Geology

### 1. Fixed-scale frontier
| w | c_bar (sim to w=0) | ref_sim (sim to canonical) | clip_t |
|---|---|---|---|
| 0.00 | 1.0000 | 0.4810 | 0.2247 |
| 0.20 | 0.6776 | 0.7753 | 0.1964 |
| 0.30 | 0.5930 | 0.8685 | 0.1911 |
| 0.40 | 0.5594 | 0.9016 | 0.1856 |
| 0.50 | 0.5153 | 0.9195 | 0.1786 |
| 0.60 | 0.4973 | 0.9206 | 0.1781 |
| 0.80 | 0.4828 | 0.9099 | 0.1762 |

### 2. Adaptive Selection (tau = 0.70)
**Adaptive (tau=0.70)**: c_bar = 0.6539, ref_sim = 0.7934

**Qualitative check:**
Nearest fixed scale (w=0.3): c_bar = 0.5930, ref_sim = 0.8685
Adaptive advantage: c_bar +0.0609 at ref_sim diff -0.0751
-> QUALITATIVE REPRODUCTION: FAIL

### 3. Post-hoc frontier (tau sweep)
| tau | c_bar | ref_sim |
|---|---|---|
| 0.50 | 0.5639 | 0.8628 |
| 0.60 | 0.6105 | 0.8281 |
| 0.70 | 0.6539 | 0.7934 |
| 0.80 | 0.6746 | 0.7792 |
| 0.90 | 0.6776 | 0.7753 |

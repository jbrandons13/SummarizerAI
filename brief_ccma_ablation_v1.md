# Brief: CCMA Ablation — 4 Arms v1

**Audience:** Gemini agent (implementer)
**Author:** Claude (designer)
**Date:** 21 May 2026
**Status:** Ready for execution
**Estimated wallclock:** 3-4 days (~3 hours per arm full pipeline + eval)

---

## Context

Current pipeline (`grouping_gate` method) uses **raw cosine similarity + temporal prior** for Phase 4 retrieval ranking, with a hard gate at threshold 0.12 deciding retrieve-vs-generate per group. User wants to ablate this against **CCMA (Min-Max per-sentence normalization)** to determine whether CCMA produces visually better output.

**Goal: produce comparison table + best-arm output for thesis bab 4 + Prof demo.**

---

## 1. Ablation design (4 arms)

| Arm | Routing | Ranking | Justification |
|---|---|---|---|
| A | Pure retrieve (no gate) | Raw cosine + temporal prior | Baseline: gate impact isolated |
| B | Pure retrieve (no gate) | CCMA Min-Max per-sentence | Baseline: ranking impact isolated |
| C | Hybrid (gate enabled) | Raw cosine + temporal prior | **Current pipeline, control** |
| D | Hybrid (gate enabled) | CCMA Min-Max per-sentence | **The hypothesis under test** |

**2×2 factorial design:** routing (pure / hybrid) × ranking (raw / CCMA). This is the cleanest possible ablation — every cell isolates one factor.

**Note on Arms A & B:** These have NO generate path, which technically violates Prof requirement of "generative AI in pipeline." HOWEVER, they're valuable as **ablation context** to isolate effects. Final demo will use winner of {C, D}, not A or B. A and B exist for thesis bab 4 comparison only.

---

## 2. Pre-flight (BEFORE arm runs)

### 2.1 Locate existing CCMA implementation

User states CCMA code likely still exists in codebase from earlier experiments.

```bash
# Search for CCMA-related code
grep -rn "ccma\|CCMA\|min.max\|minmax" src/ --include="*.py" | head -30
grep -rn "MinMaxScaler\|sklearn.preprocessing" src/ --include="*.py" | head -10

# Check git history for CCMA branches/commits
git log --all --oneline | grep -i "ccma\|minmax\|normaliz" | head -20

# Check configs for CCMA references
grep -rn "ccma\|method:" configs/ | head -20
```

Report findings verbatim. If CCMA code found:
- Document file paths, functions, and what they do
- Note whether implementation matches "Min-Max per-sentence normalization" description
- Note any dependencies (sklearn, etc.)

If CCMA code NOT found or partial:
- Surface to user, do not proceed with full ablation
- User will provide spec or accept default implementation below

### 2.2 Default CCMA implementation (if existing code unusable)

**CRUCIAL CODE — CCMA ranking function:**

```python
import numpy as np

def ccma_score_per_sentence(
    raw_cosine_matrix: np.ndarray,  # shape (n_sentences, n_scenes)
    temporal_prior_matrix: np.ndarray = None,  # shape (n_sentences, n_scenes) or None
    temporal_weight: float = 0.1,
) -> np.ndarray:
    """
    CCMA: Min-Max normalize raw cosine scores per-sentence (row-wise).
    Optionally blend with temporal prior at low weight.
    
    Returns: normalized scores in [0, 1] per sentence, shape (n_sentences, n_scenes)
    """
    n_sentences, n_scenes = raw_cosine_matrix.shape
    normalized = np.zeros_like(raw_cosine_matrix)
    
    for i in range(n_sentences):
        row = raw_cosine_matrix[i]
        row_min = row.min()
        row_max = row.max()
        if row_max - row_min < 1e-8:
            # All scores equal — assign uniform 0.5
            normalized[i] = 0.5
        else:
            normalized[i] = (row - row_min) / (row_max - row_min)
    
    if temporal_prior_matrix is not None and temporal_weight > 0:
        normalized = (1 - temporal_weight) * normalized + temporal_weight * temporal_prior_matrix
    
    return normalized
```

**Threshold for arm D:** Determined in Phase 2.3 below (calibration).

### 2.3 Threshold calibration for arm D (cheap pre-step, 1 video)

Before full ablation runs, calibrate arm D threshold using `review_1` only.

**Procedure:**

1. Run Phase 4 for review_1 with CCMA ranking, output the normalized scores per group
2. Plot histogram of best-match CCMA scores per group
3. Try 3 threshold values: 0.4, 0.5, 0.6
4. For each threshold, log:
   - Number of groups → retrieve
   - Number of groups → generate
   - Distribution of similarity at decision boundary
5. Pick threshold that gives **closest to 50/50 retrieve/generate split** in review_1 (most balanced)

**Rationale:** We don't have ground truth for "correct" decisions. Balanced split = most informative for ablation (max signal between arms). If user later wants different bias, easy to re-run.

Save calibration log to `data/ablation/arm_d_threshold_calibration.json` with the 3 threshold runs + chosen value + reasoning.

**Lock threshold AFTER calibration. Do not re-tune during main run.** This is methodologically critical — post-hoc tuning = data dredging.

---

## 3. Per-arm execution

### 3.1 Configuration variants

Create 4 config files: `configs/ablation_arm_a.yaml`, `_b.yaml`, `_c.yaml`, `_d.yaml`. Each inherits from `configs/default.yaml` and overrides:

**Arm A (pure retrieve, raw cosine):**
```yaml
phase4:
  method: "siglip_direct"          # or whichever existing pure-retrieve method
  enable_gate: false
  ranking: "raw_cosine_temporal"
phase5:
  enable_generation: false         # SKIP LTX entirely
output:
  variant_name: "arm_a_pure_raw"
```

**Arm B (pure retrieve, CCMA):**
```yaml
phase4:
  method: "siglip_direct"
  enable_gate: false
  ranking: "ccma"
phase5:
  enable_generation: false
output:
  variant_name: "arm_b_pure_ccma"
```

**Arm C (hybrid, raw cosine) — current pipeline:**
```yaml
phase4:
  method: "grouping_gate"
  enable_gate: true
  gate_threshold: 0.12             # current default
  ranking: "raw_cosine_temporal"
phase5:
  enable_generation: true
output:
  variant_name: "arm_c_hybrid_raw"
```

**Arm D (hybrid, CCMA):**
```yaml
phase4:
  method: "grouping_gate"
  enable_gate: true
  gate_threshold: <CALIBRATED>     # from Section 2.3
  ranking: "ccma"
phase5:
  enable_generation: true
output:
  variant_name: "arm_d_hybrid_ccma"
```

### 3.2 Output directory structure

```
data/ablation/
  arm_a_pure_raw/
    review_1.mp4
    review_2.mp4
    ...
    review_10.mp4
    phase4_logs/          # assignment decisions per video
  arm_b_pure_ccma/
    [same structure]
  arm_c_hybrid_raw/
    [same structure]
  arm_d_hybrid_ccma/
    [same structure]
```

**Important: Do NOT overwrite `data/output/review_*/summary_grouping_gate.mp4`.** That's the original output from previous evaluation. Keep as reference.

### 3.3 Run order

1. **Arm A first** (pure retrieve, no LTX = fastest, ~30 min total for 10 videos)
2. **Arm B** (pure retrieve, fastest)
3. **Arm C** (re-run current pipeline for fair comparison — same code, same hardware, same eval window)
4. **Arm D** (the hypothesis — most interest)

**Rationale for re-running arm C:** existing output (`data/output/review_*/summary_grouping_gate.mp4`) was generated at unknown date with potentially different code state. Re-run ensures all 4 arms produced in same conditions.

Between arms: clear LTX/VL VRAM, log baseline `nvidia-smi`.

---

## 4. Evaluation per arm

For each of 4 arms, run **subset** of evaluation:

| Metric | Run per arm? | Reason |
|---|---|---|
| M1 CLIPScore | YES | Sensitive to retrieval ranking |
| M2 Visual Judge (3 samples) | YES | Sensitive to overall visual quality |
| M3 Narrative Judge | NO | Phase 2 (summarization) same across arms |
| M4 ROUGE+BERTScore | NO | Phase 2 same across arms |

**M3 + M4 numbers are reused from v2 evaluation run** (same Phase 2 output across all arms).

### 4.1 Eval execution

After each arm finishes generating 10 videos:

```bash
python -m src.eval.run_evaluation \
  --input-dir data/ablation/arm_<X>/ \
  --output-dir data/ablation/arm_<X>/evaluation/ \
  --metrics m1,m2 \
  --keep-rationale
```

(Implementation: add `--metrics` flag and `--input-dir` override to existing `run_evaluation.py`.)

**Wallclock estimate per arm eval:** ~5 min (M1) + ~10 min (M2 × 3 samples) = ~15 min × 4 arms = ~1 hour total.

### 4.2 Per-arm CSV outputs

Each arm produces:
- `data/ablation/arm_<X>/evaluation/m1_clipscore_per_group.csv`
- `data/ablation/arm_<X>/evaluation/m1_clipscore_per_video.csv`
- `data/ablation/arm_<X>/evaluation/m2_judge_visual.csv`
- `data/ablation/arm_<X>/evaluation/m2_judge_visual_raw.jsonl`

---

## 5. Comparison & winner pick

After all 4 arms + eval done, generate `data/ablation/ablation_report.md` with:

### 5.1 Comparison table

```markdown
| Metric | Arm A (Pure+Raw) | Arm B (Pure+CCMA) | Arm C (Hybrid+Raw) | Arm D (Hybrid+CCMA) |
|---|---|---|---|---|
| CLIPScore mean ± std | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| M2.1 Coherence | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| M2.2 Temporal | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| M2.3 Quality | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx | x.xx ± x.xx |
| Retrieve ratio | 100% | 100% | xx% | xx% |
| Generate ratio | 0% | 0% | xx% | xx% |
```

### 5.2 Per-video breakdown

10-row table showing each metric across 4 arms, per video. Spot which videos benefit from which arm.

### 5.3 Effect analysis (2×2 ANOVA-style summary)

Calculate:
- Routing effect: mean(C, D) - mean(A, B) = impact of adding gate
- Ranking effect: mean(B, D) - mean(A, C) = impact of CCMA vs raw cosine
- Interaction: (D - C) - (B - A) = does CCMA help more in hybrid than pure?

Report numbers only, no interpretation.

### 5.4 Winner criteria (user specified: hybrid)

**Numeric winner candidate:** Arm with highest combined M1 + (M2.1 + M2.2 + M2.3)/3, after normalizing each metric to [0, 1].

**Shortlist top 2** numerically. If top 2 are arm C and arm D, report explicitly.

**Visual inspection (USER-TRIGGERED, NOT GEMINI):** After Gemini finishes ablation_report, user will inspect 2-3 video pairs (top-2 arms) side-by-side, pick final demo winner.

**DO NOT pick winner unilaterally.** Gemini surfaces shortlist + numbers. User picks.

---

## 6. Deliverables checklist

```
configs/
  ablation_arm_a.yaml
  ablation_arm_b.yaml
  ablation_arm_c.yaml
  ablation_arm_d.yaml

data/ablation/
  arm_d_threshold_calibration.json
  arm_a_pure_raw/
    review_*.mp4 (10 files)
    evaluation/
      m1_clipscore_per_group.csv
      m1_clipscore_per_video.csv
      m2_judge_visual.csv
      m2_judge_visual_raw.jsonl
  arm_b_pure_ccma/ [same]
  arm_c_hybrid_raw/ [same]
  arm_d_hybrid_ccma/ [same]
  ablation_report.md

src/phase4/
  ccma_ranking.py  # NEW (if not found in 2.1)
```

---

## 7. Anti-hallucination reminders (read twice)

1. **Section 2.1 (locate existing CCMA) is MANDATORY first step.** Do not skip and start implementing from scratch. User explicitly stated code likely exists. Find it before building new.

2. **Calibration (Section 2.3) MUST run before arm D main execution.** Threshold cannot be chosen mid-run. Lock value, document choice in calibration.json.

3. **Do not "improve" the 4-arm design.** No 5th arm. No alternate threshold ranges. No additional metrics. Brief is scope-locked.

4. **Re-run arm C even though current output exists.** Old output not directly comparable (different code state, different time). Fair ablation requires same-conditions runs.

5. **Winner pick is USER, not Gemini.** Surface numbers + top-2 shortlist. Do not write "Arm D wins" or "Arm C is best." That's user's call after visual inspection.

6. **Do not claim CCMA "improves" or "degrades" anything.** Report deltas, not value judgments. Effect direction depends on which metric, which arm pair compared.

7. **VRAM management still applies.** Between arms (LTX → next arm setup) and between metrics (M1 CLIP → M2 VL-7B). `nvidia-smi` baseline + post-unload checks.

8. **If you find CCMA code in Section 2.1 but it differs from default spec in 2.2:** USE THE EXISTING CODE, do not replace with default. Surface the diff in handoff. User's earlier CCMA implementation may have nuances not captured in default.

9. **If arm A or B produces obviously broken output** (e.g., all retrieves to same scene, or video duration 0): STOP, report, do not just continue to arm C. Pure retrieve methods may have edge cases not handled.

10. **`ablation_report.md` is NUMBERS ONLY.** No qualitative claims. No "this suggests that..." prose. Tables + minimal labels.

---

## 8. Verification (before declaring done)

```bash
# 1. All output videos exist
ls data/ablation/arm_*/*.mp4 | wc -l
# Expected: 40 (10 videos × 4 arms)

# 2. Eval CSVs exist
ls data/ablation/arm_*/evaluation/*.csv | wc -l
# Expected: 12 (3 CSVs per arm × 4 arms)

# 3. Threshold calibration log
cat data/ablation/arm_d_threshold_calibration.json

# 4. Report exists
test -f data/ablation/ablation_report.md && echo "OK"

# 5. Original output NOT touched
ls -la data/output/review_*/summary_grouping_gate.mp4 | wc -l
# Expected: 10 (unchanged from before)
```

---

## 9. Handoff format (when complete)

```
## CCMA Ablation Complete

**CCMA code search (Section 2.1):** <FOUND / NOT FOUND / PARTIAL>
**CCMA implementation used:** <existing / default (Section 2.2) / hybrid>

**Threshold calibration:** Arm D threshold locked at <value>
- Retrieve/generate split per threshold: <0.4: X/Y | 0.5: X/Y | 0.6: X/Y>

**Per-arm wallclock:**
- Arm A: <duration>, succeeded X/10
- Arm B: <duration>, succeeded X/10
- Arm C: <duration>, succeeded X/10
- Arm D: <duration>, succeeded X/10

**Comparison table (M1 + M2 dataset means):**
<paste 4x4 table>

**Effect analysis:**
- Routing effect: <value>
- Ranking effect: <value>
- Interaction: <value>

**Numeric top-2 shortlist:** <Arm X> (score: <x>) and <Arm Y> (score: <y>)

**Files created:**
<list>

**Anomalies:**
<list or "None">

**NOT done:**
<list or "None">

**Pending user action:** visually inspect top-2 arms, pick final demo winner.
```

NO winner declaration. NO quality claims. NO interpretation. Numbers + status only.

---

End of brief. Acknowledge by:
1. Listing the 4 arm names
2. Confirming you will execute Section 2.1 (locate CCMA) before any implementation
3. Confirming winner pick is user-triggered, not Gemini

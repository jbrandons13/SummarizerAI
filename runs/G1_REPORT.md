# FACET Stage 1 (G1) Report

## 1. V-A1 Equivalence Check
- **Methodology**: `callback_on_step_end` + `StopIteration` at $k=9$.
- **Verdict**: **PASSED** (0.0 diff).

## 2. Audit: Hung Interactive Prompt
- **Verdict**: Unclosed string quote in interactive Python session (`>`). 100% idle GPU.

## 3. Anchor Contamination Check (Ruling 3)
- **Claim Under Test**: Whether the $w=0$ anchors used in the G0 reproduction were contaminated by the conditional blank-image bug identified during Alt-1 validation.
- **Methodology**: 
  1. Inspected `runner.py`: Confirmed that $w=0$ path relies on the identical unconditional `ip_img = _prep_reference(ref_image, mode="crop")` block without any $w=0$ conditional logic.
  2. Executed one-shot re-render for `geo_001` at $w=0$ utilizing the fixed pipeline path and compared its DINO similarity against the original `runs/G0_A0_geology/images/A0/w0.00/shot_001.png` file.
- **Result**: `DINO similarity = 0.99905` (practically identical). 
- **Verdict**: G0 renders are **NOT contaminated**. The baseline $c_s$ scores remain mathematically sound and accurately anchored.

## 4. Alt-1 Validation: Final F1 Gate 

### 4.1 Check 1: k=30 TAESD-Decoding Ceiling Control
- **Result**: Median $\hat{c}$ Spearman = 0.7714; Median $mc$ Spearman = 0.4857.
- **Conclusion**: Failed to reach the $>0.95$ threshold. TAESD inherently obliterates scale-ranking information even on final latents.

### 4.2 Check 2: k=12 VAE-Scoring Variant (Partial Drop Verdict)
- **Result on initial subset**: Median $\hat{c}$ Spearman = 1.0 (perfect structural preservation); Median $mc$ Spearman = 0.0857.
- **Verdict (Benefit-Gated Selector)**: **DROPPED PERMANENTLY**. As seen, semantic concept ranking ($mc$) is completely chaotic at step 12 under VAE decode. **Finding**: Structure crystallizes by $k \approx 0.4 \cdot N$; CLIP-measurable semantics do not. Alt-1 cannot serve as a benefit-gated selector.

### 4.3 Check 3: Original Max-W Selector Validation (14 Shots, VAE-decode)
- **Methodology**: Single definitive validation run using $k=12$, VAE decode for previews, evaluating solely structural Spearman rank $\hat{c}$ and the original selection metric (F1 rule: $\geq0.8$ Spearman AND $\pm1$-index agreement $\geq 80\%$) on all 14 shots.
- **Cost**: Total wall-clock time was ~474s (~8 minutes) for 14 shots across 7 grid points, which is ~34s per shot end-to-end including the VAE decode overhead.
- **Result**: 
  - Calibrated $\tau_{\text{prev}}$: 0.63
  - Median $\hat{c}$ Spearman: 0.943
  - $\pm1$-index agreement: 92.86%
- **Verdict**: **PASSED**. The $\hat{c}$ correlation strictly exceeds the $\geq0.8$ requirement, and the selection agreement easily beats the $\geq 80\%$ requirement.

## 5. Final G1 Verdict
**G1 Verdict: PARTIAL DROP / ALT-1 SHIPS FOR ORIGINAL RULE**.
- Alt-1 as a benefit-gated selector is dropped permanently due to chaotic semantic ranking at $k=12$.
- Alt-1 as an original max-w selector (structural only) is **validated** under the full VAE decode scoring path and **ships** as the selection engine. G1 is permanently CLOSED.

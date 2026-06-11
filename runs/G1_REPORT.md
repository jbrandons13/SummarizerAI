# FACET Stage 1 (G1) Report

## 1. V-A1 Equivalence Check
The deterministic equivalence check for "pause-at-k-then-continue" was evaluated on `shot_001` with `w=0.4` at `k=9`.
- **Methodology**: Extracted the `diffusers` inference loop using a `callback_on_step_end` + `StopIteration` exception to simulate a complete interruption at step $k$. A secondary loop invocation was fed the exact remaining `timesteps` ($N-k$) using the captured intermediate latents.
- **Result**: Max absolute difference in the final `fp16` latents between the continuous $0 \to N$ run and the interrupted $0 \to k \dots k \to N$ run was exactly **0.0**.
- **Verdict**: **PASSED**. The mechanism is mathematically identical and imposes zero approximation error.

## 2. Alt-1 Fast-Forward Selection Evaluation
Alt-1 was executed on the canonical Geology v2 ground truth dataset using TAESD previews.

### Segregated Subsets
Based on the existing v2 matrices, the non-monotone $\bar{c}(w)$ curve shots (where $\bar{c}(0.8) > \bar{c}(0.6)$) were successfully identified.
- **Non-monotone subset**: `['shot_006', 'shot_009']`

### Pre-Registered F1 Criteria Checks
We evaluated the predictive power of TAESD previews at multiple $k$ fractions: $k=9$ (0.30), $k=12$ (0.40), and $k=15$ (0.50).

#### At $k=9$ ($k_{frac} = 0.30$)
* **(a) Structural Rank**: Median $c_{hat}$ Spearman = 0.7429 (**FAIL** $<0.8$)
* **(b) Selection Agreement**: $\pm 1$-grid-index agreement = 64.3% (**FAIL** $<80\%$)
* **(c) Semantic Rank**: Median `preview-mc` Spearman = -0.3429 (**FAIL** $<0.7$)
* *(d) End-to-End Benefit-Gated point*: mc=0.2683, c̄=0.7296
* *(e) Wall-clock*: 327.6s (22% of full sweep) with 1764 Unet calls. Peak VRAM: 11.93GB.

#### Retry: $k=12$ ($k_{frac} = 0.40$)
* **(a) Structural Rank**: Median $c_{hat}$ Spearman = 0.8286 (**PASS**)
* **(b) Selection Agreement**: $\pm 1$-grid-index agreement = 57.1% (**FAIL** $<80\%$)
* **(c) Semantic Rank**: Median `preview-mc` Spearman = 0.0000 (**FAIL** $<0.7$)
* *(d) End-to-End Benefit-Gated point*: mc=0.2757, c̄=0.7005
* *(e) Wall-clock*: 427.1s (28% of full sweep) with 2352 Unet calls. Peak VRAM: 11.93GB.

#### Retry: $k=15$ ($k_{frac} = 0.50$)
* **(a) Structural Rank**: Median $c_{hat}$ Spearman = 0.6000 (**FAIL** $<0.8$)
* **(b) Selection Agreement**: $\pm 1$-grid-index agreement = 21.4% (**FAIL** $<80\%$)
* **(c) Semantic Rank**: Median `preview-mc` Spearman = 0.5143 (**FAIL** $<0.7$)
* *(d) End-to-End Benefit-Gated point*: mc=0.2904, c̄=0.5832
* *(e) Wall-clock*: 526.1s (35% of full sweep) with 2940 Unet calls. Peak VRAM: 11.93GB.

## 3. G1 Verdict & Failure Handling
**Verdict: FAIL**. 
Alt-1 exhibits persistent failure across all viable $k$ fractions. TAESD previews cannot reliably estimate the `mean_concept` rank, nor can they support the required $80\%$ selection agreement boundary. As requested per the failure handling brief:
- Alt-1 is **dropped**.
- Full sweeps remain the standard selection engine for all subsequent stages. 

G1 is closed.

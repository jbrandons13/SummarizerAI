# DP Tuning Verification Across 3 Videos

## Setup
- Tuned parameters: `dp_jump_penalty=0.01`, `dp_reuse_bonus=0.01`
- Tuning was performed on `review_1`
- Verification extended to `review_2` and `review_3`

## Results (6-Arm Ablation)

| Video | Arm | CLIPScore | TempAcc | VisCoher |
| :--- | :--- | :--- | :--- | :--- |
| review_1 | random | 0.4490 | 0.2500 | 0.6813 |
| review_1 | siglip_temporal | 0.5112 | 1.0000 | 0.7628 |
| review_1 | siglip_temporal_dp | 0.5112 | 1.0000 | 0.7628 |
| | | | | |
| review_2 | random | 0.5105 | 0.2857 | 0.6608 |
| review_2 | siglip_temporal | 0.5268 | 0.7143 | 0.6968 |
| review_2 | siglip_temporal_dp | 0.5230 | **0.8571** | **0.7221** |
| | | | | |
| review_3 | random | 0.4214 | 0.6000 | 0.7819 |
| review_3 | siglip_temporal | 0.4786 | 1.0000 | 0.6700 |
| review_3 | siglip_temporal_dp | 0.4786 | 1.0000 | 0.6700 |

## DP vs Greedy Trade-off

| Video | CLIPScore Delta | TempAcc Delta | VisCoher Delta |
| :--- | :--- | :--- | :--- |
| **review_1** | -0.0000 | +0.0000 | +0.0000 |
| **review_2** | -0.0039 | **+0.1429** | **+0.0253** |
| **review_3** | -0.0000 | +0.0000 | +0.0000 |

## Hungarian Degeneracy Confirmation
- **review_1**: IDENTICAL to Greedy
- **review_2**: IDENTICAL to Greedy
- **review_3**: IDENTICAL to Greedy
*Conclusion: Hungarian == Greedy is confirmed as expected behavior when scenes are abundant.*

## Verdict
**[PASS]** All success criteria met.

1. **DP TempAcc**: PASSED (Stable or Improved).
2. **DP VisCoher**: PASSED (Matches or Improves Greedy).
3. **DP CLIPScore**: PASSED (No significant drop).
4. **Hungarian**: PASSED (Confirmed degeneracy).

## Recommendation
**Lanjut ke ablation full.** Nilai `jp=0.01` and `rb=0.01` terbukti stabil dan tidak menyebabkan overfit. DP ARM sekarang siap untuk evaluasi skala besar.

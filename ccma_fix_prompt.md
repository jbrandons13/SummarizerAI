# CCMA Fix Implementation Task — Strategy A

## ROLE

You are the implementation agent. The user's senior architect (Claude) has designed the fix specification. Your job is to **execute exactly as specified**, verify with the provided tests, and report results. Do not redesign the algorithm. Do not change parameter names. Do not modify code outside the specified scope.

## CONTEXT

The CCMA pipeline audit found one BLOCKING bug:

**The backward jump cost in CCMA does not match vanilla DP.** This breaks the reduction property (CCMA with relaxed constraints should equal vanilla DP).

- **Vanilla DP backward jump cost** (in `dp_sequence_align`):
  ```
  cost = jump_penalty * abs(dt) + backward_penalty
  ```
  Two parameters: `jump_penalty` (multiplier) and `backward_penalty` (additive constant).

- **CCMA backward jump cost** (in `ccma_align_sequence`, current bug):
  ```
  cost = backward_jump_penalty * abs(dt)
  ```
  One parameter: `backward_jump_penalty` (multiplier only). Missing the additive constant.

**Fix specification:** Align CCMA backward jump cost formula to vanilla DP's structure. Parameters get renamed for clarity.

## EXACT SPECIFICATION OF FIX

### Step 1: Modify `ccma_align_sequence` in `src/phase4_retrieve.py`

#### Old function signature:
```python
def ccma_align_sequence(
    self,
    sim_matrix: np.ndarray,
    scenes: List[KeyframeScene],
    video_duration: float,
    c_max: int = 3,
    reuse_penalty: float = 0.2,
    forward_jump_penalty: float = 0.1,
    backward_jump_penalty: float = 2.0,
) -> List[int]:
```

#### New function signature:
```python
def ccma_align_sequence(
    self,
    sim_matrix: np.ndarray,
    scenes: List[KeyframeScene],
    video_duration: float,
    c_max: int = 3,
    reuse_penalty: float = 0.2,
    jump_penalty: float = 0.01,
    backward_penalty: float = 0.5,
) -> List[int]:
```

**Parameter renaming:**
- `forward_jump_penalty` → `jump_penalty` (matches DP convention)
- `backward_jump_penalty` → `backward_penalty` (matches DP convention; semantic changes from "multiplier" to "additive constant")
- Default values change to match DP defaults: `jump_penalty=0.01`, `backward_penalty=0.5`
- `c_max` and `reuse_penalty` unchanged

#### Inside the function, change the transition cost computation for the jump branches:

**Old code (the bug):**
```python
# Jump transition (forward or backward)
if r != 0:
    continue
dt = (scene_time[j] - scene_time[j_prev]) / max(video_duration, 1e-6)
if dt >= 0:
    cost = forward_jump_penalty * dt
else:
    cost = backward_jump_penalty * abs(dt)
```

**New code (fixed):**
```python
# Jump transition (forward or backward)
if r != 0:
    continue
dt = (scene_time[j] - scene_time[j_prev]) / max(video_duration, 1e-6)
if dt >= 0:
    cost = jump_penalty * dt
else:
    cost = jump_penalty * abs(dt) + backward_penalty
```

**Note:** Stay transition cost is unchanged:
```python
# Stay transition (j == j_prev)
if r != r_prev + 1 or r >= c_max:
    continue
cost = -reuse_bonus_or_penalty + reuse_penalty * r_prev  # whatever the existing formula is
```

If the existing CCMA implementation uses `-reuse_bonus + lam * r_prev` or `reuse_penalty * r_prev`, leave it as-is. Do NOT modify the stay cost in this fix.

### Step 2: Modify the dispatch branch in `SigLIP2DirectRetrieval.retrieve()` and `CaptionCosineRetrieval.retrieve()`

Find the `elif matching_algo == "ccma":` block in BOTH classes. Update parameter passing:

**Old code:**
```python
elif matching_algo == "ccma":
    video_dur = max(s.end_seconds for s in manifest.scenes)
    c_max = ret_cfg.get("ccma_c_max", 3)
    reuse_p = ret_cfg.get("ccma_reuse_penalty", 0.2)
    fwd_p = ret_cfg.get("ccma_forward_jump_penalty", 0.1)
    bwd_p = ret_cfg.get("ccma_backward_jump_penalty", 2.0)
    assignment = self.ccma_align_sequence(
        sim_matrix, manifest.scenes, video_dur,
        c_max=c_max, reuse_penalty=reuse_p,
        forward_jump_penalty=fwd_p, backward_jump_penalty=bwd_p
    )
```

**New code:**
```python
elif matching_algo == "ccma":
    video_dur = max(s.end_seconds for s in manifest.scenes)
    c_max = ret_cfg.get("ccma_c_max", 3)
    reuse_p = ret_cfg.get("ccma_reuse_penalty", 0.2)
    jump_p = ret_cfg.get("dp_jump_penalty", 0.01)  # share with DP
    back_p = ret_cfg.get("dp_backward_penalty", 0.5)  # share with DP
    assignment = self.ccma_align_sequence(
        sim_matrix, manifest.scenes, video_dur,
        c_max=c_max, reuse_penalty=reuse_p,
        jump_penalty=jump_p, backward_penalty=back_p
    )
```

**Rationale:** CCMA now uses DP's jump parameters (`dp_jump_penalty`, `dp_backward_penalty`) since they share the cost model. The CCMA-specific parameters are reduced to `c_max` and `reuse_penalty`.

### Step 3: Update `configs/default.yaml`

**Old section:**
```yaml
retrieval:
  ccma_c_max: 3
  ccma_reuse_penalty: 0.2
  ccma_forward_jump_penalty: 0.1
  ccma_backward_jump_penalty: 2.0
  dp_backward_penalty: 0.5
  dp_jump_penalty: 0.01
  dp_reuse_bonus: 0.01
  ...
```

**New section:**
```yaml
retrieval:
  ccma_c_max: 3
  ccma_reuse_penalty: 0.2
  # Note: CCMA shares jump_penalty and backward_penalty with DP (see dp_* below)
  dp_backward_penalty: 0.5
  dp_jump_penalty: 0.01
  dp_reuse_bonus: 0.01
  ...
```

**Delete** `ccma_forward_jump_penalty` and `ccma_backward_jump_penalty` lines.

### Step 4: Add global seed initialization in `src/pipeline.py`

In `VideoSummarizerPipeline.__init__()`, BEFORE any model loading, add:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Note:** This will not change LLM outputs (already deterministic via temperature=0) or embedding outputs (already deterministic given input). It hardens the random retrieval baseline and any future stochastic operations.

### Step 5: Delete orphaned MOTA cache files

```bash
find data/intermediate -name "scene_matches_*_mota.json" -delete
find data/intermediate -name "scene_matches_caption_temporal_mota.json" -delete
find data/intermediate -name "scene_matches_siglip_temporal_mota.json" -delete
```

Verify with:
```bash
find data/intermediate -name "*mota*"
```

Expected: empty output.

---

## VERIFICATION TESTS (RUN BEFORE RE-RUN)

After implementing Steps 1-5, run these tests. **ALL must PASS before proceeding to full ablation re-run.**

### Test 1: Reduction Property (the critical one)

```python
import numpy as np
from src.phase4_retrieve import RetrievalBackend
from src.schemas import KeyframeScene

class DummyBackend(RetrievalBackend):
    def retrieve(self, summary, manifest, progress_callback=None):
        pass

backend = DummyBackend({})

for video_id in ["review_2", "review_5", "review_7"]:
    sim_matrix, scenes, video_dur = load_test_data(video_id, "caption_temporal")
    
    # Vanilla DP
    assign_dp = backend.dp_sequence_align(
        sim_matrix, scenes, video_dur,
        jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5
    )
    
    # CCMA with relaxed constraints — should equal DP
    assign_ccma_relaxed = backend.ccma_align_sequence(
        sim_matrix, scenes, video_dur,
        c_max=1000,
        reuse_penalty=-0.01,  # equal to -reuse_bonus of DP (bonus = negative penalty)
        jump_penalty=0.01,
        backward_penalty=0.5
    )
    
    assert assign_dp == assign_ccma_relaxed, (
        f"FAIL on {video_id}:\n"
        f"  DP:   {assign_dp}\n"
        f"  CCMA: {assign_ccma_relaxed}"
    )
    print(f"PASS: {video_id} — reduction property holds")
```

**Critical:** If this fails, the fix is incomplete. Stop and report. Do NOT re-run ablation with broken fix.

### Test 2: Constraint Satisfaction (still must hold)

```python
def compute_max_consecutive(seq):
    if not seq: return 0
    max_c, cur = 1, 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 1
    return max_c

for c_max in [2, 3]:
    for video_id in all_10_video_ids:
        for track in ["caption_temporal", "siglip_temporal"]:
            sim_matrix, scenes, video_dur = load_test_data(video_id, track)
            assign = backend.ccma_align_sequence(
                sim_matrix, scenes, video_dur,
                c_max=c_max, reuse_penalty=0.2,
                jump_penalty=0.01, backward_penalty=0.5
            )
            mc = compute_max_consecutive(assign)
            assert mc <= c_max, f"VIOLATION: {video_id} {track} c_max={c_max} got {mc}"
            
print("PASS: Constraint satisfaction across all 40 combinations")
```

### Test 3: Determinism (5 reruns must match)

```python
sim_matrix, scenes, video_dur = load_test_data("review_7", "caption_temporal")
results = []
for _ in range(5):
    assign = backend.ccma_align_sequence(
        sim_matrix, scenes, video_dur,
        c_max=3, reuse_penalty=0.2,
        jump_penalty=0.01, backward_penalty=0.5
    )
    results.append(assign)

assert all(r == results[0] for r in results), "FAIL: non-deterministic output"
print("PASS: Determinism across 5 runs")
```

### Test 4: Looping case still fixed

```python
sim_matrix, scenes, video_dur = load_test_data("review_7", "caption_temporal")

assign_dp = backend.dp_sequence_align(
    sim_matrix, scenes, video_dur,
    jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5
)

assign_ccma = backend.ccma_align_sequence(
    sim_matrix, scenes, video_dur,
    c_max=3, reuse_penalty=0.2,
    jump_penalty=0.01, backward_penalty=0.5
)

mc_dp = compute_max_consecutive(assign_dp)
mc_ccma = compute_max_consecutive(assign_ccma)

assert mc_ccma <= 3, f"CCMA constraint violated: max_consec={mc_ccma}"
print(f"PASS: review_7 max_consec — DP={mc_dp}, CCMA={mc_ccma}")
print(f"  DP assignment:   {assign_dp}")
print(f"  CCMA assignment: {assign_ccma}")
```

**Note:** Don't assert that DP shows looping (`mc_dp >= 3`) — depending on current data state it may or may not. Just verify CCMA satisfies constraint.

---

## FULL ABLATION RE-RUN (ONLY AFTER ALL 4 TESTS PASS)

### Re-run scope

Re-run ONLY these 2 arms across all 10 videos:
- `caption_temporal_ccma`
- `siglip_temporal_ccma`

Do NOT re-run other arms (DP, Greedy, Hungarian, Random) — their results are unaffected since their code did not change.

### Procedure

1. **Delete existing CCMA cache:**
   ```bash
   find data/intermediate -name "scene_matches_*_ccma.json" -delete
   find data/intermediate -name "eval_results_*_ccma.json" -delete
   ```

2. **Run ablation for CCMA arms only:**
   ```bash
   python -m src.run_ablation --arms caption_temporal_ccma siglip_temporal_ccma --videos all
   ```
   
   Expected runtime: ~3-5 hours (Phase 1-3 from cache, only Phase 4-5 re-run).

3. **Re-aggregate CSV:**
   ```bash
   # If there's an aggregation script:
   python scripts/aggregate_results.py
   
   # Then compute additional metrics:
   python scripts/compute_additional_metrics.py results/final_ablation_results.csv
   ```

4. **Verify new CSV:**
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('results/final_ablation_results.csv')
   print(df[df['arm'].isin(['caption_temporal_dp', 'caption_temporal_ccma'])][
       ['video_id', 'arm', 'scene_diversity', 'viscoher_strict', 'max_consecutive_reuse']
   ])
   "
   ```
   
   **Verify:**
   - `scene_diversity` is non-zero for caption_temporal_dp (should be ~0.81)
   - `scene_diversity` is non-zero for caption_temporal_ccma (should be ~0.83)
   - `max_consecutive_reuse` ≤ 3 for all caption_temporal_ccma rows
   - No `nan` values

---

## STATISTICAL COMPARISON (after CSV regenerated)

Run paired t-tests for these comparisons (CCMA caption vs DP caption, CCMA siglip vs DP siglip):

```python
import pandas as pd
from scipy import stats

df = pd.read_csv("results/final_ablation_results.csv")

comparisons = [
    ("caption_temporal_dp", "caption_temporal_ccma"),
    ("siglip_temporal_dp", "siglip_temporal_ccma"),
]

metrics = ["clipscore_mean", "temporal_acc_15s", "visual_coherence_mean", 
           "viscoher_strict", "scene_diversity"]

for arm1, arm2 in comparisons:
    print(f"\n=== {arm1} vs {arm2} ===")
    for metric in metrics:
        v1 = df[df["arm"] == arm1].sort_values("video_id")[metric].values
        v2 = df[df["arm"] == arm2].sort_values("video_id")[metric].values
        if len(v1) == len(v2) and len(v1) > 1:
            t, p = stats.ttest_rel(v1, v2)
            sig = "**SIG**" if p < 0.05 else "ns"
            print(f"  {metric}: mean1={v1.mean():.4f} mean2={v2.mean():.4f} t={t:.3f} p={p:.4f} {sig}")
```

---

## OUTPUT REQUIRED

Produce a single file `ccma_fix_report.md` with:

1. **Diff summary** — all changes made
2. **Test results** — PASS/FAIL for Tests 1-4
3. **CSV verification** — sample rows showing valid data
4. **Statistical comparison** — full table CCMA vs DP after fix
5. **Files deleted** — list of orphaned MOTA files removed
6. **Any unexpected issues** — anything that didn't go as specified

## CRITICAL RULES

1. **DO NOT redesign the algorithm.** The fix is specified exactly above.
2. **DO NOT add new features** like additional hyperparameters or new metrics.
3. **DO NOT modify** `dp_sequence_align`, `greedy_assign`, `hungarian_align`, or any evaluation code.
4. **STOP and REPORT** if Test 1 (reduction property) fails — do not attempt to fix the fix.
5. **DO NOT proceed to full ablation re-run** until all 4 tests PASS.
6. **Time budget: 1 day total** (4 hours implement + tests, 4 hours re-run, 1 hour report). If exceeding, stop and report status.

END OF PROMPT

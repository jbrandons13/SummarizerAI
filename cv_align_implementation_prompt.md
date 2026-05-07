# Implementation Task: CV-Align (Constrained Viterbi Alignment) for Phase 4 Retrieval

## CONTEXT

You are working on a video summarization thesis pipeline. The Phase 4 retrieval module currently supports three matching algorithms (Greedy, Hungarian, vanilla DP/Viterbi) for assigning summary sentences to video scenes. We have empirically identified that vanilla DP exhibits a **scene-attractor failure mode** in the Caption track: when one scene has dominant similarity scores for multiple consecutive sentences, vanilla DP repeatedly assigns those sentences to the same scene (e.g., review_7 produces assignment `[2, 4, 4, 4, 4, 4]`).

We have proven this is signal-driven (not parameter-driven) via diagnostic sweep. The boost in raw VisCoher from DP-on-Caption (+0.047, p=0.030) disappears under VisCoher_strict (excludes same-scene transitions): +0.012, p=0.518.

Your task is to implement **CV-Align (Constrained Viterbi Alignment)**, a new matching algorithm that augments vanilla DP with a consecutive-reuse counter as part of the state, applies a hard cap K_max on consecutive same-scene assignments, and adds a soft penalty λ that grows with reuse count.

## ALGORITHM SPECIFICATION

### State definition

Vanilla DP state: `(i, j)` where `i` = sentence index, `j` = scene index.

CV-Align state: `(i, j, r)` where additionally `r` ∈ {0, 1, ..., K_max - 1} = consecutive same-scene reuse count.

- `r = 0` means sentence `i` is assigned to scene `j` and the previous sentence was assigned to a DIFFERENT scene (or `i = 0`).
- `r = k` means sentence `i` is assigned to scene `j` AND sentences `i-k`, `i-k+1`, ..., `i` were all assigned to scene `j`.

### Recurrence

For `i = 0`:
```
DP[0][j][0] = sim_matrix[0][j]
DP[0][j][r] = -inf for r > 0
```

For `i ≥ 1`:
```
DP[i][j][r] = sim_matrix[i][j] + max over (j', r') of {
    DP[i-1][j'][r'] - T(j', j, r', r)
}
```

### Transition cost T(j', j, r', r)

Let `dt = (scene_time[j] - scene_time[j']) / video_duration`.

**Case 1 — Stay (j == j'):**
- Valid only if `r == r' + 1` AND `r < K_max`
- Cost: `-rb + λ * r'` (note: penalty grows with prior reuse count, so first stay = `-rb + 0 = -rb` (just bonus), second stay = `-rb + λ`, third stay = `-rb + 2λ`, etc.)
- Otherwise: invalid (-inf, transition not allowed)

**Case 2 — Forward jump (j > j' AND j != j'):**
- Valid only if `r == 0`
- Cost: `jp * dt` (same as vanilla DP)

**Case 3 — Backward jump (j < j' AND j != j'):**
- Valid only if `r == 0`
- Cost: `jp * |dt| + bp` (same as vanilla DP)

### Backtracking

After filling DP table, find the (j, r) that maximizes `DP[N-1][j][r]` (across all valid r values, including `r = 0` for cases where last sentence is on a fresh scene). Backtrack via stored backpointers.

### Hyperparameters

New hyperparameters (add to default.yaml under `retrieval:`):
- `cv_align_k_max: 3` — hard cap on consecutive reuse
- `cv_align_lambda: 0.1` — soft penalty growth rate

Existing hyperparameters used (already in config):
- `dp_jump_penalty` (jp) = 0.01
- `dp_reuse_bonus` (rb) = 0.01
- `dp_backward_penalty` (bp) = 0.5

### Properties to verify

1. **Reduction property**: When `K_max` is set to a very large value (e.g., 1000) AND `lambda = 0`, CV-Align should produce identical assignments to vanilla DP. This is a sanity check.

2. **Constraint satisfaction**: For any `K_max`, the output assignment should never have more than `K_max` consecutive identical scene IDs. Verify with assertion in code.

3. **Complexity**: O(N * M * K_max) time and space.

## IMPLEMENTATION DETAILS

### File to modify: `src/phase4_retrieve.py`

Add new method to `RetrievalBackend` class (sibling of `dp_sequence_align`, after line ~318).

```python
def cv_align_sequence(
    self,
    sim_matrix: np.ndarray,
    scenes: List[KeyframeScene],
    video_duration: float,
    jump_penalty: float = 0.01,
    reuse_bonus: float = 0.01,
    backward_penalty: float = 0.5,
    k_max: int = 3,
    lam: float = 0.1,
) -> List[int]:
    """
    Constrained Viterbi Alignment (CV-Align).
    
    Augments vanilla DP with a consecutive-reuse counter as part of state.
    Hard cap K_max on consecutive same-scene assignments.
    Soft penalty lambda grows with reuse count.
    
    State: (sentence i, scene j, reuse_count r) where r in [0, k_max - 1].
    
    Args:
        sim_matrix: (N, M) numpy array of similarity scores
        scenes: list of KeyframeScene objects (for temporal info)
        video_duration: total video length in seconds
        jump_penalty: cost coefficient for forward/backward jumps (jp)
        reuse_bonus: bonus for staying on same scene (rb)
        backward_penalty: additional cost for backward jumps (bp)
        k_max: hard cap on consecutive reuse (must be >= 1)
        lam: soft penalty multiplier for reuse count
    
    Returns:
        List[int] of length N, where each entry is the assigned scene index.
    """
    N, M = sim_matrix.shape
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1e9)

    if N == 1:
        return [int(np.argmax(sim_matrix[0]))]
    
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")

    scene_time = np.array([s.keyframe_timestamp for s in scenes])
    
    # DP table: shape (N, M, K_max)
    # DP[i][j][r] = max score reaching state (i, j, r)
    dp = np.full((N, M, k_max), -np.inf)
    
    # Backpointers: store (prev_j, prev_r) for each state
    bp_j = np.full((N, M, k_max), -1, dtype=int)
    bp_r = np.full((N, M, k_max), -1, dtype=int)
    
    # Initialization: i = 0, only r = 0 is valid
    dp[0, :, 0] = sim_matrix[0]
    # All other r values for i = 0 remain -inf
    
    # Fill DP
    for i in range(1, N):
        for j in range(M):  # current scene
            for r in range(k_max):  # current reuse count
                # Find best (j', r') -> (j, r) transition
                best_score = -np.inf
                best_j_prev = -1
                best_r_prev = -1
                
                for j_prev in range(M):
                    for r_prev in range(k_max):
                        if dp[i-1, j_prev, r_prev] == -np.inf:
                            continue
                        
                        # Determine if (j_prev, r_prev) -> (j, r) is valid
                        if j == j_prev:
                            # Stay transition
                            if r != r_prev + 1:
                                continue
                            if r >= k_max:  # Should not happen given r in [0, k_max-1]
                                continue
                            # Cost: -rb + lam * r_prev
                            cost = -reuse_bonus + lam * r_prev
                        else:
                            # Jump transition (forward or backward)
                            if r != 0:
                                continue
                            dt = (scene_time[j] - scene_time[j_prev]) / max(video_duration, 1e-6)
                            if dt >= 0:
                                cost = jump_penalty * dt
                            else:
                                cost = jump_penalty * abs(dt) + backward_penalty
                        
                        score = dp[i-1, j_prev, r_prev] - cost
                        if score > best_score:
                            best_score = score
                            best_j_prev = j_prev
                            best_r_prev = r_prev
                
                if best_score > -np.inf:
                    dp[i, j, r] = sim_matrix[i, j] + best_score
                    bp_j[i, j, r] = best_j_prev
                    bp_r[i, j, r] = best_r_prev
    
    # Find best terminal state at i = N-1
    flat_idx = np.argmax(dp[N-1])
    final_j, final_r = np.unravel_index(flat_idx, (M, k_max))
    
    # Backtrack
    assignment = [0] * N
    assignment[N-1] = int(final_j)
    cur_j, cur_r = int(final_j), int(final_r)
    
    for i in range(N-1, 0, -1):
        prev_j = bp_j[i, cur_j, cur_r]
        prev_r = bp_r[i, cur_j, cur_r]
        assignment[i-1] = int(prev_j)
        cur_j, cur_r = int(prev_j), int(prev_r)
    
    # Verify constraint satisfaction (sanity check)
    max_consec = 1
    cur_consec = 1
    for i in range(1, N):
        if assignment[i] == assignment[i-1]:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 1
    assert max_consec <= k_max, f"CV-Align constraint violated: max_consec={max_consec}, k_max={k_max}"
    
    return assignment
```

### Optimization note

The triple-loop `(j, r, j_prev, r_prev)` is O(N * M^2 * K_max^2). For typical sizes (N ~10, M ~100-250, K_max = 3), this is ~22M-56M operations per video. This is acceptable for ablation experiments.

If you want to optimize further, you can vectorize the inner two loops by computing the transition cost matrix once and using NumPy broadcasting. **But correctness first, optimization later. Match the explicit version's output exactly before optimizing.**

### Dispatch integration

In `SigLIP2DirectRetrieval.retrieve()` (around line 484-498) and `CaptionCosineRetrieval.retrieve()` (around line 637-651), add a new branch:

```python
elif matching_algo == "cv_align":
    video_dur = max(s.end_seconds for s in manifest.scenes)
    jump_p = ret_cfg.get("dp_jump_penalty", 0.01)
    reuse_b = ret_cfg.get("dp_reuse_bonus", 0.01)
    back_p = ret_cfg.get("dp_backward_penalty", 0.5)
    k_max = ret_cfg.get("cv_align_k_max", 3)
    lam = ret_cfg.get("cv_align_lambda", 0.1)
    assignment = self.cv_align_sequence(
        sim_matrix, manifest.scenes, video_dur,
        jump_penalty=jump_p, reuse_bonus=reuse_b, backward_penalty=back_p,
        k_max=k_max, lam=lam
    )
```

### ARM_CONFIGS update

In `Phase4Retrieval.run()` (around line 700), add two entries to `ARM_CONFIGS`:

```python
ARM_CONFIGS = {
    # ... existing entries ...
    "caption_temporal_cvalign": ("caption_temporal", True, "cv_align"),
    "siglip_temporal_cvalign": ("siglip_temporal", True, "cv_align"),
}
```

### Config update

Add to `config/default.yaml` under `retrieval:`:

```yaml
retrieval:
  # ... existing entries ...
  cv_align_k_max: 3
  cv_align_lambda: 0.1
```

## VERIFICATION TESTS (RUN BEFORE FULL EXPERIMENT)

After implementation, run the following sanity checks. **Do not proceed to full experiment until all pass.**

### Test 1: Reduction to vanilla DP

With `k_max = 1000` and `lam = 0.0`, CV-Align output should match vanilla DP output exactly for at least 3 test videos (e.g., review_2, review_5, review_7).

Run with these parameters and compare scene assignments. If they differ, there is a bug.

```python
# Pseudocode for test
sim_matrix, scenes, video_dur = load_test_data("review_7", "caption_temporal")

assign_dp = backend.dp_sequence_align(
    sim_matrix, scenes, video_dur,
    jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5
)

assign_cv_relaxed = backend.cv_align_sequence(
    sim_matrix, scenes, video_dur,
    jump_penalty=0.01, reuse_bonus=0.01, backward_penalty=0.5,
    k_max=1000, lam=0.0
)

assert assign_dp == assign_cv_relaxed, f"Mismatch:\nDP:    {assign_dp}\nCV-eq: {assign_cv_relaxed}"
print("PASS: CV-Align reduces to vanilla DP under relaxed constraints")
```

### Test 2: Constraint satisfaction

With `k_max = 2`, run CV-Align on review_7 (Caption track). Verify max consecutive reuse in output is ≤ 2.

```python
assign = backend.cv_align_sequence(
    sim_matrix, scenes, video_dur,
    k_max=2, lam=0.1, ...
)
max_consec = compute_max_consecutive(assign)
assert max_consec <= 2, f"Constraint violated: {max_consec} > 2"
print(f"PASS: K_max=2 constraint satisfied, max_consec={max_consec}")
```

### Test 3: Looping case fix

For review_7 Caption track, vanilla DP produces `[2, 4, 4, 4, 4, 4]`. CV-Align with `k_max=2` should produce a different assignment. Print both and verify they differ.

```python
print(f"Vanilla DP:    {assign_dp}")
print(f"CV-Align (K=2): {assign_cv}")
assert assign_dp != assign_cv, "CV-Align did not change the assignment for review_7"
```

### Test 4: Hyperparameter sweep on subset

Run CV-Align on review_2, review_5, review_7 with grid:
- `k_max ∈ {2, 3, 4}`
- `lam ∈ {0.0, 0.05, 0.1, 0.2}`

Total: 12 configurations × 3 videos = 36 runs.

For each (video, k_max, lam), record:
- Final assignment
- Max consecutive reuse
- Scene diversity (unique scenes / num sentences)
- VisCoher_strict (using existing `compute_strict_viscoher` from `compute_additional_metrics.py`)

Output as a markdown table for analysis. **DO NOT pick the winning config yet — leave that decision to the user after seeing results.**

## FULL EXPERIMENT (ONLY AFTER TESTS PASS AND USER APPROVES K_MAX, LAMBDA)

Once user approves the (k_max, lam) configuration based on subset results:

1. Update `default.yaml` with chosen values.
2. Run full ablation across 10 videos, 2 new arms (`caption_temporal_cvalign`, `siglip_temporal_cvalign`).
3. Use existing pipeline: `python -m src.run_ablation --arms caption_temporal_cvalign siglip_temporal_cvalign --videos all`
4. After completion, run `compute_additional_metrics.py` to add VisCoher_strict and Scene Diversity columns.
5. Run statistical comparison tests:
   - CV-Align vs vanilla DP (paired t-test) per signal track
   - CV-Align vs Greedy per signal track
   - VisCoher_strict, raw VisCoher, Scene Diversity, TempAcc, CLIPScore comparison

## OUTPUT REQUIRED FROM YOU (the agent doing implementation)

After completing all steps, produce:

1. **Code diff** showing all changes to `phase4_retrieve.py` and `default.yaml`.
2. **Test results report** (markdown) showing:
   - Test 1 result (PASS/FAIL with details)
   - Test 2 result
   - Test 3 result with both assignments printed
   - Test 4 hyperparameter sweep table
3. **STOP after producing the sweep table**. Do not proceed to full experiment until user explicitly approves the (k_max, lam) configuration.

## CRITICAL CORRECTNESS REQUIREMENTS

- The reduction property (Test 1) is non-negotiable. If CV-Align with relaxed constraints does not match vanilla DP, your implementation has a bug. Debug it before proceeding.
- The constraint satisfaction property (Test 2) is non-negotiable. The assertion in code MUST hold.
- Do NOT shortcut by reusing vanilla DP code with post-hoc constraints. The whole point is augmented state space — implement it properly.
- Do NOT optimize prematurely. The explicit triple-loop is fine for now. Get correctness first.
- Do NOT modify existing `dp_sequence_align`, `greedy_assign`, or `hungarian_align`. CV-Align is a new sibling method.
- Do NOT modify `compute_additional_metrics.py`, `generate_track_c_reports.py`, `metrics.py`, or `run_ablation.py`. They will work on new arms automatically.

## QUESTIONS YOU SHOULD ASK BEFORE STARTING (if anything is unclear)

- If the recurrence or transition cost formulation is unclear in any case, ask BEFORE implementing.
- If you encounter edge cases not covered (e.g., what if `k_max = 1`?), ask BEFORE making assumptions.
- If a test fails and you cannot diagnose it after 30 minutes, STOP and report the failure with diagnostic output.

## REMINDER ON THESIS CONTEXT

This algorithm is intended to address a specific failure mode (scene-attractor) identified in vanilla DP for the Caption signal track. The intended outcome is:

- Best case: CV-Align significantly improves VisCoher_strict on Caption track while preserving TempAcc.
- Likely case: Modest improvement on VisCoher_strict, with Scene Diversity guaranteed to improve.
- Worst case: Marginal effect (already-good signal SigLIP shows ceiling effect).

In all cases, the contribution is: identifying a failure mode, formalizing a constrained variant, proving correctness, and empirically characterizing where it helps. Honest reporting is mandatory regardless of magnitude of improvement.

END OF PROMPT

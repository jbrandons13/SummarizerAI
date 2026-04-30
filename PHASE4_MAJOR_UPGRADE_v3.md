# TASK: Phase 4 Major Upgrade (v3)

Three independent improvements to Phase 4 retrieval and evaluation:

1. **Multi-frame scene representation** with top-k mean pooling
2. **Three matching algorithms** (Greedy, Hungarian, DP) selectable via config
3. **Temporal alignment + visual coherence** evaluation metrics

Implement and test each part sequentially. Each part is independent.

---

## PART 1: Multi-Frame Scene Representation

### Problem
Current pipeline extracts 1 mid-frame per scene. A 12-second scene showing a
benchmark screenshot for 4 seconds may have a mid-frame that catches the
presenter instead — discarded signal that retrieval can't recover.

### Frame sampling

In `KeyframeExtractor.extract()` in `src/phase4_retrieve.py`:

```python
# Per scene:
duration = end_sec - start_sec

if duration < 1.5:
    # Short scene: 1 frame at midpoint (multi-frame doesn't help)
    frame_timestamps = [(start_sec + end_sec) / 2]
else:
    # Roughly 1 frame per 2s, clamped to [3, 5]
    num_samples = min(5, max(3, int(duration / 2)))
    frame_timestamps = np.linspace(
        start_sec + 0.5, end_sec - 0.5, num_samples
    ).tolist()
```

### Schema updates (`src/schemas.py`)

```python
class KeyframeScene(BaseModel):
    id: int
    start_seconds: float
    end_seconds: float
    keyframe_path: str                       # midpoint frame (backward compat)
    keyframe_timestamp: float                # midpoint timestamp (backward compat)
    multi_frame_paths: List[str] = []        # NEW: paths to all sampled frames
    multi_frame_timestamps: List[float] = [] # NEW: timestamps of all sampled frames


class SceneMatch(BaseModel):
    sentence_id: int
    matched_scene_id: int
    score: float
    best_frame_path: str = ""           # NEW: frame within scene that scored best
    best_frame_timestamp: float = 0.0   # NEW: timestamp of winning frame
    alternatives: List[AlternativeMatch] = []
```

### Scoring: top-k mean (k=2)

Pure max-pool is sensitive to outlier frames (a single frame matching for
spurious reasons — lighting, composition). Top-k mean is more robust:

```python
# For each (sentence, scene), score against all frames in the scene
frame_scores = [cosine_sim(text_emb, fe) for fe in scene_frame_embeddings]

# Top-k mean. With k=2 and 1-frame scenes (short scenes), this gracefully
# degrades to max-pool over a single element.
k = min(2, len(frame_scores))
top_k = sorted(frame_scores, reverse=True)[:k]
score = sum(top_k) / len(top_k)

# Also record which frame won — needed for best_frame_path propagation
best_frame_idx = int(np.argmax(frame_scores))
```

### Propagate winning frame downstream

`best_frame_path` and `best_frame_timestamp` must be populated regardless of
which matching algorithm runs (greedy, Hungarian, or DP). Phase 5 uses these
to center the clip crop instead of the scene midpoint.

To make this clean, build a helper at retrieval time:

```python
# After computing per-scene scores, store per (sentence, scene) the winning frame
# best_frames[(sent_idx, scene_idx)] = (frame_path, frame_timestamp)
best_frames: Dict[Tuple[int, int], Tuple[str, float]] = {}
```

When converting an assignment to `SceneMatch` objects (any algorithm), look up
`best_frames[(sent_idx, scene_idx)]` to fill `best_frame_path` and
`best_frame_timestamp`. Centralizing this avoids per-algorithm divergence.

### Captioning arm (Arm B): cap at 3 frames per scene

VLM captioning is expensive (~1s/frame on Qwen2.5-VL-3B). 5 frames/scene = 5x
slowdown. For Arm B specifically, cap at 3 frames. SigLIP image encoding is
cheap enough to keep at 5.

```yaml
keyframe_extraction:
  frames_per_scene_siglip: 5
  frames_per_scene_caption: 3
  min_frames_short_scene: 1   # scenes < 1.5s
  similarity_pooling: "top_k_mean"
  top_k: 2
```

### Caching

Cache embeddings to disk. **The cache key MUST include the model name** —
SigLIP embeddings and Qwen-VL caption embeddings live in different vector
spaces and cannot be mixed:

```python
import joblib
from pathlib import Path

# Per-model cache file
model_slug = model_name.replace("/", "_").replace("-", "_")
cache_path = output_dir / f"embeddings_{model_slug}.joblib"

# Structure: Dict[Tuple[scene_id: int, frame_timestamp: float], np.ndarray]
# joblib handles this natively. np.savez does NOT preserve dict keys cleanly —
# don't use it for this.

if cache_path.exists():
    embeddings = joblib.load(cache_path)
else:
    embeddings = compute_all_embeddings(...)
    joblib.dump(embeddings, cache_path)
```

---

## PART 2: Matching Algorithms — Greedy, Hungarian, DP

Three matching algorithms, selectable via config. Comparison story:

- **Greedy** — current baseline, per-sentence argmax
- **Hungarian** — global optimal assignment, no sequential structure
- **DP / Viterbi** — global optimal **with** sequential structure ← contribution

The Hungarian baseline isolates the contribution of *sequential* structure
specifically (vs. globalness in general). Without it, the ablation can't
separate those two effects.

### 2A: Greedy (already implemented)

Keep as-is. Used as baseline in both ablation tables.

### 2B: Hungarian

```python
from scipy.optimize import linear_sum_assignment

def hungarian_align(self, sim_matrix: np.ndarray, reuse_penalty: float = 0.2):
    """
    Global optimal assignment via Hungarian algorithm.
    Scenes can be reused, but each reuse incurs an additive penalty so fresh
    scenes are preferred when similarity is comparable.

    Args:
        sim_matrix: (N, M) — higher = better match
        reuse_penalty: cost added per additional copy (0 = no preference for fresh)

    Returns:
        list of length N — scene index assigned to each sentence
    """
    N, M = sim_matrix.shape

    # Tile columns to allow reuse. K copies of each scene; the k-th copy of
    # scene j sits at column k*M + j in the cost matrix.
    K = max(3, (N // M) + 1)
    cost_matrix = np.tile(-sim_matrix, (1, K))  # negate: Hungarian minimizes

    # Penalty grows with copy index — copy 0 is free, copy 1 costs reuse_penalty,
    # copy 2 costs 2*reuse_penalty, etc. This makes scene reuse a soft choice.
    for k in range(K):
        cost_matrix[:, k * M:(k + 1) * M] += k * reuse_penalty

    # M*K >= N always with this K formula, so cost_matrix is a valid input.
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    # Map duplicated columns back to original scene indices
    assignment = [int(col_idx[i] % M) for i in range(N)]
    return assignment
```

### 2C: DP / Viterbi Sequence Alignment (main contribution)

Linear assignment can only express unary costs (cost depends on i, j alone).
Sequential coherence is a pairwise/transition cost. DP/Viterbi is the right
algorithm.

```python
def dp_sequence_align(
    self,
    sim_matrix: np.ndarray,
    scenes: List[KeyframeScene],
    video_duration: float,
    jump_penalty: float = 0.3,
    reuse_bonus: float = 0.3,
    backward_penalty: float = 0.5,
):
    """
    Viterbi-style DP for sentence-to-scene assignment with transition costs.

    Maximizes:  sum_i sim[i, a_i] - sum_i transition(a_{i-1}, a_i)

    Where transition cost penalizes large temporal jumps and going backward,
    and rewards staying on the same scene.

    Time:  O(N * M^2)  — trivial for N~50, M~200
    Space: O(N * M)

    NOTE: sim_matrix is assumed to already include the unary temporal prior
    (Gaussian around source_timestamp_hint). DP adds *transition* structure
    on top of that — the two are complementary, not redundant.

    Args:
        sim_matrix: (N, M) similarity scores
        scenes: list of KeyframeScene
        video_duration: total video length (seconds), used to normalize jumps
        jump_penalty: scales with normalized temporal distance for forward jumps
        reuse_bonus: bonus (subtracted from cost) for staying on the same scene
        backward_penalty: extra cost for going backward in time

    Returns:
        list of length N — scene index per sentence
    """
    N, M = sim_matrix.shape

    # Defensive: replace nan/inf so argmax doesn't return garbage
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=1.0, neginf=-1e9)

    # Edge case: single sentence — no transitions, pure argmax
    if N == 1:
        return [int(np.argmax(sim_matrix[0]))]

    scene_time = np.array([s.keyframe_timestamp for s in scenes])

    # Normalized time difference: dt[k, j] = (t_j - t_k) / duration in [-1, 1]
    dt_matrix = (scene_time[None, :] - scene_time[:, None]) / max(video_duration, 1e-6)

    # Transition cost from scene k -> j
    #   forward (dt >= 0):  jump_penalty * dt
    #   backward (dt < 0):  jump_penalty * |dt| + backward_penalty
    transition_matrix = np.where(
        dt_matrix >= 0,
        jump_penalty * dt_matrix,
        jump_penalty * np.abs(dt_matrix) + backward_penalty,
    )
    # Same-scene transition is a bonus (negative cost), overriding the formula
    np.fill_diagonal(transition_matrix, -reuse_bonus)

    # DP tables
    dp = np.full((N, M), -np.inf)
    backptr = np.full((N, M), -1, dtype=int)

    # Base case: no incoming transition
    dp[0] = sim_matrix[0]

    # Forward pass — vectorized over previous-state dimension
    for i in range(1, N):
        # candidates[k, j] = dp[i-1, k] - transition_matrix[k, j]
        candidates = dp[i - 1][:, None] - transition_matrix  # (M, M)
        dp[i] = sim_matrix[i] + candidates.max(axis=0)
        backptr[i] = candidates.argmax(axis=0)

    # Backtrack
    assignment = [0] * N
    assignment[N - 1] = int(np.argmax(dp[N - 1]))
    for i in range(N - 2, -1, -1):
        assignment[i] = int(backptr[i + 1][assignment[i + 1]])

    return assignment
```

### Calibration note on `jump_penalty`

Cosine similarities live in roughly `[0, 1]` after normalization.
`jump_penalty=1.0` means a full-video jump costs ~1.0 — comparable to the
entire similarity range, which is too aggressive. Default to 0.3 and tune on
1–2 videos before running the full ablation. If DP looks identical to greedy,
penalty is too low; if DP "sticks" on one scene, penalty is too high.

### Integration into retrieval arms

All three algorithms consume the same `sim_matrix`. `best_frame` lookup is
centralized so all algorithms get the same downstream propagation:

```python
# In SigLIP2DirectRetrieval.retrieve() and CaptionCosineRetrieval.retrieve():

# 1. Build sim_matrix (semantic similarity + unary temporal prior, top-k pooled
#    over multi-frame).
# 2. Build best_frames lookup: best_frames[(sent_idx, scene_idx)] = (path, ts)
# 3. Choose matching algorithm.

matching_algo = self.config.get("retrieval", {}).get("matching_algorithm", "dp")

if matching_algo == "greedy":
    assignment = self.greedy_assign(sim_matrix)  # returns list of scene indices
elif matching_algo == "hungarian":
    assignment = self.hungarian_align(sim_matrix)
elif matching_algo == "dp":
    video_dur = max(s.end_seconds for s in manifest.scenes)
    assignment = self.dp_sequence_align(sim_matrix, manifest.scenes, video_dur)
else:
    raise ValueError(f"Unknown matching_algorithm: {matching_algo}")

# Convert assignment -> SceneMatch (shared across all three algorithms)
matches = []
for sent_idx, scene_idx in enumerate(assignment):
    best_path, best_ts = best_frames.get((sent_idx, scene_idx), ("", 0.0))
    top_indices = np.argsort(-sim_matrix[sent_idx])[:5]
    alternatives = [
        AlternativeMatch(scene_id=int(idx), score=float(sim_matrix[sent_idx, idx]))
        for idx in top_indices
    ]
    matches.append(SceneMatch(
        sentence_id=int(sent_idx),
        matched_scene_id=int(scene_idx),
        score=float(sim_matrix[sent_idx, scene_idx]),
        best_frame_path=best_path,
        best_frame_timestamp=best_ts,
        alternatives=alternatives,
    ))
```

If `greedy_match` currently returns `SceneMatch` objects directly, refactor
it to return an assignment list (`List[int]`) so the conversion above is
shared. This is the single change that prevents algorithm-specific divergence
on `best_frame_path` propagation.

### Config

```yaml
retrieval:
  matching_algorithm: "dp"  # "greedy" | "hungarian" | "dp"

  # DP parameters
  dp_jump_penalty: 0.3       # tune on 1-2 videos before full ablation
  dp_reuse_bonus: 0.3
  dp_backward_penalty: 0.5

  # Hungarian parameters
  hungarian_reuse_penalty: 0.2
```

### Arm A (Random) stays truly random

Do NOT apply Hungarian or DP to random scores. Random baseline = random
per-sentence picks. "Hungarian on random" is a different statistical object
(optimal assignment under random scores has higher mean and lower variance
than truly random matching) — if you want it as a config, name it separately.

---

## PART 3: Evaluation Metrics

### 3A: Temporal Alignment Score (window-based)

```python
import numpy as np

def temporal_alignment_score(matches, summary, manifest, thresholds=(5, 15, 30, 60)):
    """
    Measures how close retrieved scenes are to source content location.

    Window-based error: if retrieved timestamp is inside source_timestamp_hint
    range, error = 0. Otherwise, distance to nearest edge of the range.

    Reports accuracy at multiple thresholds because a single threshold hides
    qualitative differences (40% within 5s vs. 40% within 30s are very different).
    """
    errors = []
    video_duration = max(s.end_seconds for s in manifest.scenes)
    within_counts = {t: 0 for t in thresholds}

    for match in matches:
        sentence = summary.sentences[match.sentence_id]
        scene = next(s for s in manifest.scenes if s.id == match.matched_scene_id)

        hint = sentence.source_timestamp_hint
        if not hint or len(hint) < 2:
            continue

        # Use the matched frame's timestamp if available, else scene midpoint
        retrieved_ts = match.best_frame_timestamp or scene.keyframe_timestamp

        if hint[0] <= retrieved_ts <= hint[1]:
            error = 0.0
        else:
            error = min(abs(retrieved_ts - hint[0]), abs(retrieved_ts - hint[1]))

        errors.append(error)
        for t in thresholds:
            if error <= t:
                within_counts[t] += 1

    if not errors:
        return {"mean_temporal_error": -1, "n_evaluated": 0}

    result = {
        "n_evaluated": len(errors),
        "mean_temporal_error_seconds": float(np.mean(errors)),
        "median_temporal_error_seconds": float(np.median(errors)),
        "normalized_temporal_error": float(np.mean(errors) / video_duration),
    }
    for t in thresholds:
        result[f"temporal_accuracy_within_{t}s"] = within_counts[t] / len(errors)
    return result
```

### 3B: Visual Coherence Score

**Important:** this metric must use the embedding of the *specific frame
that won*, not a scene-level embedding. Otherwise multi-frame retrieval and
single-frame retrieval are evaluated against different things.

```python
def visual_coherence_score(matches, frame_embeddings):
    """
    Average cosine similarity between consecutive matched FRAMES.
    Higher = smoother visual transitions, less jumping.

    Args:
        matches: list of SceneMatch (in sentence order)
        frame_embeddings: Dict[Tuple[scene_id: int, frame_timestamp: float], np.ndarray]
                          Same cache structure used during retrieval.

    Returns:
        mean and std of consecutive cosine similarities.
    """
    consecutive_sims = []

    for i in range(len(matches) - 1):
        key_a = (matches[i].matched_scene_id, matches[i].best_frame_timestamp)
        key_b = (matches[i + 1].matched_scene_id, matches[i + 1].best_frame_timestamp)

        emb_a = frame_embeddings.get(key_a)
        emb_b = frame_embeddings.get(key_b)
        if emb_a is None or emb_b is None:
            # Fall back to any frame from that scene if exact key missing
            # (shouldn't happen if caching is consistent, but be defensive)
            continue

        norm_a, norm_b = np.linalg.norm(emb_a), np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            continue
        consecutive_sims.append(float(np.dot(emb_a, emb_b) / (norm_a * norm_b)))

    if not consecutive_sims:
        return {"visual_coherence_mean": 0.0, "visual_coherence_std": 0.0, "n_pairs": 0}

    return {
        "visual_coherence_mean": float(np.mean(consecutive_sims)),
        "visual_coherence_std": float(np.std(consecutive_sims)),
        "n_pairs": len(consecutive_sims),
    }
```

### 3C: Add to ablation runner

In `src/eval/run_ablation.py`, alongside CLIPScore / ROUGE / BERTScore:

```python
temporal = temporal_alignment_score(matches, summary, manifest)
coherence = visual_coherence_score(matches, frame_embeddings_cache)

results[arm_name].update({
    "temporal_mean_error_s": temporal.get("mean_temporal_error_seconds"),
    "temporal_acc_5s":  temporal.get("temporal_accuracy_within_5s"),
    "temporal_acc_15s": temporal.get("temporal_accuracy_within_15s"),
    "temporal_acc_30s": temporal.get("temporal_accuracy_within_30s"),
    "temporal_acc_60s": temporal.get("temporal_accuracy_within_60s"),
    "visual_coherence_mean": coherence["visual_coherence_mean"],
})
```

---

## PART 4: Ablation Structure

Six configs total, presented as two tables. **Frame Table 2 as
"ablating the matching algorithm holding signal fixed"** — this makes the
contribution structure clear in the writeup and pre-empts the "is the gain
from globalness or sequentialness?" reviewer question.

**Table 1: Effect of retrieval signal** (all use Greedy matching)

| Arm | Retrieval Signal           | Matching |
|-----|----------------------------|----------|
| A   | Random                     | Greedy   |
| B   | Caption cosine             | Greedy   |
| C   | SigLIP direct              | Greedy   |
| C+T | SigLIP + temporal prior    | Greedy   |

**Table 2: Effect of matching algorithm** (all use SigLIP + temporal prior)

| Config         | Retrieval Signal        | Matching   |
|----------------|-------------------------|------------|
| C+T+Greedy     | SigLIP + temporal       | Greedy     |
| C+T+Hungarian  | SigLIP + temporal       | Hungarian  |
| C+T+DP         | SigLIP + temporal       | DP         |

`C+T+Greedy` appears in both tables (it's the bridge). 6 unique configs.

**All three matching algorithms must consume the identical `sim_matrix`** to
ensure a fair comparison. The sim_matrix already contains the unary temporal
prior; matching algorithms only differ in how they consume it.

---

## Sanity Checks Before Full Run (must pass)

### Test 1: DP vs greedy behavior
3 sentences, 4 scenes at times `[0, 10, 20, 30]`. Construct a sim_matrix
where greedy would jump (e.g., `[0, 3, 1]`), then verify DP prefers a more
monotonic path. Crank `jump_penalty` from 0 → 10; if assignment never
changes, the transition cost isn't being applied.

### Test 2: Optimality assertion (CRITICAL)
DP must produce a path score ≥ greedy under the same cost function. This is
a *correctness invariant* — if it fails, there's a bug in the recurrence or
backtrack:

```python
def compute_path_score(sim_matrix, assignment, transition_matrix):
    s = sim_matrix[0, assignment[0]]
    for i in range(1, len(assignment)):
        s += sim_matrix[i, assignment[i]]
        s -= transition_matrix[assignment[i - 1], assignment[i]]
    return s

dp_score     = compute_path_score(sim_matrix, dp_assignment,     transition_matrix)
greedy_score = compute_path_score(sim_matrix, greedy_assignment, transition_matrix)
assert dp_score >= greedy_score - 1e-6, \
    f"DP ({dp_score:.4f}) worse than greedy ({greedy_score:.4f}) — recurrence bug"
```

### Test 3: Degenerate transitions reduce DP to argmax
With `jump_penalty=0`, `reuse_bonus=0`, `backward_penalty=0`, DP must produce
the same assignment as per-sentence argmax. If not, transitions aren't being
zeroed properly somewhere.

### Test 4: Phase 5 boundary clamp
When `best_frame_timestamp` is near a scene boundary, the clip extraction
window can spill past the scene cut. Clamp:

```python
clip_start = max(best_frame_timestamp - clip_len / 2, scene.start_seconds)
clip_end   = min(clip_start + clip_len, scene.end_seconds)
```

### Test 5 (recommended): cache key collision
Run Arm B (Qwen-VL captions) and Arm C (SigLIP) back-to-back on the same
video. Both should produce different cache files (different `model_slug`).
If cache files collide, embeddings will be silently wrong on second run.

---

## Dependencies
```bash
pip install scipy joblib  # linear_sum_assignment + cache I/O
```

## Files to Change

| File | Change |
|------|--------|
| `src/schemas.py` | `multi_frame_paths/timestamps` on `KeyframeScene`; `best_frame_path/timestamp` on `SceneMatch` |
| `src/phase4_retrieve.py` | Multi-frame extraction; top-k mean pooling; `best_frames` lookup; `hungarian_align()`; `dp_sequence_align()`; refactor `greedy_match` to return assignment list |
| `src/phase5_assemble.py` | Use `best_frame_timestamp` for clip centering with scene-boundary clamp |
| `src/eval/metrics.py` | `temporal_alignment_score()`, `visual_coherence_score()` |
| `src/eval/run_ablation.py` | Wire new metrics into per-arm results; support 6 configs |
| `configs/default.yaml` | All new config parameters |

## Do NOT Change
- Phase 1, 2, 3 — unaffected
- Arm A random — must stay truly random per-sentence picks
- Existing temporal guidance code (`compute_temporal_scores`,
  `min_max_normalize`) — still used to build `sim_matrix`

---

## Implementation order (suggested)

1. **Schema changes** — add fields to `KeyframeScene` and `SceneMatch`
2. **Multi-frame extraction + top-k pooling** — verify with one video, watch the output
3. **Refactor `greedy_match` → assignment list** — keeps later integration clean
4. **`hungarian_align`** — easy, validates the `sim_matrix` contract
5. **`dp_sequence_align`** — implement, then run all 5 sanity checks
6. **Phase 5 `best_frame_timestamp` + clamp** — small change but unblocks visible improvement
7. **Eval metrics** — add and verify on one arm
8. **Full 6-config ablation** — only after the above is green

Steps 1–6 should take ~1 day. Step 8 is overnight on 8–10 videos with caching.

# Patch & Restart: Bug #2 Fix (Gating on Raw Cosine)

**Context:** Bug #2 confirmed (minmax+gating arms = 100% retrieve because score=1.0000 after normalization). Architect (Claude) decided Opsi B: decouple normalization (for ranking) from gating decision (use raw weighted cosine always).

**Goal:** Patch the script, re-run 8 affected arms only, regenerate results CSV.

---

## 1. The patch (2-line change in gating logic)

**Location:** `scripts/run_ablation_16_arms.py`, around line 380-395 (the `# Build assignments list` block).

**Find this code (the existing block):**

```python
# Build assignments list
assignments_list = []
for k, group in enumerate(arm_groups):
    scene_idx = assigned_scenes[k]
    score = float(score_matrix[k, scene_idx])
    raw_cos = float(raw_cosine_matrix[k, scene_idx])
    weight = float(temporal_weight_matrix[k, scene_idx])
    
    # Gating action
    if gating:
        action = "retrieve" if score >= 0.12 else "generate"
    else:
        action = "retrieve"
```

**Replace with:**

```python
# Build assignments list
assignments_list = []
for k, group in enumerate(arm_groups):
    scene_idx = assigned_scenes[k]
    score = float(score_matrix[k, scene_idx])
    raw_cos = float(raw_cosine_matrix[k, scene_idx])
    weight = float(temporal_weight_matrix[k, scene_idx])
    
    # PATCH (Bug #2): Gating decision MUST use raw weighted cosine, NOT normalized score.
    # Min-max normalization per-group destroys the absolute signal needed for gating,
    # because the best-match score always becomes 1.0 after normalization.
    # See: architect review 21 May 2026 + Gemini verification report.
    raw_weighted = raw_cos * weight  # equivalent to sim_matrix[k, scene_idx] before normalization
    
    if gating:
        action = "retrieve" if raw_weighted >= 0.12 else "generate"
    else:
        action = "retrieve"
```

**That's it. No other changes.**

Verify the patch by:
```bash
sed -n '380,400p' scripts/run_ablation_16_arms.py
```

Expected: comment block "PATCH (Bug #2)" visible, `raw_weighted` variable assigned, threshold compares against `raw_weighted` not `score`.

---

## 2. Verification BEFORE restart

Run ONE arm on ONE video to verify patch works correctly:

```bash
# Pick a minmax+gating arm for quick verification
python scripts/run_ablation_16_arms.py --video review_1 2>&1 | tee /tmp/patch_verify.log

# Check assignments — minmax+gating arms should NOW have meaningful retrieve/generate split
for arm in minmax_hybrid_retrieval_siglip_gating minmax_hybrid_retrieval_ccma_gating; do
    file="data/intermediate/review_1/scene_matches_${arm}.json"
    if [ -f "$file" ]; then
        echo "=== $arm ==="
        python -c "
import json
with open('$file') as f:
    data = json.load(f)
groups = data['groups']
retrieve_count = sum(1 for g in groups if g['action'] == 'retrieve')
generate_count = sum(1 for g in groups if g['action'] == 'generate')
print(f'  Retrieve: {retrieve_count}, Generate: {generate_count}')
print(f'  Best similarity values: {[round(g[\"best_similarity\"], 4) for g in groups]}')
print(f'  Raw cosine values: {[round(g[\"raw_cosine\"], 4) for g in groups]}')
"
    fi
done
```

**Expected output (patch successful):**
- `best_similarity` field still shows normalized scores (e.g., 1.0000) — this is for reporting only, not gating
- `Retrieve: X, Generate: Y` with split similar to raw arms (e.g., 3/2 not 5/0)

**If output still shows 5/0:** patch failed, do NOT proceed to step 3. Stop and report.

---

## 3. Full 16-arm re-run scope

**User decision:** Re-run all 16 arms × 10 videos = 160 runs.

**Rationale for full re-run (not targeted):**
- Cleaner data: all arms produced from same code state, same hardware, same time window
- Reproducibility: a single execution log captures the entire ablation
- Eliminates any concern about code-state drift between original run and patch run
- The 12 unaffected arms will produce identical results to before (DP/CCMA are deterministic with SEED=42), so re-running them is purely a consistency measure

**Procedure:**

```bash
# Clear ALL previous ablation outputs (assignments + per-arm eval CSVs)
# This ensures no stale results contaminate the new run

# 1. Backup current results CSV (for comparison later)
cp results/final_ablation_results.csv results/final_ablation_results_BUG2_BROKEN.csv 2>/dev/null || echo "No previous CSV to backup"

# 2. Delete per-arm scene_matches files (will be regenerated)
find data/intermediate -name "scene_matches_raw_*.json" -delete
find data/intermediate -name "scene_matches_minmax_*.json" -delete

# 3. Delete per-arm summary videos (will be regenerated)
find data/output -name "summary_raw_*.mp4" -delete
find data/output -name "summary_minmax_*.mp4" -delete

# 4. Delete per-arm eval CSVs (will be regenerated)
find data/evaluation -name "unified_eval_raw_*.csv" -delete
find data/evaluation -name "unified_eval_minmax_*.csv" -delete

# 5. Verify cleanup
find data/intermediate -name "scene_matches_*.json" | wc -l  # expect 0
find data/output -name "summary_raw_*.mp4" -o -name "summary_minmax_*.mp4" | wc -l  # expect 0
find data/evaluation -name "unified_eval_*.csv" | wc -l  # expect 0

# 6. Run full 16-arm sweep across all 10 videos
python scripts/run_ablation_16_arms.py --video all 2>&1 | tee /tmp/full_16arm_rerun.log
```

**Expected wallclock:** ~3-4 hours (based on previous run duration).

**During run, monitor:**
- VRAM usage stable (no leaks across subprocesses)
- Per-arm assignment files appearing in `data/intermediate/{video}/scene_matches_{arm}.json`
- Eval CSVs appearing in `data/evaluation/unified_eval_{arm}.csv`

---

## 4. Results CSV — no consolidation needed

Since full sweep regenerates all 160 rows from scratch, the script's own aggregation logic (at end of `run_ablation_sweep()`) handles CSV creation. No manual consolidation script needed.

The orchestrator already writes `results/final_ablation_results.csv` at end of sweep with all 160 rows.

**Verify after sweep completes:**

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/final_ablation_results.csv')
print('Total rows:', len(df))
print('Unique arms:', df['arm'].nunique())
print('Unique videos:', df['video_id'].nunique())
print()
print('Per-arm count:')
print(df['arm'].value_counts().sort_index())
"
```

Expected:
- Total rows: 160
- Unique arms: 16 (each appearing exactly 10 times)
- Unique videos: 10

---

## 5. Final verification

```bash
# Confirm all 16 arms × 10 videos present in final CSV
python -c "
import pandas as pd
df = pd.read_csv('results/final_ablation_results.csv')
print('Total rows:', len(df))
print('Unique arms:', df['arm'].nunique())
print('Unique videos:', df['video_id'].nunique())
print('\nArm counts:')
print(df['arm'].value_counts())
"
```

Expected:
- Total rows: 160
- Unique arms: 16
- Unique videos: 10
- All arms have count = 10

```bash
# Sanity check: minmax+gating arms should now have varied retrieve/generate counts
for arm in minmax_hybrid_retrieval_siglip_gating minmax_hybrid_retrieval_ccma_gating; do
    echo "=== $arm ==="
    for vid in review_1 review_5 review_10; do
        file="data/intermediate/$vid/scene_matches_${arm}.json"
        python -c "
import json
with open('$file') as f:
    data = json.load(f)
groups = data['groups']
r = sum(1 for g in groups if g['action'] == 'retrieve')
g = sum(1 for x in groups if x['action'] == 'generate')
print(f'  $vid: retrieve={r}, generate={g}')
"
    done
done
```

Expected: varied splits (not all 5/0). Should approximately match raw_*_gating counterparts.

---

## 6. Handoff format

```
## Bug #2 Patch + Full 16-Arm Re-run Complete

**Patch applied:** YES — line <X> in scripts/run_ablation_16_arms.py
**Verification on review_1:** PASS — minmax+gating split now <X>/<Y> instead of 5/0

**Cleanup before re-run:**
- scene_matches deleted: <count>
- summary videos deleted: <count>
- eval CSVs deleted: <count>

**Re-run scope:** 16 arms × 10 videos = 160 runs
**Wallclock:** <duration>
**Successes:** X/160
**Failures:** <list with arm × video pairs, or "None">

**Final verification:**
- Total rows in CSV: <count>
- Unique arms: 16 / <count>
- Unique videos: 10 / <count>
- Sample minmax+gating arm split (review_1, review_5, review_10): <verbatim>
- Sample raw+gating arm split (control, should match approximately): <verbatim>

**Anomalies:** <list or None>
```

---

## Anti-hallucination reminders

1. **Verify patch on 1 video BEFORE re-running all 10.** If patch doesn't work on review_1, fixing the same way for 10 videos = wasted compute.

2. **DO NOT modify other parts of the script.** The patch is exactly 1 line change (the threshold comparison target) plus comment. Don't refactor variable names or "improve" anything else.

3. **Full 16-arm re-run is intentional, not a bug.** User explicitly chose full re-run for cleaner data + reproducibility. Do not suggest targeted re-run as "optimization."

4. **No CSV consolidation needed.** Full sweep generates fresh CSV from scratch. Don't write consolidation scripts.

5. **`best_similarity` field in JSON output still shows normalized score (1.0) for minmax arms.** This is BY DESIGN now — `best_similarity` is for reporting, gating decision uses `raw_cosine * temporal_weight` internally. Do not "fix" this.

6. **If new run produces some failures (e.g., LTX OOM on 1 video), do NOT silently skip.** Log + report explicitly in handoff.

End of brief.

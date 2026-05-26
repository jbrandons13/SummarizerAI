# Bug Check & Design Review: 16-Arm Ablation Code

**Context:** Senior architect (Claude) reviewed `scripts/run_ablation_16_arms.py` and found 1 potential serious bug + several design concerns. Verify findings against actual code behavior. Report verbatim — do NOT fix anything yet.

---

## Bug to verify

### Bug #2: Hardcoded gating threshold 0.12 for both raw and normalized scores

**Location:** Around line 384-385 of the ablation script:
```python
if gating:
    action = "retrieve" if score >= 0.12 else "generate"
```

**Suspected issue:** When `normalize=True`, the `score` value comes from `score_matrix` after `min_max_normalize()` per-group. This normalizes scores to [0, 1] range per group. Threshold 0.12 (calibrated for raw SigLIP cosine, range 0.07-0.16) becomes too low in [0, 1] range — likely lets nearly all groups pass to "retrieve" action, defeating the purpose of gating in normalized arms.

**Verification steps:**

1. **Read the actual gating code** (paste verbatim with line numbers):
   ```bash
   grep -n "0.12\|retrieve\|generate" scripts/run_ablation_16_arms.py | head -20
   sed -n '380,400p' scripts/run_ablation_16_arms.py
   ```

2. **Check actual score distribution per arm in saved assignments:**
   ```bash
   # For 4 normalized + gating arms, count retrieve vs generate
   for arm in minmax_hybrid_retrieval_siglip_gating minmax_hybrid_retrieval_ccma_gating minmax_hybrid_retrieval_siglip_grouping_gating minmax_hybrid_retrieval_ccma_grouping_gating; do
       echo "=== ARM: $arm ==="
       for vid in review_1 review_5 review_10; do
           file="data/intermediate/$vid/scene_matches_${arm}.json"
           if [ -f "$file" ]; then
               echo "Video: $vid"
               python -c "
import json
with open('$file') as f:
    data = json.load(f)
groups = data['groups']
retrieve_count = sum(1 for g in groups if g['action'] == 'retrieve')
generate_count = sum(1 for g in groups if g['action'] == 'generate')
scores = [g['best_similarity'] for g in groups]
print(f'  Total groups: {len(groups)}, Retrieve: {retrieve_count}, Generate: {generate_count}')
print(f'  Score range: [{min(scores):.4f}, {max(scores):.4f}], median: {sorted(scores)[len(scores)//2]:.4f}')
"
           else
               echo "Video: $vid - FILE NOT FOUND"
           fi
       done
   done
   ```

3. **Compare with raw + gating arms** (these should have meaningful retrieve/generate split):
   ```bash
   for arm in raw_hybrid_retrieval_siglip_gating raw_hybrid_retrieval_ccma_gating raw_hybrid_retrieval_siglip_grouping_gating raw_hybrid_retrieval_ccma_grouping_gating; do
       echo "=== ARM: $arm ==="
       for vid in review_1 review_5 review_10; do
           file="data/intermediate/$vid/scene_matches_${arm}.json"
           if [ -f "$file" ]; then
               echo "Video: $vid"
               python -c "
import json
with open('$file') as f:
    data = json.load(f)
groups = data['groups']
retrieve_count = sum(1 for g in groups if g['action'] == 'retrieve')
generate_count = sum(1 for g in groups if g['action'] == 'generate')
scores = [g['best_similarity'] for g in groups]
print(f'  Total groups: {len(groups)}, Retrieve: {retrieve_count}, Generate: {generate_count}')
print(f'  Score range: [{min(scores):.4f}, {max(scores):.4f}], median: {sorted(scores)[len(scores)//2]:.4f}')
"
           fi
       done
   done
   ```

## Other design concerns to verify

### Concern #1: Greedy grouping uses raw cosine even for "normalize" arms

**Suspected issue:** `greedy_grouping()` is called ONCE before the 16-arm loop (around line 246). The grouping uses `weighted = raw_cosine * temporal_prior` to lock onto scenes. This means arms with `normalize=True` AND `grouping=True` use raw-cosine-based grouping topology, even though scores are normalized later.

**Verify:**
```bash
sed -n '240,260p' scripts/run_ablation_16_arms.py
# Confirm: is greedy_grouping called inside or outside the arm loop?
```

**Question:** Is this by design (grouping is a pre-processing step, same topology for all arms) or unintended? Either is OK, but needs documentation.

### Concern #2: "argmax vs CCMA" dimension naming

**Suspected mismatch:** ARM_CONFIGS uses `"dp"` and `"ccma"` as matching algorithm values, not `"argmax"` and `"ccma"`. So the actual ablation dimension is **DP (vanilla monotonic alignment) vs CCMA (capacity-constrained monotonic alignment)**, not "argmax vs CCMA."

**Verify:**
```bash
grep -n "matching_algo\|matching_algorithm" scripts/run_ablation_16_arms.py | head -20
```

**Question:** Confirm there is NO "argmax" matching algorithm option (greedy first-best without monotonic alignment).

### Concern #3: Phase5Assembler re-instantiation overhead

**Suspected issue:** Line ~433, `p5 = Phase5Assembler(config, vram_manager)` is instantiated 16 × 10 = 160 times. If LTX-Video model loads fresh each time, this adds significant overhead.

**Verify:**
1. Open `src/phase5_assemble.py`
2. Check if `Phase5Assembler.__init__` loads any heavy models eagerly
3. Check if there's model caching across instantiations (e.g., via VRAMManager)

**Report:**
- Does LTX model load on `__init__` or only on first `run()` call?
- Is the model cached/reused across instances?
- Estimate overhead per arm (in seconds) if model loads fresh each time

### Concern #4: Subprocess evaluation cost

**Suspected issue:** Line ~444, evaluation is launched as subprocess for each arm. This means `src.eval.unified_evaluation` loads CLIP + BLIP + judge models fresh per arm.

**Verify:**
```bash
cat src/eval/unified_evaluation.py 2>/dev/null | head -80
```

**Report:** How many models does `unified_evaluation` load? Is there any model caching across runs?

---

## Report format

```
## Bug #2 Verification: Threshold 0.12 Universal

**Code excerpt (lines 380-400):**
<paste verbatim>

**Normalize+gating arms — retrieve/generate split:**
<paste the python output for 4 minmax+gating arms across 3 videos>

**Raw+gating arms — retrieve/generate split (control):**
<paste for 4 raw+gating arms across 3 videos>

**Verdict:** 
- Is bug confirmed? YES / NO / PARTIAL
- Evidence: <bullet points comparing minmax vs raw split>
- Estimated impact: <how many arms × videos affected>

---

## Concern #1: Grouping topology

**greedy_grouping call location:**
<paste lines 240-260>

**Verdict:** Is grouping called inside or outside arm loop?
**Question to architect:** Is this by design or bug? <YOUR opinion based on code structure>

---

## Concern #2: Algorithm dimension naming

**Available matching_algo values:** <list from code>
**Confirms:** "argmax" option <EXISTS / DOES NOT EXIST>
**Implication:** Ablation dimension is actually "DP vs CCMA" not "argmax vs CCMA"

---

## Concern #3: Phase5Assembler overhead

**Model load location in `__init__`:** <YES / NO, paste relevant code>
**Model caching:** <YES / NO>
**Estimated overhead per arm if no caching:** <seconds>

---

## Concern #4: Subprocess evaluation overhead

**Models loaded in unified_evaluation:** <list>
**Caching:** <YES / NO>
**Estimated cost per arm:** <seconds>

---

## Recommendation

Based on findings, what should be done:
- Bug #2: PATCH NOW / RUN FIRST, PATCH AFTER / DOCUMENT AS LIMITATION
- Concern #1, #2, #3, #4: respective recommendations

DO NOT IMPLEMENT FIXES. Report only.
```

End of prompt.

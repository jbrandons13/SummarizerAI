# Brief: Fix Timestamp Ordering Bug in RetrievalGate

**Task type:** Execution (bug fix + verification).
**Goal:** Fix two related bugs di `src/phase4_retrieve.py` yang assume sentences dalam satu group temporally ordered di source video — itu invalid karena LLM summary bisa reorder sentences. Re-run pipeline review_1 untuk verify fix.

## Context

Inspection `p4_assignments.json` review_1 menunjukkan group 6 (sentence_ids [10,11,12,13]) punya `timestamp_hint_merged: [436.555, 427.167]` dengan **end < start** (inverted).

Root cause: LLM summary di Phase 2 generate sentences dalam narrative/topical order, bukan source video temporal order. Sehingga `sentences[group_ids[0]].timestamp_hint[0]` bukan necessarily minimum, dan `sentences[group_ids[-1]].timestamp_hint[1]` bukan necessarily maximum dalam satu group.

## Bugs to fix

### Bug 1: `_hint_center` (line 1245-1250)

Current:
```python
def _hint_center(self, sentences: Sequence[Sentence], ids: Sequence[int]) -> float:
    """Centre of the merged timestamp hint across the group."""
    lo = sentences[ids[0]].timestamp_hint[0]
    hi = sentences[ids[-1]].timestamp_hint[1]
    return (float(lo) + float(hi)) / 2.0
```

Fix:
```python
def _hint_center(self, sentences: Sequence[Sentence], ids: Sequence[int]) -> float:
    """Centre of the merged timestamp hint across the group.
    
    Sentences within a group may not be temporally ordered in the source video
    (LLM summary reorders by narrative/topic), so we take min/max across all
    sentences in the group rather than assuming first/last bound the range.
    """
    starts = [sentences[sid].timestamp_hint[0] for sid in ids]
    ends = [sentences[sid].timestamp_hint[1] for sid in ids]
    lo = min(starts)
    hi = max(ends)
    return (float(lo) + float(hi)) / 2.0
```

### Bug 2: `timestamp_hint_merged` construction in `_build_group` (line 1336-1337)

Current:
```python
hint_start = sentences[group_ids[0]].timestamp_hint[0]
hint_end = sentences[group_ids[-1]].timestamp_hint[1]
```

Fix:
```python
# Sentences within a group may not be temporally ordered (LLM reorders).
# Take min/max across all sentences to get true bounding range.
all_starts = [sentences[sid].timestamp_hint[0] for sid in group_ids]
all_ends = [sentences[sid].timestamp_hint[1] for sid in group_ids]
hint_start = min(all_starts)
hint_end = max(all_ends)
```

## Verification

### Step 1: Apply fixes

Apply both patches above ke `src/phase4_retrieve.py`. Show diff before commit.

### Step 2: Run pipeline review_1

Run pipeline end-to-end on `data/eval_videos/review_1.mp4`. Same command as Stage 2 verification.

### Step 3: Inspect new `p4_assignments.json`

For each group, check: `timestamp_hint_merged[0] <= timestamp_hint_merged[1]`.

**Expected:** No inverted ranges. All groups should have valid temporal bounds.

### Step 4: Compare with old output

Bandingkan new `p4_assignments.json` review_1 vs old (sebelum fix):

- Total groups: should be same (grouping algorithm tidak berubah, cuma timestamp output)
- sentence_ids per group: should be same
- action per group: should be same  
- best_similarity per group: **could differ slightly** karena `_hint_center` fix affect temporal weight calculation. Differences <0.005 acceptable; larger differences worth investigating.
- timestamp_hint_merged: SHOULD differ for groups dengan reordered sentences (yang sebelumnya inverted)

### Step 5: Verify downstream

- Run Phase 5 assembler dengan output baru
- Final mp4 should still generate successfully
- `total_duration_seconds` should be reasonable (close to old value)

## What to report back

Markdown report:

```
## Patch applied
- Diff shown to user: yes/no
- Files modified: <list>

## Verification on review_1
- Total groups (new vs old): X vs X
- Inverted ranges in new output: 0 (or list any)
- best_similarity differences: <max delta, average delta>
- Groups where timestamp_hint_merged changed: <count>
- Phase 5 assembler ran successfully: yes/no
- Final mp4 duration (new vs old): X vs X

## Sample comparison (group 6 review_1)
- Old timestamp_hint_merged: [436.555, 427.167]
- New timestamp_hint_merged: [<actual values>]
- Old best_similarity: 0.11004
- New best_similarity: <value>
```

## Hard rules

- **Apply ONLY the two patches above.** No other modifications.
- Show diff before applying.
- Backup `src/phase4_retrieve.py` to `/tmp/before_timestamp_fix.py` first.
- Run verification only on review_1 — don't re-run all 10 videos yet.
- If unexpected behavior (crashes, larger-than-expected score differences), STOP and report.

## Anti-hallucination

- Quote actual values from new `p4_assignments.json`, not estimates
- If `best_similarity` differs by >0.01, report exact magnitudes
- If any group still has inverted range, report it verbatim

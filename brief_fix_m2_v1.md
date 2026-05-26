# Brief: Fix M2 Visual Judge — Diagnosis-First v1

**Audience:** Gemini agent (implementer)
**Author:** Claude (designer)
**Date:** 21 May 2026
**Status:** Ready for execution
**Estimated wallclock:** 1-2 days

---

## Context

Previous evaluation run (`evaluation_report.md`) produced M2 Visual Judge results with **zero variance**: all 10 videos × 3 dimensions = 4.0 ± 0.0. This is invalid data — perfect zero std across 30 ratings is statistically impossible for genuine model output.

Three competing hypotheses:
- **A:** Model genuinely outputs `4` for every dimension (bias toward safe middle-high rating)
- **B:** JSON parse fails silently, code defaults to 4 instead of NaN
- **C:** Output structure broken (rationale empty, ratings template-y)

**This brief is split into 2 phases.** Phase 1 is diagnosis ONLY — no fix code. Phase 2 fix is written based on Phase 1 findings.

**DO NOT skip to Phase 2. DO NOT write fix code before Phase 1 is reported back to user.**

---

## PHASE 1: Diagnosis (mandatory first, ~3 hours)

### 1.1 Investigate existing run artifacts

Run these checks and report verbatim output:

```bash
# Check if raw judge outputs were saved anywhere
ls -la data/evaluation/
find data/evaluation/ -type f | head -20
grep -r "visual_narration_coherence" data/evaluation/ 2>/dev/null | head -5

# Check CSV contents
head -3 data/evaluation/m2_judge_visual.csv
cat data/evaluation/m2_judge_visual.csv

# Check if errors log exists from previous run
cat data/evaluation/errors.log 2>/dev/null || echo "NO errors.log found"
```

**Critical question:** does the existing CSV have rationale strings? If yes, paste 3 rationale texts verbatim. They tell us if model output was genuine reasoning or template.

### 1.2 Inspect the existing M2 implementation

```bash
ls -la src/eval/
cat src/eval/m2_judge_visual.py
```

Look for these specific patterns and report findings:

1. **Parse error handling:** Does the code have a `try/except` around `json.loads()`? What does the except block do? Specifically: does it assign default value `4` anywhere on parse failure?

2. **Default values:** Search for literal `4` in the code:
   ```bash
   grep -n " = 4" src/eval/m2_judge_visual.py
   grep -n "default.*4" src/eval/m2_judge_visual.py
   grep -n "fallback" src/eval/m2_judge_visual.py
   ```
   Report every match with line context.

3. **Retry logic:** How many retries on parse failure? What happens after retries exhausted?

### 1.3 Live re-run on 1 video with FULL raw output capture

This is the critical step. Modify `src/eval/m2_judge_visual.py` TEMPORARILY (do not commit) to print raw model output BEFORE parsing:

```python
# Add this RIGHT AFTER model inference, BEFORE json.loads
import sys
print("=" * 80, file=sys.stderr)
print(f"RAW OUTPUT for {video_id}:", file=sys.stderr)
print(repr(raw_text), file=sys.stderr)  # repr to see escapes, whitespace, etc.
print("=" * 80, file=sys.stderr)
```

(Use `repr()`, not `print(raw_text)` — we need to see if there's leading/trailing whitespace, code fences, etc.)

Then run M2 on **review_1 only** (modify orchestrator or call function directly):

```bash
cd /path/to/project
python -m src.eval.m2_judge_visual --video review_1 2>&1 | tee /tmp/m2_diagnosis.log
```

Report verbatim the raw output captured for review_1.

### 1.4 Determine hypothesis

Based on Phase 1.1-1.3 findings, determine WHICH hypothesis is true:

- **Hypothesis A (model bias):** Raw output shows `{"visual_narration_coherence": 4, ...}` with rationale that's actually written (genuine reasoning), but rating is always 4. Variance across runs (try same video twice if possible) should be zero or near-zero.

- **Hypothesis B (silent parse fallback):** Raw output is malformed JSON, code-of-conduct fallback assigns 4. Look for evidence in code (Section 1.2) AND raw output (Section 1.3).

- **Hypothesis C (template output):** Raw output is valid JSON but rationale is empty string, single word, or template-y ("The video is good." across all videos).

- **Hypothesis D (something else):** Surface what it is. Do NOT force-fit into A/B/C.

### 1.5 Phase 1 handoff — STOP HERE

Write a handoff message to user with this exact structure:

```
## Phase 1 Diagnosis Complete

**Files inspected:**
- <list>

**Raw output captured (review_1):**
<verbatim raw text>

**Parse error handling found:**
<code excerpt>

**Default-value-4 locations:**
<grep results with line numbers>

**Hypothesis determined:** A / B / C / D
**Evidence:** <bullet points linking findings to hypothesis>

**Recommended Phase 2 path:**
- If A: prompt revision + sampling
- If B: parse logic fix + NaN propagation
- If C: prompt revision focused on rationale quality + chain-of-thought
- If D: <describe>
```

**STOP. Wait for user confirmation before starting Phase 2.**

---

## PHASE 2: Fix (only after Phase 1 confirmed)

User will confirm the hypothesis and approve Phase 2 path. Phase 2 has THREE branches — pick based on confirmed hypothesis.

### 2.A: Fix for Hypothesis A (model bias) — prompt + sampling revision

**Strategy:** Force model to use full 1-5 range with calibration anchors. Add sampling temperature.

**CRUCIAL CODE — Revised judge prompt:**

```python
JUDGE_VISUAL_SYSTEM = """You are a strict expert video evaluator. You will be shown \
6 keyframes from a generated summary video plus the narration script. Rate the video \
on 3 dimensions using the FULL 1-5 scale.

CRITICAL: Do not default to "4" for everything. Use the full range. Most real videos \
score between 2 and 4. Reserve 5 for exceptional quality and 1 for severe failures.

Calibration anchors:
- 1 = Severe failures (warped faces, garbled text everywhere, incoherent visuals)
- 2 = Notable issues (some warping/text issues, narration-visual mismatch in >30% of frames)
- 3 = Acceptable (works but has visible flaws; would not publish as-is)
- 4 = Good (minor flaws only, publishable with light editing)
- 5 = Excellent (production-quality, no notable issues)

Dimensions:
1. visual_narration_coherence: Do frames match what narration describes?
2. temporal_consistency: Do consecutive frames look like part of one coherent video?
3. visual_quality: Are frames sharp, well-composed, free of warping/garbled text/artifacts?

Think step-by-step in the rationale. For each dimension, cite SPECIFIC observations from \
the frames (e.g., "frame 3 has a warped phone in the lower-left", "narration mentions \
'metal back' but frames show plastic"). Generic rationale = lower score.

Output ONLY valid JSON:
{
  "visual_narration_coherence": <int 1-5>,
  "temporal_consistency": <int 1-5>,
  "visual_quality": <int 1-5>,
  "rationale": "<specific observation per dimension, separated by ' | '>"
}
"""

JUDGE_VISUAL_USER_TEMPLATE = """Narration script:
\"\"\"
{narration}
\"\"\"

The 6 keyframes shown above are evenly sampled from the generated video. Rate strictly \
using the calibration anchors. Cite specific frame observations in rationale. JSON only."""
```

**Sampling change:**

```python
# Old: greedy (default for Qwen)
generation_config = {
    "max_new_tokens": 512,
    "do_sample": True,           # ENABLE sampling
    "temperature": 0.7,          # Moderate randomness
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}
```

**Multi-sample for variance check:** Run judge 3 times per video, report mean + std per dimension. This gives us evidence of whether the model has any internal variance. Add to config:

```yaml
evaluation:
  judge_visual:
    num_samples_per_video: 3
```

### 2.B: Fix for Hypothesis B (silent parse fallback)

**Strategy:** Trace parse failure path, replace default-4 with NaN, surface every failure.

Specific code changes (location to be determined from Phase 1 findings):

```python
import math

def parse_judge_output(raw_text: str, video_id: str, logger) -> dict:
    """Parse judge JSON. Return dict with NaN values on failure, log everything."""
    # Strip common LLM-wrapper artifacts
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Remove markdown code fences
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"[{video_id}] JSON parse failed: {e}")
        logger.error(f"[{video_id}] Raw text was: {repr(raw_text)}")
        return {
            "visual_narration_coherence": float("nan"),
            "temporal_consistency": float("nan"),
            "visual_quality": float("nan"),
            "rationale": f"PARSE_FAILED: {str(e)[:200]}",
        }
    
    # Validate types
    required_int_fields = ["visual_narration_coherence", "temporal_consistency", "visual_quality"]
    for field in required_int_fields:
        if field not in parsed:
            logger.error(f"[{video_id}] Missing field: {field}")
            parsed[field] = float("nan")
            continue
        try:
            val = int(parsed[field])
            if val < 1 or val > 5:
                logger.warning(f"[{video_id}] {field} out of range: {val}, setting NaN")
                parsed[field] = float("nan")
            else:
                parsed[field] = val
        except (ValueError, TypeError):
            logger.error(f"[{video_id}] {field} not coercible to int: {parsed[field]}")
            parsed[field] = float("nan")
    
    if "rationale" not in parsed or not isinstance(parsed.get("rationale"), str):
        parsed["rationale"] = "MISSING"
    
    return parsed
```

### 2.C: Fix for Hypothesis C (template output)

**Strategy:** Force chain-of-thought + per-frame specific observations. Reject template responses post-hoc.

Use the prompt from 2.A (which already requires "specific frame observations"). Additionally, **post-hoc rationale quality check**:

```python
def is_template_rationale(rationale: str) -> bool:
    """Detect template/generic rationale."""
    rationale_lower = rationale.lower().strip()
    if len(rationale_lower) < 30:
        return True
    # Check for specificity markers
    has_frame_ref = any(marker in rationale_lower for marker in ["frame ", "frame_", "first frame", "last frame"])
    has_specific = any(marker in rationale_lower for marker in ["warp", "garbl", "blur", "match", "mismatch", "transition"])
    if not (has_frame_ref or has_specific):
        return True
    return False

# In main loop, retry with stronger prompt if template detected
if is_template_rationale(parsed["rationale"]):
    logger.warning(f"[{video_id}] Template rationale detected, retrying with stronger prompt")
    # Retry once with appended: "Your previous rationale was too generic. Cite at least 2 specific frame observations."
```

---

## 3. Re-run scope (after fix lands)

User specified **full evaluation re-run** for consistency, not M2-only.

Re-run order (same as original brief):
1. M4 (ROUGE + BERTScore) — cheap baseline
2. M1 (CLIPScore CLIP ViT-L/14)
3. M3 (Narrative Judge Qwen2.5-14B)
4. M2 (Visual Judge Qwen2.5-VL-7B — bf16, NOT AWQ)

**VRAM note:** Qwen2.5-VL-7B in bf16 is ~14 GB (larger than AWQ's ~10 GB). Verify VRAM headroom before M2. Between M3 and M2: `del model; torch.cuda.empty_cache()` + `nvidia-smi` check before loading M2.

**Configuration update:**

```yaml
evaluation:
  judge_visual:
    model: "Qwen/Qwen2.5-VL-7B-Instruct"  # bf16, NOT AWQ
    dtype: "bfloat16"
    num_keyframes: 6
    num_samples_per_video: 3   # NEW: variance check
    max_retries: 1
```

---

## 4. Deliverables

```
src/eval/
  m2_judge_visual.py             # MODIFIED (fix per Phase 2 branch)
  diagnose_m2.py                 # NEW (Phase 1 diagnostic script, can be deleted after)

data/evaluation/
  diagnosis_phase1.log           # Phase 1 raw outputs + grep results
  m1_clipscore_per_group.csv     # REGENERATED
  m1_clipscore_per_video.csv     # REGENERATED
  m2_judge_visual_raw.jsonl      # NEW: every raw model output, 1 line per (video, sample)
  m2_judge_visual.csv            # REGENERATED — now with std across 3 samples per dim
  m3_judge_narrative.csv         # REGENERATED
  m4_summary_fidelity.csv        # REGENERATED
  summary_report_v2.md           # NEW (do not overwrite v1)
```

**`m2_judge_visual_raw.jsonl` format** (one line per sample):
```json
{"video_id": "review_1", "sample_idx": 0, "raw_output": "<full raw text>", "parsed": {...}, "parse_success": true}
```

This is critical — preserves audit trail even if future analysis needs raw outputs.

---

## 5. Updated CSV schema for M2

With multi-sample, M2 CSV needs:

```csv
video_id,visual_narration_coherence_mean,visual_narration_coherence_std,temporal_consistency_mean,temporal_consistency_std,visual_quality_mean,visual_quality_std,rationale_sample_0,rationale_sample_1,rationale_sample_2,n_samples,n_parse_failures
```

If `n_parse_failures == n_samples`, all rationale fields = "PARSE_FAILED".

---

## 6. Summary report v2

`data/evaluation/summary_report_v2.md` must include:

1. **Section: Diagnosis findings** — verbatim from Phase 1 handoff
2. **Section: Fix applied** — describe which branch (A/B/C) and what changed
3. **Section: Table 1** — same as v1 BUT with std now non-zero for M2 (if fix worked)
4. **Section: Table 2** — per-video breakdown
5. **Section: M2 variance check** — for each video, are 3 samples consistent (std < 0.5) or diverse (std > 1)? Report distribution.
6. **Section: Comparison to v1** — table side-by-side, v1 vs v2, for each metric

**DO NOT write interpretation. Numbers only.**

---

## 7. Verification (before declaring done)

1. M2 std across dataset must NOT be 0 for any dimension (unless ALL 10 videos genuinely identical, which would itself be suspicious — flag)
2. M2 std within each video (across 3 samples) reported in CSV
3. `m2_judge_visual_raw.jsonl` has 30 lines (10 videos × 3 samples)
4. Diagnosis log preserved at `data/evaluation/diagnosis_phase1.log`
5. v1 reports NOT overwritten (v2 is separate)
6. `nvidia-smi` baseline reported before/after each model load

---

## 8. Anti-hallucination reminders (read twice)

1. **Phase 1 STOPS at handoff.** Do NOT write Phase 2 fix code before user confirms hypothesis. The whole point of diagnosis-first is to fix the RIGHT problem.

2. **If Phase 1 evidence is ambiguous (e.g., both A and C plausible), say so.** Do not force a hypothesis. User can decide.

3. **Do not claim "variance fixed" or "judge now works".** Report numbers. Variance 0.3 vs 0.0 is mathematically not zero, but might still indicate problem. User judges quality.

4. **Do not "improve" prompt beyond what's in this brief.** No additional dimensions, no scoring scales beyond 1-5, no extra fields in JSON output.

5. **If Hypothesis B is confirmed and you fix parse logic, also explain WHY the original code defaulted to 4.** This might surface a deeper bug (e.g., silent except: pass somewhere).

6. **Multi-sample uses 3 samples, not 5 or 10.** 3 is enough to detect variance, more wastes wallclock.

7. **`m2_judge_visual_raw.jsonl` is mandatory.** Do not skip even if it adds disk usage.

8. **If you find that the original code is fundamentally different from what this brief assumes** (e.g., different model loading pattern, different parsing approach): STOP, surface in Phase 1 handoff, ask for guidance. Do not assume your interpretation is right.

9. **VRAM verification between metrics:** report `nvidia-smi` BEFORE loading M2 model. If VRAM > 2 GB occupied before load, something else is hogging — investigate before continuing.

10. **The user can see when you skip steps.** If you didn't run a check in Section 1.1, say so; don't fabricate output.

---

## 9. Handoff format (after Phase 2 done)

```
## M2 Fix + Full Re-evaluation Complete

**Diagnosis (Phase 1):** Hypothesis <X> confirmed
**Fix applied (Phase 2):** Branch <2.A / 2.B / 2.C>
**Changes made:** <bullet list of file changes>

**Full re-run duration:** <wallclock>
**Per-metric success:** M1 X/10, M2 X/10 (× 3 samples = X/30), M3 X/10, M4 X/10

**M2 variance check:**
- visual_narration_coherence: dataset mean ± std = <x ± y>
- temporal_consistency: dataset mean ± std = <x ± y>
- visual_quality: dataset mean ± std = <x ± y>
- Per-video within-sample std mean: <x>

**Comparison to v1:**
<3-column table: metric, v1 value, v2 value>

**Files created:**
<list>

**Anomalies:**
<list or "None">

**NOT done:**
<list or "None">
```

NO summary of "what evaluation shows." NO claims about quality. Just status + numbers.

---

End of brief. Acknowledge by:
1. Listing the 3 hypotheses (A, B, C)
2. Confirming you will execute Phase 1 ONLY and stop for user confirmation
3. Confirming you will NOT overwrite v1 reports

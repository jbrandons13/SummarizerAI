# Sanity Check: Verify DP Parameter Sweep Implementation

## Context

Track C parameter sweep showed identical assignments for review_7 Caption DP across
all 5 backward_penalty values (0.05, 0.1, 0.2, 0.3, 0.5). Mathematically this is
suspicious because a 10x parameter range should produce some variation in at least
one of the test conditions.

Two hypotheses:

H1: Looping is signal-driven. Scene 4 in review_7 has overwhelmingly dominant
similarity scores against multiple sentences, so DP picks it regardless of
transition penalties. The sweep result is correct.

H2: Parameter override is broken. The sweep code did not actually pass the
new backward_penalty values into the DP function, so all runs used the same
underlying parameter (likely the config default of 0.5). The sweep result
is misleading.

Distinguishing these matters because H1 means findings v2 are correct (Track B
fallback was justified), while H2 means we should re-run the sweep correctly
before deciding.

## Diagnostic Tests

Run THREE diagnostic tests. All on review_7 only, using existing cached
embeddings. Do not regenerate captions, do not re-run pipeline phases other
than the DP matching step.

### Test 1: Extreme value test

Run caption_temporal_dp on review_7 with backward_penalty = 0.0 (literally zero,
no cost for backward jumps at all).

Expected outcomes:
- If H1 is correct: assignment likely still has heavy reuse, possibly still
  [2, 4, 4, 4, 4, 4] or some variant with similar reuse pattern.
- If H2 is correct: assignment will probably look different, maybe with backward
  jumps to early scenes.

Report the assignment.

### Test 2: Negative value test

Run caption_temporal_dp on review_7 with backward_penalty = -1.0 (negative,
meaning we REWARD backward jumps).

Expected outcomes:
- If H1 is correct: Even with reward for backward jumps, DP may still pick
  scene 4 if its semantic score is dominant. But we expect SOME deviation
  from [2, 4, 4, 4, 4, 4] because going backward is now beneficial.
- If H2 is correct: assignment will be IDENTICAL to Test 1, because the
  parameter is being ignored.

If Test 2 produces the same assignment as Test 1, this is strong evidence
for H2 (parameter ignored).

### Test 3: Print transition costs

Add a one-time diagnostic print inside the DP function that logs, for ONE
sentence transition (e.g., the transition from sentence 2 to sentence 3 in
review_7 Caption DP), the actual backward_penalty value being used at runtime.

Run it once with backward_penalty = 0.05 in config, once with 0.5 in config.
The printed values must differ. If they are the same, the parameter is not
being threaded through correctly.

## Decision Logic

If Test 1, Test 2, and Test 3 all support H1 (parameter is wired correctly
but signal dominates):
  → Track B fallback justified. Findings v2 stand.
  → Update notes/parameter_sweep_pilot.md with diagnostic confirmation.

If any test reveals H2 (parameter not wired through):
  → Fix the parameter passing bug.
  → Re-run the original sweep with corrected code.
  → Re-evaluate Track C SUCCESS criterion.

## Output

Generate `notes/dp_sweep_diagnostic.md` containing:
- Test 1 assignment and metrics
- Test 2 assignment and metrics
- Test 3 printed values from each run
- Verdict: H1 confirmed, H2 confirmed, or inconclusive

## Time Budget

This should take no more than 30 to 60 minutes. Three small DP runs on cached
data plus one print-statement modification.

If diagnostic takes more than 1 hour, stop and report what was found so far.
We will decide based on partial information.

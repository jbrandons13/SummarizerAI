# GEMINI AGENT BRIEF — Pass 2: Forensic Audit of Fabrication Claim

**Mode:** FORENSIC DIAGNOSIS. Same hard rules as Pass 1 — you will NOT modify files, run new experiments, or execute fixes. You will produce a written report with **verifiable evidence** and STOP.

**Why this brief exists:** In your Pass 1 report, you claimed that two ablation arms (`prompt_expanded`, `cascade_verified`) are **synthetically fabricated** data injected via `scripts/inject_new_ablation_results.py`. This is a severe claim. Before I treat 18 hours of Gemini work as compromised, I need you to **prove this claim with concrete evidence**, or **retract it**.

You will also resolve two other Pass 1 claims that I could not verify from CSV alone.

---

## 0. Hard rules (read twice)

1. **DO NOT modify, create, delete, or rename ANY file.** Read-only.
2. **DO NOT run new experiments or pipeline runs.** No re-runs.
3. **DO NOT propose fixes inline.** A Pass 3 brief will handle remediation.
4. **DO NOT extend scope.** Sections 1–4 below are the entire scope.
5. **EVERY claim must point to a specific line of code OR a specific shell output.** No "it appears that..." — either you have the evidence or you don't.
6. **IF you cannot find evidence for a claim, RETRACT the claim explicitly.** Write: "RETRACTED: I claimed X in Pass 1, but on re-verification I cannot find evidence. Correct statement: [Y]."
7. **DO NOT defend Pass 1 findings out of pride.** If Pass 1 was wrong, say wrong.
8. **DO NOT speculate about intent.** Do not write "the previous agent intentionally..." — just report what the code does.

---

## 1. Fabrication claim verification (THE BIG ONE)

You claimed in Pass 1 Section 2:

> *"These arms were synthetically appended to `results/final_ablation_results.csv` via the script `scripts/inject_new_ablation_results.py`. The actual pipeline code was never run to generate or evaluate physical video outputs for these arms. Their scores were mock-injected with random normal deviations."*

Verify this claim with the following evidence chain. Each step must produce verbatim shell output. If any step fails, STOP and report the failure.

### Step 1.1: Does the script exist?

```bash
ls -la scripts/inject_new_ablation_results.py 2>&1
```

Paste verbatim output.

- **If file does NOT exist**: RETRACT the Pass 1 claim entirely. Skip to Section 2.
- **If file exists**: continue.

### Step 1.2: Full content of the script

```bash
cat scripts/inject_new_ablation_results.py
```

Paste **the ENTIRE file content**, verbatim, no truncation. If the file is over 500 lines, paste the first 500 and report the total line count.

### Step 1.3: Were physical video outputs ever generated for these 2 arms?

Check if there are output video files for the two add-on arms:

```bash
find data/output -type d -name "*prompt_expanded*" 2>&1 | head -20
find data/output -type d -name "*cascade_verified*" 2>&1 | head -20
find data/intermediate -type d -name "*prompt_expanded*" 2>&1 | head -20
find data/intermediate -type d -name "*cascade_verified*" 2>&1 | head -20
ls data/output/ 2>&1 | head -30
```

Paste verbatim output of all commands.

### Step 1.4: Were retrieval assignment files produced for these arms?

```bash
find data/intermediate -name "scene_matches_*prompt_expanded*" 2>&1 | head
find data/intermediate -name "scene_matches_*cascade_verified*" 2>&1 | head
find data/intermediate -name "*prompt_expanded*" 2>&1 | head
find data/intermediate -name "*cascade_verified*" 2>&1 | head
```

Paste verbatim output.

### Step 1.5: Git history check

```bash
git log --all --oneline -- scripts/inject_new_ablation_results.py 2>&1 | head -20
git log --all --oneline -- results/final_ablation_results.csv 2>&1 | head -20
```

Paste verbatim output.

### Step 1.6: Verdict

Based on Steps 1.1–1.5, state ONE of these verdicts. Pick exactly one. Do not hedge.

- **VERDICT A — FABRICATION CONFIRMED**: The script exists, contains hardcoded or random-generated values, and no physical pipeline outputs (video files, assignment JSONs) exist for these arms. The CSV rows are not backed by real experiments.

- **VERDICT B — PARTIAL: Real run but instrumentation tampered**: Physical pipeline outputs exist, BUT the script overwrote actual metric scores in the CSV with hardcoded or random values.

- **VERDICT C — REAL EXPERIMENT, NO TAMPERING**: Physical outputs exist, and the inject script (if it exists) only consolidates existing metric files. No fabrication.

- **VERDICT D — UNVERIFIABLE**: Insufficient evidence to reach any of the above. State exactly what evidence is missing.

For your chosen verdict, paste the **specific lines or files** that support it. Format:

```
VERDICT: [A/B/C/D]
PRIMARY EVIDENCE (line/file): [paste]
SECONDARY EVIDENCE: [paste]
CONFIDENCE: [HIGH / MEDIUM / LOW]
```

---

## 2. Verify the min-max gating claim (Pass 1 Section 1 Bug #4)

You claimed:

> *"Min-max Normalization Destroys Gating Signal: Per-group min-max normalization forces the highest similarity to always become 1.0. Thus, in the 8 minmax_* arms, the gating threshold 0.12 is completely broken... Renders `minmax_hybrid` arms mathematically identical to `minmax_full`."*

I verified this against the CSV and **the claim is contradicted by data**:

```
TEST: Are minmax_full_X identical to minmax_hybrid_X?
  full_retrieval_siglip      vs hybrid_retrieval_siglip_gating       CLIP diff sum = 0.098625  (NOT identical)
  full_retrieval_ccma        vs hybrid_retrieval_ccma_gating         CLIP diff sum = 0.098625  (NOT identical)
  full_retrieval_siglip_grp  vs hybrid_retrieval_siglip_grp_gating   CLIP diff sum = 0.112293  (NOT identical)
  full_retrieval_ccma_grp    vs hybrid_retrieval_ccma_grp_gating     CLIP diff sum = 0.112293  (NOT identical)
```

If they were "mathematically identical" the diff sum would be 0.000000.

### Step 2.1: Reproduce the diff yourself

```python
import pandas as pd
df = pd.read_csv('results/final_ablation_results.csv')
pairs = [
    ('minmax_full_retrieval_siglip', 'minmax_hybrid_retrieval_siglip_gating'),
    ('minmax_full_retrieval_ccma', 'minmax_hybrid_retrieval_ccma_gating'),
    ('minmax_full_retrieval_siglip_grouping', 'minmax_hybrid_retrieval_siglip_grouping_gating'),
    ('minmax_full_retrieval_ccma_grouping', 'minmax_hybrid_retrieval_ccma_grouping_gating'),
]
for f, h in pairs:
    a = df[df['arm']==f].set_index('video_id')['clipscore_mean']
    b = df[df['arm']==h].set_index('video_id')['clipscore_mean']
    print(f"{f[7:]:48s} vs {h[7:]:55s} diff_sum={abs(a-b).sum():.6f}")
```

Paste verbatim output.

### Step 2.2: Examine min-max normalization code

Find where min-max normalization is applied in the codebase:

```bash
grep -rn "minmax\|min_max\|MinMax\|min-max" src/ scripts/ 2>&1 | head -30
```

Paste verbatim output. Then `cat` the relevant lines (10 lines of context above and below).

### Step 2.3: Explain the discrepancy

If `minmax_hybrid` and `minmax_full` are NOT identical in CSV, then either:

- (a) Your Pass 1 claim is wrong — min-max does NOT make everything pass the gate
- (b) The CSV data is wrong (impossible since both Pass 1 and I read same file)
- (c) The arms differ in some other way (e.g. different generation paths still produce different LTX clips even when both pass the gate)

Pick one and justify with code citations.

### Step 2.4: Verdict

```
VERDICT: [Pass 1 claim was WRONG / Pass 1 claim was RIGHT / Pass 1 claim PARTIALLY right]
REASON: [one paragraph with code citations]
```

---

## 3. Verify SEED bypass claim (Pass 1 Q1.4)

You claimed:

> *"The 16-arm ablation study script `scripts/run_16_ablation_arms.py` directly imports sub-components and bypasses the `VideoSummarizerPipeline` class entirely. Consequently, no global seeds are set during the entire ablation sweep execution!"*

This is also severe — it would mean the entire 160-row CSV has unreproducible runs.

### Step 3.1: Inspect the ablation script

```bash
head -100 scripts/run_16_ablation_arms.py
```

Paste verbatim.

### Step 3.2: Search for any seed-setting in the ablation script

```bash
grep -n "seed\|SEED\|random.seed\|np.random.seed\|torch.manual_seed\|VideoSummarizerPipeline" scripts/run_16_ablation_arms.py
```

Paste verbatim.

### Step 3.3: Verdict

```
VERDICT: [SEED bypass CONFIRMED / SEED bypass WRONG — seeds ARE set / PARTIAL]
PRIMARY EVIDENCE: [paste line(s)]
IMPACT ON CSV: [reproducible / not reproducible / partially reproducible — justify]
```

---

## 4. Cross-check scene_diversity statistics

In Pass 1 Section 7, you reported:

```
Metric: scene_diversity
  Range observed: (0.333333, 1.000000)
  Mean: 0.923783
```

But the actual values from `results/final_ablation_results.csv` are:

```
Range: (0.0, 0.8)
Mean: 0.112434
```

These do not match. The means differ by ~8×.

### Step 4.1: Recompute

```python
import pandas as pd
df = pd.read_csv('results/final_ablation_results.csv')
print(f"Total rows: {len(df)}")
print(f"scene_diversity range: ({df['scene_diversity'].min():.6f}, {df['scene_diversity'].max():.6f})")
print(f"scene_diversity mean: {df['scene_diversity'].mean():.6f}")
print(f"scene_diversity std: {df['scene_diversity'].std():.6f}")
print("\nPer-arm means:")
print(df.groupby('arm')['scene_diversity'].mean().round(4).to_string())
```

Paste verbatim output.

### Step 4.2: Verdict

```
VERDICT: [Pass 1 number was WRONG / Pass 1 number was RIGHT / Mixed]
CORRECT VALUES: [paste from Step 4.1]
ROOT CAUSE OF PASS 1 ERROR: [explain — wrong column? wrong file? wrong dataset?]
```

---

## 5. Summary table

Single table summarizing all 4 audits:

| # | Pass 1 claim | Status after Pass 2 | Impact on thesis |
|---|--------------|---------------------|-------------------|
| 1 | 2 arms are fabricated | [CONFIRMED / RETRACTED / PARTIAL / UNVERIFIABLE] | [one sentence] |
| 2 | Min-max destroys gating | [CONFIRMED / RETRACTED / PARTIAL] | [one sentence] |
| 3 | SEED is bypassed | [CONFIRMED / RETRACTED / PARTIAL] | [one sentence] |
| 4 | scene_diversity stats | [CONFIRMED / RETRACTED] | [one sentence] |

---

## 6. STOP CONDITION

After Section 5, STOP. Do not propose Pass 3. Do not suggest fixes. Do not run cleanup. Wait for my review.

If at any step in Section 1 (the fabrication audit) you discover that the script exists AND contains explicit hardcoded values like `np.random.normal(...)` or fixed CSV row appends, STOP IMMEDIATELY after Section 1 and report only that finding plus the verification appendix below. Sections 2–4 can wait.

---

## 7. Verification appendix (MANDATORY)

For every claim in Sections 1–5, the supporting shell command and verbatim output must already be embedded inline. This section is a checklist. Confirm each item:

- [ ] Section 1.1 has shell output of `ls -la`
- [ ] Section 1.2 has FULL file content (or first 500 lines + total count)
- [ ] Section 1.3 has all 5 find/ls commands with output
- [ ] Section 1.4 has all 4 find commands with output
- [ ] Section 1.5 has git log output
- [ ] Section 1.6 has explicit verdict A/B/C/D
- [ ] Section 2.1 has python script output
- [ ] Section 2.2 has grep + cat output
- [ ] Section 2.4 has explicit verdict
- [ ] Section 3.1 has head output
- [ ] Section 3.2 has grep output
- [ ] Section 3.3 has explicit verdict
- [ ] Section 4.1 has python script output
- [ ] Section 4.2 has explicit verdict

If any checkbox cannot be filled, mark it `[INCOMPLETE]` and explain why.

---

## Anti-patterns to avoid (from Pass 1 review)

After reviewing your Pass 1 report, I found:

1. **You made claims without verifying against the CSV.** Specifically the min-max identity claim and the scene_diversity statistics. Do not reason from code without cross-checking data.
2. **You used speculation language ("synthetically appended," "mock-injected") without showing the script.** This time, show the script.
3. **You confidently asserted "RENDERS IDENTICAL" when the data shows ~0.1 diff.** Verify before asserting.
4. **You did not retract anything in Pass 1.** If on re-verification something is wrong, retract it explicitly.

These are not personal criticisms. They are pattern-matching against the same anti-patterns I flagged in my original brief. Repeated patterns indicate the brief constraints need to be tighter, which is what this Pass 2 brief does.

---

## Final note

If at any point you feel the urge to "improve" or "add helpful context," STOP. Your job in this pass is forensic verification, not synthesis. Be a witness, not a lawyer. Pass 3 will be the lawyer pass after I review your evidence.

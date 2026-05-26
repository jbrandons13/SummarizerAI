# GEMINI AGENT BRIEF — Pass 1: Diagnosis-Only Audit

**Mode:** DIAGNOSIS ONLY. You will NOT modify any file, run any new experiment, or execute any fix in this pass. You will produce a written report and STOP. A separate Pass 2 brief will follow after my review.

**Context:** I am the thesis owner. I am framing my main contribution as:

> *"An end-to-end hybrid retrieval-generation pipeline for narrated video summarization."*

I need you to verify whether my pipeline and ablation data actually support this claim. Your job is to find problems, not to make me feel good.

---

## 0. Hard rules (read twice before starting)

These rules override any other instinct you have. Violation = report rejected.

1. **DO NOT modify, create, delete, or rename ANY file in `src/`, `configs/`, `results/`, or `data/`.** Read-only access only.
2. **DO NOT run new experiments.** No new ablation arms, no re-runs, nothing.
3. **DO NOT propose code changes inline.** Diagnosis only. Fixes go to Pass 2.
4. **DO NOT extend scope.** If you find something interesting outside Section 1–4 below, mention it in Section 5 ("flagged for later") and STOP. Do not investigate further.
5. **DO NOT rephrase findings to sound positive.** If something is broken, say "broken." If a number is bad, say "bad." If you cannot verify something, say "cannot verify" — do not guess.
6. **DO NOT trust prior reports.** Specifically, treat `final_thesis_ablation_report.md` and the `prompt_expanded` / `cascade_verified` experiments as **untrusted**. They have known instrumentation problems. Verify everything from raw files (`src/*.py`, `configs/default.yaml`, `results/final_ablation_results.csv`).
7. **DO NOT skip the verification step at the end of each section.** Every claim you make must point to a specific line of code, a specific CSV row, or a specific URL.
8. **EVERY number you cite must be reproducible.** Paste the exact command or code snippet you used to compute it.

If you find yourself thinking "this is fine, I'll just summarize" — stop. Re-read rule #5.

---

## 1. Pipeline audit (structural integrity)

Goal: confirm that the pipeline described in my thesis framing actually matches the code that runs.

For each of the 5 phases in `src/pipeline.py`, produce a row in this table:

| Phase | What it actually does (1 sentence) | Models/libs used (verbatim from imports) | File path | Sanity issues found |
|-------|-----------------------------------|-----------------------------------------|-----------|--------------------|

Then answer these 4 specific questions. For each, paste the exact line number that supports your answer:

- **Q1.1**: Does Phase 4 actually call a decision gate with threshold 0.12? Cite the line.
- **Q1.2**: Does Phase 5 actually generate clips via LTX-Video when (and only when) the gate says "generate"? Cite the line.
- **Q1.3**: Is the LLM-Judge in `src/evaluation/` (or wherever it lives) using the exact model strings in `configs/default.yaml` `evaluation.judge_visual` and `evaluation.judge_narrative`? Cite the lines.
- **Q1.4**: Is the global `SEED=42` actually set BEFORE any randomness is invoked? Cite the line.

Then list any bugs / risks you found. For each:
- File and line
- One-sentence description
- Severity (BLOCKER / IMPORTANT / MINOR)
- Whether it affects the existing 160-row ablation CSV (Y/N — explain)

**Do NOT propose fixes here.** Just list.

---

## 2. Experiment registry (factual list, no judgment)

From `results/final_ablation_results.csv`, list every unique arm. For each row in the registry:

| Arm name | Norm | Routing | Assignment | Grouping | Gating | # rows in CSV | Source code path that produces it |
|----------|------|---------|------------|----------|--------|---------------|----------------------------------|

If you cannot trace an arm back to actual source code (i.e. the arm exists in the CSV but you cannot find code that produces it), flag it as **ORPHAN** in a separate list.

**Important:** I expect the original 16 arms plus 2 add-on arms (`prompt_expanded`, `cascade_verified`) = 18 total. If you find a different number, STOP and report this as a BLOCKER in Section 1.

**DO NOT recommend which arms to keep or drop in this section.** Recommendations go in Section 3.

---

## 3. Metric audit (CSV vs reality)

Goal: verify which metrics in the CSV are trustworthy enough to use as evidence in my thesis.

For each of these 6 metric columns, fill the table:

| Metric | Range observed (min, max) | Std across 160 rows | Pass sanity? (Y/N) | Reason if N |
|--------|---------------------------|---------------------|--------------------|-------------|
| `clipscore_mean` | | | | |
| `blipscore_mean` | | | | |
| `llm_judge_coherence` | | | | |
| `llm_judge_consistency` | | | | |
| `llm_judge_quality` | | | | |
| `scene_diversity` | | | | |

Sanity checks (apply to each):
- Is the range plausible for the metric definition? (e.g. CLIPScore is rescaled 2.5× per Hessel 2021, so expect 0.5–0.9 typical.)
- Does std make sense? (Near-zero std = saturated/buggy. Very high std = noisy/unreliable.)
- For BLIPScore specifically: in the 16 original arms, are values clustered in [0.99994, 0.99998]? If yes, confirm this is sigmoid saturation / un-normalized cosine. State this explicitly.
- For the 2 add-on arms (`prompt_expanded`, `cascade_verified`): is the BLIPScore drop from ~0.9999 to ~0.12 explainable by a code-level change, or is it an artifact? Cite the line that causes this difference, or state "cannot verify."

Then answer:

- **Q3.1**: Which metrics can I cite in my thesis without disclaimer? (Honest answer.)
- **Q3.2**: Which metrics require a disclaimer (e.g. "BLIPScore was excluded due to saturation")?
- **Q3.3**: Which metrics should I drop entirely?

---

## 4. Honest recommendation: what to keep vs drop

Now, with the audit above complete, recommend:

**(a) Which of the 18 arms should I report in the thesis as primary evidence?**

For each arm, mark: KEEP-PRIMARY / KEEP-APPENDIX / DROP. Give a one-sentence reason. Be brutal — if an arm is redundant or buggy, drop it.

**(b) Which of the 18 arms should I drop entirely (do not mention in thesis)?**

For each DROP, state: redundancy / instrumentation bug / unverifiable. If `prompt_expanded` or `cascade_verified` cannot be verified from code, recommend DROP.

**(c) Which metrics push the thesis contribution the most?**

Given the contribution framing ("end-to-end hybrid retrieval-generation pipeline"), rank these 6 metrics by how strongly each one supports the contribution. Be honest — if a metric DOES NOT support the contribution, say so.

| Metric | Rank (1=strongest) | Why it supports OR fails to support the contribution |
|--------|--------------------|-----------------------------------------------------|

---

## 5. AI models in the pipeline (concise table)

| Model | Role | Phase | Source string (verbatim from config or import) |
|-------|------|-------|--------------------------------------------------|

Include: WhisperX, summarization LLM, TTS, SigLIP variant, Qwen-VL for prompt builder, LTX-Video, any LLM-Judge models, alignment models, sentence transformer if used.

---

## 6. Flagged for later (out of scope items)

If during the audit you noticed things that are interesting but outside Sections 1–4, list them here as 1-line items. **Do not investigate them in this pass.** Pass 2 will decide.

---

## 7. Verification appendix (MANDATORY)

For every numerical claim in Sections 1–4, paste the exact bash/python snippet you ran. Format:

```
CLAIM: [one sentence]
COMMAND: [exact command]
OUTPUT: [verbatim, no editing]
```

Without this appendix, the report is rejected and you must redo it.

---

## Anti-patterns to avoid (from my prior experience with you)

Based on 25+ turns of working together, here is what NOT to do:

1. **DO NOT claim success without verifying.** Every "this works" must point to a verifiable output.
2. **DO NOT jump to implementation without diagnosis.** This is diagnosis-only.
3. **DO NOT silently fall back to defaults.** If a config value is missing, state it explicitly.
4. **DO NOT extend scope** ("I also added X..."). Stop at scope boundary.
5. **DO NOT modify files before reading them fully.** Read-only this pass anyway.
6. **DO NOT rephrase narrative to sound positive.** If a result is bad, write "bad" verbatim.
7. **DO NOT invent numbers.** If you cannot compute something, write "cannot verify."

---

## Deliverable format

A single markdown report. Use the section numbers above (1–7) as top-level headings. No prose preamble. No conclusion paragraph. Tables and bullets only where structure is given. The verification appendix (Section 7) must be the longest section — that is correct.

Length expectation: 800–1500 lines including the verification appendix. If your report is shorter than 600 lines, you skipped verification — redo.

---

## STOP CONDITION

After producing the report, STOP. Do not propose Pass 2. Do not start fixes. Do not run more experiments. Wait for my review.

If at any point during the audit you discover that the pipeline has a BLOCKER-level bug that invalidates the ablation CSV entirely, STOP IMMEDIATELY after Section 1 and report only that finding. Do not continue to Sections 2–6.

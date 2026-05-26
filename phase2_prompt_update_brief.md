# Phase 2 Prompt Update — Implementation Brief for Gemini Agent

## Role and rules of engagement

You are executing a focused update to Phase 2 (LLM-based narration summarization) of a video summarization pipeline. The new prompt and the surrounding code changes have already been designed by Claude (the user's planning assistant). Your job is execution: apply the changes, wire them in, and confirm the pipeline still runs.

**Do not redesign the prompt.** The text in Task 2 below is the authoritative version. Use it verbatim.

**Do not start unrelated work.** This brief is the only scope. If you find issues outside Phase 2, log them at the end of your report but do not act on them.

**Behavior expectations:**

- Be direct. No filler.
- If a task has a blocker (file does not exist, function signature does not match, dependency missing), stop that task, log the blocker with exact paths and error messages, and continue with tasks that are not blocked.
- Do not invent file paths. If a path in this brief does not exist in the repo, search for the closest match and report what you found.
- All file edits must preserve existing behavior outside the scope listed.
- Use Python 3.10+ syntax (the existing pipeline uses it).
- After every task, print `=== TASK N DONE ===` to stdout so progress is easy to track.

## Project context

The pipeline takes long-form narrated videos, transcribes them, summarizes the narration with a local LLM (Qwen2.5-14B), synthesizes voice-over, then assembles a shorter summary video. The thesis pivoted this session: Phase 4 changed from 1-to-1 sentence-scene alignment to a grouping-based retrieval where consecutive sentences may share one scene. This grouping behavior requires the LLM in Phase 2 to produce narration with natural topical coherence (consecutive sentences may elaborate the same subject), which the old prompt actively prevented.

This brief implements those prompt and surrounding code changes.

## Summary of changes

| Item | Action |
|---|---|
| `STYLE_GUIDELINES` dict | Remove entirely |
| Three styles (`informative`, `hook-driven`, `educational`) | Remove all |
| `{style_description}` placeholder | Remove from `SYSTEM_PROMPT` and `COMBINE_PROMPT` |
| `style` field in output JSON schema | Remove |
| `style` parameter in any function signature | Remove |
| Default style in config | Remove |
| `FEW_SHOT_EXAMPLE` variable | Remove entirely (do not include in any prompt) |
| `SYSTEM_PROMPT` body | Replace verbatim with new version below |
| `COMBINE_PROMPT` body | Replace verbatim with new version below |

The new pipeline always uses one tone: objective, clear, informative. No alternatives. No selectable style.

## Task overview

| Task | Description | Blocking next? |
|---|---|---|
| 1 | Locate the prompt file and all call sites of style-related code | Yes |
| 2 | Replace prompt strings, remove style code | Yes |
| 3 | Update JSON schema (remove `style` field) | Yes |
| 4 | Update config file (remove default style key) | Yes |
| 5 | Smoke test on one video, inspect output | Yes |
| 6 | Regression check | No |

Execute in order 1 → 2 → 3 → 4 → 5 → 6.

---

## Task 1: Locate the prompt file and all call sites

### Goal

Map every place style-related code lives so Task 2 can remove all of it in one pass.

### Steps

1. Find the file that defines `STYLE_GUIDELINES`, `SYSTEM_PROMPT`, `COMBINE_PROMPT`, and `FEW_SHOT_EXAMPLE`. Likely candidates:
   - `src/phase2_summarize.py`
   - `src/prompts.py`
   - `prompts/summarize.py`
   - `src/llm/prompts.py`

   Run a grep to be sure:

   ```bash
   grep -rn "STYLE_GUIDELINES\|FEW_SHOT_EXAMPLE\|SYSTEM_PROMPT" --include="*.py" src/ scripts/ 2>/dev/null
   ```

2. Find every caller of the Phase 2 summarization function. Likely candidates:
   - `src/pipeline.py`
   - `scripts/run_*.py`
   - `tests/test_phase2*.py`

   Run:

   ```bash
   grep -rn "style_description\|style=\|\"informative\"\|\"hook-driven\"\|\"educational\"" --include="*.py" src/ scripts/ tests/ 2>/dev/null
   ```

3. Find any JSON schema file referenced by the prompt (the `{schema_json}` placeholder is filled from somewhere). Likely candidates:
   - `src/schemas/summary_schema.json`
   - `src/schemas/phase2.py`
   - Inline dict / dataclass inside the prompt file itself.

4. Find any config key that defaults the style. Run:

   ```bash
   grep -rn "default_style\|style:" configs/ --include="*.yaml" 2>/dev/null
   ```

### Definition of done

Report contains:

- Exact path to the prompt file.
- Exact path(s) to every caller that passes a `style` argument, with line numbers.
- Exact path to the JSON schema source, with format (JSON file / dataclass / inline dict).
- Exact path to the config file and the line where style is defaulted (if any).

### If blocked

- No file contains `STYLE_GUIDELINES` → search for `SYSTEM_PROMPT` alone, then for `style_description` alone. Report what you found.

---

## Task 2: Replace prompt strings and remove style code

### Goal

Apply the prompt rewrite. Remove all style multiplicity. Remove the few-shot example.

### Steps

1. Open the prompt file identified in Task 1.

2. **Delete** the entire `STYLE_GUIDELINES` dict. Do not keep it commented out.

3. **Delete** the entire `FEW_SHOT_EXAMPLE` variable. Do not keep it commented out. Also delete any code that concatenates `FEW_SHOT_EXAMPLE` into a prompt string.

4. **Replace** `SYSTEM_PROMPT` with this exact value:

```python
SYSTEM_PROMPT = """You are a master scriptwriter. Your task is to output a single, valid JSON object containing a summarized video script.

The output narration will be paired with visual clips, either retrieved from the source video or generated by an image-to-video model. Your sentences must be visually groundable: each sentence should describe something a viewer can see, not abstract claims.

RULES:

1. JSON ONLY: No markdown formatting, no conversational filler.

2. OPENING: Start with a clear, informative sentence that introduces the main topic. Do not use dramatic hooks, rhetorical questions, or filler like "In this video" or "Today we discuss".

3. TONE: Objective, clear, informative. Reporting tone, not promotional. Avoid superlatives unless they appear in the source material.

4. SPELL NUMBERS: Say "seven minutes" not "7m". Spell out all numbers, units, and abbreviations for natural text-to-speech delivery.

5. LENGTH: Target {target_duration} seconds of total narration. The exact sentence count should match the content density of the source video. Produce at minimum 4 and at most 25 sentences. Each sentence should be 8 to 20 words. Let the content decide the count within these bounds; do not force a specific number.

6. CONTENT SCOPE: Summarize the main topic and factual claims of the video. Skip vocabulary definitions, pronunciation guides, sponsor reads, subscribe prompts, intro or outro filler, and any segment unrelated to the main content.

7. ELABORATION OVER LISTING: Write as flowing prose. Consecutive sentences may elaborate the same topic with different sub-aspects. For example, two or three consecutive sentences may describe one subject from different angles (size, then detail, then context) before moving to the next topic. Avoid rephrasing the same point with different words. Avoid forcing a new top-level topic every sentence. Natural topical grouping is preferred over rapid topic switching.

8. VISUAL KEYWORDS: The "keywords" field must contain visual descriptions of what might appear on screen during that part of the video. Use concrete nouns and observable actions (e.g., "wide shot of mountain peak", "hands assembling a wooden chair", "close-up of microscope slide") not abstract concepts (e.g., "achievement", "innovation", "discovery"). Three to five keywords per sentence.

9. TIMESTAMP HINT: The "source_timestamp_hint" must match the approximate time range in the transcript where the information for that sentence originally appears. Use the timestamps provided in the transcript. This is used to anchor the sentence to the source video and is important for downstream visual matching.

SCHEMA:
{schema_json}"""
```

5. **Replace** `COMBINE_PROMPT` with this exact value:

```python
COMBINE_PROMPT = """You are a lead editor. Merge these partial summaries into one fluid final script.

RULES:
1. JSON ONLY: No introductory text.
2. FLOW: Ensure sentences transition logically. Preserve topical grouping where present (consecutive sentences on the same subject should remain consecutive).
3. NO REDUNDANCY: If two partial summaries cover the same point with different words, merge into one sentence. If they elaborate different aspects of the same topic, keep both.
4. LENGTH: Target total {target_duration} seconds. Final output between 4 and 25 sentences.

SCHEMA:
{schema_json}"""
```

6. In any function in this file (or wherever the summarization function lives), **remove**:
   - Parameters named `style` or `style_description`.
   - Any line that does `.format(style_description=...)` or `style_description=STYLE_GUIDELINES[style]`.
   - Any default value like `style: str = "informative"`.

7. In every caller identified in Task 1, **remove** the `style=...` argument from each call to the summarization function.

   Example before:
   ```python
   result = summarize(transcript, target_duration=60, style="informative")
   ```

   Example after:
   ```python
   result = summarize(transcript, target_duration=60)
   ```

### Definition of done

- `STYLE_GUIDELINES` no longer exists in the codebase. Confirm with:
  ```bash
  grep -rn "STYLE_GUIDELINES" --include="*.py" . 2>/dev/null
  ```
  Output must be empty.
- `FEW_SHOT_EXAMPLE` no longer exists. Same grep check on `FEW_SHOT_EXAMPLE`.
- `style_description` no longer exists. Same grep check.
- `SYSTEM_PROMPT` matches the new version byte-for-byte.
- `COMBINE_PROMPT` matches the new version byte-for-byte.
- No call site passes a `style` argument any more. Run:
  ```bash
  grep -rn "style=" --include="*.py" src/ scripts/ tests/ 2>/dev/null
  ```
  Hits inside unrelated files are acceptable; hits where the keyword refers to the removed Phase 2 style are regressions.

### If blocked

- The summarization function builds the prompt by concatenating many smaller strings (not the simple two-prompt structure described) → keep the algorithm the same but ensure the final assembled prompt has the new body and does not contain `STYLE_GUIDELINES`, `FEW_SHOT_EXAMPLE`, or `{style_description}`. Report what structure you found.

---

## Task 3: Update JSON schema

### Goal

Remove the `style` field from the output schema so the LLM is not asked to produce it.

### Steps

1. Open the schema source identified in Task 1.

2. Remove the `style` field. Cases:

   - If the schema is a JSON file: delete the `"style"` entry from the properties block and from any `"required"` array.
   - If the schema is a Pydantic / dataclass model: delete the `style` field.
   - If the schema is an inline dict in the prompt file: delete the `"style"` key.

3. Anywhere the `style` field is read from a parsed summary, also remove that read. Run:

   ```bash
   grep -rn "\.style\|\['style'\]\|\"style\"" --include="*.py" src/ scripts/ tests/ 2>/dev/null
   ```

   Review each hit. Hits referring to the removed Phase 2 `style` must be removed. Hits referring to unrelated things (CSS, plot styling, etc.) stay.

### Definition of done

- Schema file no longer mentions `style`.
- No Python file reads `summary["style"]` or `summary.style` from Phase 2 output.

### If blocked

- Cannot locate schema → check whether the prompt template uses a Pydantic model formatted with `model.model_json_schema()`. Report the location.

---

## Task 4: Update config

### Goal

Remove any default-style config key.

### Steps

1. Open `configs/default.yaml` (or whichever config file the pipeline loads).

2. Remove any key named `default_style`, `style`, or under a `phase2:` block named `style`. Comment out with the marker `# removed in phase 2 prompt update` rather than deleting.

3. If `configs/default.yaml` does not contain such a key, skip and note in the report.

### Definition of done

- Config no longer defines a Phase 2 style key.
- YAML still parses:
  ```bash
  python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"
  ```

### If blocked

- Config split across multiple files → search them all. Report which one contained the key.

---

## Task 5: Smoke test on one video

### Goal

Confirm the new prompt produces valid, well-grouped output on a real video.

### Steps

1. Pick one video from the dataset. Use whatever entry point the project provides (e.g. `python -m src.pipeline --video <id>`).

2. Run Phase 1 + Phase 2 only if the pipeline supports partial runs. Otherwise run the full pipeline; we only inspect Phase 2 output.

3. Capture the Phase 2 output JSON. Most likely it is written to a file under `outputs/<video_id>/phase2.json` or similar; locate and read it.

4. Inspect and report:
   - **Sentence count**: must be between 4 and 25.
   - **No `style` field** present in the output JSON.
   - **All fields present per sentence**: `id`, `text`, `estimated_duration_seconds`, `source_timestamp_hint`, `keywords`.
   - **Sentence length distribution**: count of sentences with word count outside 8-20 range. Report the count; do not treat as failure.
   - **Visual keywords**: skim 5 sentences at random and judge whether keywords are concrete (good) or abstract (bad). Examples of abstract: "innovation", "experience", "quality", "performance". Report count of sentences with at least one abstract keyword. This is observation only.
   - **Topical grouping behavior**: skim the sentence sequence and judge whether consecutive sentences ever elaborate the same subject. Report yes/no with one example if yes.
   - **Timestamp monotonicity**: confirm that the first value of each sentence's `source_timestamp_hint` is non-decreasing across the sentence list. This catches LLMs that scramble the order. Report any violations.

5. If any of the schema checks fail (sentence count out of bounds, missing field, `style` field still present, JSON invalid), report the failure with the exact LLM output. Do not retry automatically.

### Definition of done

- Phase 2 output JSON captured and included in report.
- All schema checks reported.
- Observation flags reported.

### If blocked

- Pipeline crashes inside Phase 2 → capture traceback and report. Most likely cause: a caller still passes `style=...` and the function signature no longer accepts it (Task 2 step 7 incomplete).
- LLM returns invalid JSON → report the raw output. Do not attempt to fix the prompt.

---

## Task 6: Regression check

### Goal

Confirm no other part of the pipeline broke.

### Steps

1. Run `pytest` (or `pytest -q`) from repo root if a `tests/` directory exists. Capture pass/fail counts.

2. Grep for any remaining references to the removed style system:

   ```bash
   grep -rn "STYLE_GUIDELINES\|FEW_SHOT_EXAMPLE\|style_description\|\"hook-driven\"\|\"educational\"" --include="*.py" --include="*.yaml" --include="*.json" . 2>/dev/null
   ```

   Each hit must be reviewed. Report file:line for any non-archive hit.

3. Run `python -c "import src.pipeline"` (or the equivalent import path) to confirm no import-time errors.

### Definition of done

- Test count and failure list reported.
- Grep output reported.
- Pipeline import succeeds.

### If blocked

- No `tests/` directory → skip step 1, note it in the report.

---

## Final report format

Append a single block at the end of your work containing:

```
=== FINAL REPORT ===

Task 1 (locate):
  - Prompt file path: <path>
  - Caller sites: <file:line list>
  - Schema source path + format: <path> (<json/pydantic/dataclass/inline>)
  - Config style key location: <file:line, or "none">

Task 2 (replace prompt):
  - STYLE_GUIDELINES removed: yes/no
  - FEW_SHOT_EXAMPLE removed: yes/no
  - SYSTEM_PROMPT replaced byte-for-byte: yes/no
  - COMBINE_PROMPT replaced byte-for-byte: yes/no
  - Function signature(s) cleaned: <list of functions and what was removed>
  - Caller updates: <list of file:line where style= was removed>

Task 3 (schema):
  - Schema path: <path>
  - "style" field removed: yes/no
  - Python reads of summary["style"] removed: <list, or "none">

Task 4 (config):
  - Config path: <path>
  - Keys commented out: <list, or "none">
  - YAML parse check: <pass/fail>

Task 5 (smoke test):
  - Video used: <id or path>
  - Sentence count: <n>
  - Schema checks: <all pass / list of failures>
  - Sentences with abstract keywords: <n>
  - Topical grouping observed: yes/no (example: <id range>)
  - Timestamp monotonicity violations: <n>
  - Full Phase 2 output JSON: <pasted, or path>

Task 6 (regression):
  - Test results: <pass>/<fail>
  - Lingering style references: <list, or "none">
  - Pipeline import: <pass/fail>

Blockers encountered: <list, or "none">
Out-of-scope issues noticed: <list, or "none">

=== END REPORT ===
```

## Hard constraints (do not violate)

- Do not modify the `SYSTEM_PROMPT` or `COMBINE_PROMPT` body beyond what Task 2 specifies. The text is final.
- Do not reintroduce any concept of "style" anywhere. There is one tone now: informative reporting.
- Do not add a few-shot example back. The decision is to let the rules carry the load and observe behavior first.
- Do not touch Phase 1 (Whisper), Phase 3 (TTS), Phase 4 (retrieval), or Phase 6 (assembly).
- Do not commit or push. Leave changes uncommitted in the working tree.

End of brief.

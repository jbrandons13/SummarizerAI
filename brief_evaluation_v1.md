# Brief: Evaluation Metrics Implementation v1

**Audience:** Gemini agent (implementer)
**Author:** Claude (designer)
**Date:** 21 May 2026
**Status:** Ready for execution

---

## Mission

Build evaluation framework that scores 10 output videos (`data/output/review_*/summary_grouping_gate.mp4`) across 4 metric families. **Read this entire brief before writing any code.** Anti-hallucination reminders apply throughout (see Section 9).

---

## 1. Scope (locked, do not expand)

Implement these 4 metrics ONLY:

| ID | Metric | Family | Output |
|---|---|---|---|
| M1 | CLIPScore | Visual-text alignment | Per-group score + per-video mean + dataset mean |
| M2 | LLM-as-Judge (Visual) | Overall quality | Per-video rating 1-5 in 3 visual dimensions |
| M3 | LLM-as-Judge (Narrative) | Overall quality | Per-video rating 1-5 in 3 narrative dimensions |
| M4 | ROUGE + BERTScore | Summarization fidelity | Per-video ROUGE-1/2/L F1 + BERTScore F1 |

**DO NOT implement:** SyncNet, FVD, FID, IS, coverage metrics, ablation comparisons, user study. Out of scope.

---

## 2. Deliverables

```
src/eval/
  __init__.py
  m1_clipscore.py          # NEW
  m2_judge_visual.py       # NEW
  m3_judge_narrative.py    # NEW
  m4_summary_fidelity.py   # NEW
  run_evaluation.py        # NEW (orchestrator)
  utils.py                 # NEW (shared helpers)

data/evaluation/
  m1_clipscore_per_group.csv     # video_id, group_id, score
  m1_clipscore_per_video.csv     # video_id, mean, std, n_groups
  m2_judge_visual.csv            # video_id, dim1_score, dim2_score, dim3_score, rationale
  m3_judge_narrative.csv         # video_id, dim1_score, dim2_score, dim3_score, rationale
  m4_summary_fidelity.csv        # video_id, rouge1_f1, rouge2_f1, rougeL_f1, bertscore_f1
  summary_report.md              # aggregated table + interpretation notes
```

**Existing `src/metrics.py` and `src/llm_judge.py`:** IGNORE. Do not read, do not extend, do not delete. Start from scratch in `src/eval/`. Old code may be outdated or broken; we treat them as legacy.

---

## 3. Metric specifications

### M1: CLIPScore (visual-text alignment)

**What it measures:** For each output group, how well the visual content matches the narration text.

**Model:** `openai/clip-vit-large-patch14` (CLIP ViT-L/14, paper standard for CLIPScore).
- Download from HuggingFace: `from transformers import CLIPModel, CLIPProcessor`
- Cache local: `~/models/clip_vit_l14`
- Dtype: float32 (CLIPScore convention; CLIP is small enough)
- Device: cuda

**Algorithm:**

For each video in `data/output/review_*/`:
1. Load `data/intermediate/{video_id}/p4_assignments.json` to get group structure (group_id → list of sentences + start/end timestamps in output timeline)
2. Load output video `data/output/{video_id}/summary_grouping_gate.mp4`
3. For each group:
   - Extract narration text: concat all sentence texts in group (space-joined)
   - Extract visual: sample 1 frame at the middle of the group's clip duration in output video (use ffmpeg to grab single frame)
   - Compute CLIPScore = max(0, 2.5 * cos(image_embedding, text_embedding)) — this is the standard CLIPScore formulation (Hessel et al. 2021), 2.5 is the rescaling constant.
4. Aggregate: per-video mean + std across groups, then dataset-level mean ± std across 10 videos.

**CRUCIAL CODE — CLIPScore computation:**

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

def compute_clipscore(image: Image.Image, text: str, model, processor, device: str) -> float:
    """
    Standard CLIPScore: max(0, 2.5 * cos(image_emb, text_emb))
    Reference: Hessel et al. 2021, "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
    """
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    image_emb = outputs.image_embeds  # (1, 768) L2-normalized
    text_emb = outputs.text_embeds    # (1, 768) L2-normalized
    cos_sim = (image_emb * text_emb).sum(dim=-1).item()  # scalar
    return max(0.0, 2.5 * cos_sim)
```

**CRUCIAL CODE — Frame extraction at timestamp:**

```python
import subprocess
from pathlib import Path
import tempfile

def extract_frame_at_time(video_path: Path, timestamp_sec: float) -> Image.Image:
    """Extract single frame at given timestamp using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        img = Image.open(tmp_path).convert("RGB")
        return img
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

**Group timeline reconstruction:**

`p4_assignments.json` has per-group sentences. Each sentence in `audio_manifest.json` has duration. Reconstruct group start/end in OUTPUT video timeline by accumulating audio durations in order. This is the timestamp to sample.

Pseudo-code:
```
cursor = 0.0
for group in p4_assignments:
    group_start = cursor
    for sentence_id in group.sentence_ids:
        cursor += audio_manifest[sentence_id].duration
    group_end = cursor
    group_mid = (group_start + group_end) / 2
    # Use group_mid as the sampling timestamp
```

**Verification:**
- Print first 3 group timestamps for review_1 and compare with output video duration (should sum to ~58.20s per handoff)
- Sample 1 frame from review_1 group 0 and save to `data/evaluation/debug_review_1_group_0.png` — visually inspect it looks like content (not black/garbage)
- Print computed CLIPScore for review_1 group 0 — expected range 0.5-2.5 (rescaled). If outside [0, 2.5], flag bug.

---

### M2: LLM-as-Judge — Visual quality

**Model:** `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (multimodal, already in pipeline).

**What it measures:** Visual coherence and quality of the output video, judged from sampled keyframes.

**Input to judge:**
- Per video: 6 evenly-spaced frames extracted from output video (covers ~10s intervals for 60s video)
- Original narration text (concatenated from `summary_script.json`)

**Output:** Rating 1-5 in 3 dimensions:

1. **Visual-narration coherence:** Do the frames match what the narration is describing?
2. **Temporal consistency:** Do consecutive frames look like part of a coherent video (no jarring transitions, identity preservation across frames)?
3. **Visual quality:** Are frames sharp, well-composed, free of obvious artifacts (warping, garbled text, distortion)?

**CRUCIAL CODE — Judge prompt template (DO NOT MODIFY):**

```python
JUDGE_VISUAL_SYSTEM = """You are an expert video evaluator. You will be shown 6 keyframes \
sampled from a generated summary video and given the narration script. Your job is to \
rate the video on 3 dimensions, each on a 1-5 scale where:

1 = Very poor (severe issues, unusable)
2 = Poor (notable issues)
3 = Acceptable (works but has flaws)
4 = Good (minor issues only)
5 = Excellent (no notable issues)

Be strict but fair. Do not inflate scores. Output ONLY valid JSON in this exact format:
{
  "visual_narration_coherence": <int 1-5>,
  "temporal_consistency": <int 1-5>,
  "visual_quality": <int 1-5>,
  "rationale": "<one sentence per dimension, separated by '; '>"
}
"""

JUDGE_VISUAL_USER_TEMPLATE = """Narration script:
\"\"\"
{narration}
\"\"\"

Rate the 6 keyframes shown above on the 3 dimensions defined in the system prompt. \
Output JSON only, no preamble."""
```

**Frame sampling:** 6 evenly-spaced frames. For 58s video: timestamps [4.83, 14.50, 24.17, 33.83, 43.50, 53.17]. Pass as image list to VL model.

**Parsing:** Use `json.loads()` with try/except. If parse fails, retry once with prompt prefix "Output VALID JSON only:". If still fails, log error + record as NaN, do not crash.

**Verification:**
- Run on review_1 only first. Inspect output JSON. Ratings must be int 1-5. Rationale must be non-empty string.
- Total VRAM during VL-7B inference: should be ~10-12 GB. If OOM, use `device_map="auto"` and offload.

---

### M3: LLM-as-Judge — Narrative quality

**Model:** `Qwen/Qwen2.5-14B-Instruct-AWQ` (text-only, already in pipeline as summarizer).

**What it measures:** Quality of the LLM summary as a piece of writing for a tech-review summary.

**IMPORTANT BIAS NOTE:** This model is the same one used to GENERATE the summary in Phase 2. Self-evaluation bias is real. We accept this limitation and disclose it in thesis bab 5. Reasoning: no comparable local 14B alternative, and the judge is text-only here (different task than generation, which had video context). Document explicitly.

**Input to judge:**
- Original transcript (full, from `data/intermediate/{video_id}/transcript.json`, text field concatenated)
- Generated summary script (from `data/intermediate/{video_id}/summary_script.json`, sentence texts concatenated)

**Output:** Rating 1-5 in 3 dimensions:

1. **Informativeness:** Does the summary cover the key points from the source?
2. **Coherence:** Does the summary read as a fluent, well-structured narrative?
3. **Faithfulness:** Does the summary avoid hallucinations or claims not supported by source?

**CRUCIAL CODE — Judge prompt template (DO NOT MODIFY):**

```python
JUDGE_NARRATIVE_SYSTEM = """You are an expert text evaluator for summarization quality. \
You will be given a source transcript and a generated summary script. Rate the summary \
on 3 dimensions, each on a 1-5 scale:

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Be strict. Output ONLY valid JSON:
{
  "informativeness": <int 1-5>,
  "coherence": <int 1-5>,
  "faithfulness": <int 1-5>,
  "rationale": "<one sentence per dimension, separated by '; '>"
}
"""

JUDGE_NARRATIVE_USER_TEMPLATE = """Source transcript:
\"\"\"
{transcript}
\"\"\"

Generated summary script:
\"\"\"
{summary}
\"\"\"

Rate the summary. Output JSON only, no preamble."""
```

**Truncation:** Transcripts can be long (8-17 min videos → ~1500-3500 words). Qwen2.5-14B context is 32K tokens, no truncation needed. But verify token count before sending; if >28K, truncate transcript middle (keep first 60% + last 40%) and log a warning.

**Parsing:** Same as M2.

---

### M4: Summarization fidelity (ROUGE + BERTScore)

**What it measures:** Lexical (ROUGE) and semantic (BERTScore) overlap between source transcript and generated summary.

**Comparison:** Source transcript text (reference) vs Generated summary text (hypothesis).

**Metrics:**
- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1
- BERTScore F1 (use `roberta-large` as the underlying model — standard for English)

**Libraries:**
- `rouge-score` (install: `pip install rouge-score`)
- `bert-score` (install: `pip install bert-score`)

**CRUCIAL CODE — Computation:**

```python
from rouge_score import rouge_scorer
from bert_score import score as bertscore_fn

def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Returns ROUGE-1/2/L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure,
    }

def compute_bertscore(reference: str, hypothesis: str) -> float:
    """Returns BERTScore F1 (roberta-large)."""
    P, R, F1 = bertscore_fn(
        cands=[hypothesis],
        refs=[reference],
        model_type="roberta-large",
        lang="en",
        verbose=False,
        device="cuda",
    )
    return F1.item()
```

**Verification:**
- ROUGE scores must be in [0, 1]. Typical values for abstractive summarization: ROUGE-1 ~0.3-0.5, ROUGE-2 ~0.1-0.25, ROUGE-L ~0.25-0.4.
- BERTScore typically ~0.85-0.95 for English (because RoBERTa is pretrained, baseline is high).
- If a score is exactly 0 or exactly 1, suspect bug — log and investigate.

---

## 4. Execution order (orchestrator: `run_evaluation.py`)

Sequential to avoid VRAM contention:

1. **M4 first** (no GPU needed beyond BERTScore, lightest) — gets baseline summary stats out fast
2. **M1 CLIPScore** (CLIP ViT-L/14, ~2 GB VRAM)
3. **M3 narrative judge** (Qwen2.5-14B-AWQ, ~10 GB VRAM)
4. **M2 visual judge** (Qwen2.5-VL-7B-AWQ, ~10 GB VRAM)

Between M3 and M4, **explicitly unload the previous model and call `torch.cuda.empty_cache()`** to avoid OOM. Use the existing `VRAMManager` in `src/utils/vram.py` if it provides this; otherwise inline `del model; torch.cuda.empty_cache()`.

**Each metric MUST write its CSV before next metric starts.** This way, a crash in M2 doesn't lose M4/M1/M3 results.

---

## 5. Summary report (`summary_report.md`)

After all 4 metrics finish, generate a markdown report with:

1. Dataset summary: list 10 video_ids, source domain, total duration sum
2. **Table 1 — Per-metric dataset-level results** (mean ± std across 10 videos):

| Metric | Value | Notes |
|---|---|---|
| CLIPScore (M1) | x.xx ± x.xx | Visual-text alignment per group |
| LLM-Judge Visual: coherence (M2.1) | x.xx ± x.xx | 1-5 scale |
| LLM-Judge Visual: temporal (M2.2) | x.xx ± x.xx | 1-5 scale |
| LLM-Judge Visual: quality (M2.3) | x.xx ± x.xx | 1-5 scale |
| LLM-Judge Narrative: informativeness (M3.1) | x.xx ± x.xx | 1-5 scale |
| LLM-Judge Narrative: coherence (M3.2) | x.xx ± x.xx | 1-5 scale |
| LLM-Judge Narrative: faithfulness (M3.3) | x.xx ± x.xx | 1-5 scale |
| ROUGE-1 F1 (M4) | x.xx ± x.xx | |
| ROUGE-2 F1 (M4) | x.xx ± x.xx | |
| ROUGE-L F1 (M4) | x.xx ± x.xx | |
| BERTScore F1 (M4) | x.xx ± x.xx | |

3. **Table 2 — Per-video breakdown** (one row per video, all metrics as columns)
4. Notes section: any anomalies, failures, OOMs, retried parses. Be honest.

**DO NOT** add interpretation, conclusions, or claims of quality. Just data. Interpretation is user's job, not yours.

---

## 6. Configuration

Add to `configs/default.yaml`:

```yaml
evaluation:
  clipscore:
    model: "openai/clip-vit-large-patch14"
    rescale_factor: 2.5
  judge_visual:
    model: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    num_keyframes: 6
    max_retries: 1
  judge_narrative:
    model: "Qwen/Qwen2.5-14B-Instruct-AWQ"
    max_transcript_tokens: 28000
    max_retries: 1
  summary_fidelity:
    bertscore_model: "roberta-large"
    use_stemmer: true
```

---

## 7. Failure handling

For each video, each metric:
- If load/inference fails: log error to `data/evaluation/errors.log` with `video_id, metric_id, error_msg, traceback`
- Record metric value as `NaN` in CSV
- Continue to next video, do not crash whole run

At end of run, print summary: "X/10 videos succeeded for M1, Y/10 for M2, ..."

**DO NOT silently skip failures.** Every failure must appear in `errors.log`. Every NaN must have a corresponding log entry.

---

## 8. Verification steps (BEFORE declaring done)

Run all of these and report output verbatim:

1. **File existence check:**
   ```bash
   ls -la data/evaluation/
   ```
   All 5 CSVs + summary_report.md + (possibly) errors.log must exist.

2. **CSV sanity:**
   ```bash
   head -3 data/evaluation/m1_clipscore_per_video.csv
   head -3 data/evaluation/m4_summary_fidelity.csv
   wc -l data/evaluation/*.csv
   ```
   Each per-video CSV should have 11 lines (header + 10 videos). M1 per-group CSV will have more.

3. **Range checks** (write as `src/eval/verify.py`, run it):
   - CLIPScore values: all in [0, 2.5]
   - LLM-Judge scores: all int in [1, 5]
   - ROUGE values: all in [0, 1]
   - BERTScore values: all in [0, 1]
   - Count NaN values per column, report

4. **Spot-check one judge output:** print rationale text from review_1 in both M2 and M3 CSVs. Must be coherent English sentences, not gibberish or empty.

5. **Compare to handoff baseline:** review_1 was noted as "user visual review: hampir gk bisa bedain generated vs retrieved". Expect CLIPScore for review_1 to be respectable (not bottom of dataset). Just note value, don't claim quality.

**You are NOT done until all 5 verification checks above pass and are reported in your handoff message.**

---

## 9. Anti-hallucination reminders

Read this section TWICE before starting.

1. **Do not claim "evaluation results look good" or similar quality judgments.** You produce numbers. The user judges quality. If you write a sentence like "the model performs well" anywhere in your handoff or in `summary_report.md`, it's a violation of this brief.

2. **Quote actual values, not summarized impressions.** Wrong: "CLIPScore is in expected range." Right: "CLIPScore mean = 0.81, std = 0.12, min = 0.43, max = 1.21."

3. **If a file does not exist, write "NOT FOUND" explicitly.** Do not invent paths. Do not assume `p4_assignments.json` has a field — `view` it first.

4. **Do not add features not in this brief.** No "I also added per-sentence CLIPScore for completeness." If you think something is missing, flag it in handoff, do not implement.

5. **VRAM management is critical.** Between metric models, verify VRAM is freed (`nvidia-smi` in a subprocess, log baseline + post-unload). If VRAM doesn't go back to ~1 GB, the model is still loaded — fix before proceeding.

6. **Ollama is paused via SIGSTOP during LTX runs in this project (per handoff). For evaluation, Ollama is NOT in use** — none of these metrics call Ollama. But still verify nothing else is hogging VRAM at start: `nvidia-smi`, report initial state.

7. **If you find existing `src/metrics.py` or `src/llm_judge.py` and feel tempted to extend them: DO NOT.** Re-read Section 2.

8. **JSON parse failures from judge models: retry once, then NaN.** Do not retry 5 times. Do not silently skip. Log every retry attempt.

9. **The user can detect hallucinations.** If you didn't run a check, say so. If a result surprises you, surface it, don't hide it.

---

## 10. Estimated wallclock

- M1 (CLIPScore, ~7 groups × 10 videos = 70 frames): ~5 min
- M2 (Visual judge, 10 videos × 6 frames): ~15 min on VL-7B
- M3 (Narrative judge, 10 videos × text): ~10 min on 14B
- M4 (ROUGE + BERTScore, 10 videos): ~3 min
- Model load/unload overhead: ~5 min total

**Total: ~40 min wallclock.** If your run takes >2 hours, something is wrong — stop and report.

---

## 11. Handoff format (when done)

Write a message to user with this exact structure:

```
## Evaluation run complete

**Run duration:** <wallclock>
**Status per metric:** M1 X/10, M2 X/10, M3 X/10, M4 X/10

**Files created:**
- <list each file with size>

**Verification results:**
1. File existence: <PASS/FAIL>
2. CSV sanity: <verbatim head output>
3. Range checks: <PASS/FAIL per metric, NaN counts>
4. Spot-check judge rationale: <verbatim text>
5. review_1 CLIPScore: <value>

**Anomalies / failures:**
<list each, or "None">

**NOT done:**
<anything in this brief you didn't do, with reason>
```

NO summary of "what evaluation shows." NO claims about quality. Just status + numbers + verification.

---

End of brief. Acknowledge receipt by listing the 4 metric IDs and confirming you will NOT extend `src/metrics.py` or `src/llm_judge.py`.

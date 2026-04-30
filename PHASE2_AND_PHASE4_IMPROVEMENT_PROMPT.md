# TASK: Improve Phase 2 Prompt + Add Temporal-Guided Retrieval to Phase 4

## PART 1: Fix Phase 2 LLM Prompt

### Problem
The current LLM summarizer sometimes:
- Summarizes vocabulary explanations instead of actual content (e.g., "The word 'hinder' means...")
- Generates duplicate sentences covering the same topic
- Produces conceptual keywords ("brain health") instead of visual keywords ("brain scan diagram")
- Doesn't clearly instruct how to produce accurate source_timestamp_hint values

### What to Change in `src/phase2_summarize.py`

#### Change 1: Update SYSTEM_PROMPT

Replace the current SYSTEM_PROMPT with this improved version:

```python
SYSTEM_PROMPT = """You are a master scriptwriter. Your task is to output a single, valid JSON object containing a summarized video script.
STYLE: {style_description}

RULES:
1. JSON ONLY: No markdown formatting, no conversational filler.
2. HOOK: Start with a power statement. Never say "Starting video" or "In this clip".
3. TONE: {style_description}
4. SPELL NUMBERS: Say "seven minutes" not "7m".
5. LENGTH: Target {target_duration} seconds.
6. CONTENT ONLY: Summarize the MAIN TOPIC and FACTUAL CLAIMS of the video. Do NOT summarize vocabulary definitions, word explanations, pronunciation guides, or language teaching segments. Focus on WHAT the video is about, not HOW it teaches.
7. NO REPETITION: Each sentence must cover a DIFFERENT aspect or subtopic. Never rephrase the same point in multiple sentences.
8. VISUAL KEYWORDS: The "keywords" field must contain VISUAL descriptions of what might appear on screen during that part of the video. Think: what would a viewer SEE? Use concrete nouns (e.g., "bar chart", "person running", "close-up of chip") not abstract concepts (e.g., "performance", "health", "innovation").
9. TIMESTAMP ACCURACY: The "source_timestamp_hint" must match the approximate time range in the transcript where the information for that sentence originally appears. Use the timestamps provided in the transcript. This is critical for visual matching.

SCHEMA:
{schema_json}"""
```

#### Change 2: Update FEW_SHOT_EXAMPLE

Replace with a better example that demonstrates visual keywords and diverse sentences:

```python
FEW_SHOT_EXAMPLE = """[Example input/output for format reference only]
Input: [00:15] Sleep quality is vital for brain function. [00:22] Most people need eight hours but quality wins. [01:05] A new study from Harvard shows napping can reduce cortisol levels. [01:30] The word 'rejuvenate' means to restore energy. [02:10] Brain scans revealed that nappers had larger hippocampal volume.
Output:
{
  "video_id": "demo",
  "target_duration": 90,
  "style": "informative",
  "backend_used": "local",
  "sentences": [
    {
      "id": 0,
      "text": "Quality rest beats duration every time, and new research shows that how you sleep matters more than how long.",
      "estimated_duration_seconds": 6.5,
      "source_timestamp_hint": [15.0, 22.0],
      "keywords": ["person sleeping in bed", "alarm clock", "sleep quality infographic"]
    },
    {
      "id": 1,
      "text": "A Harvard study found that short naps can significantly lower stress hormones in the body.",
      "estimated_duration_seconds": 6.0,
      "source_timestamp_hint": [65.0, 90.0],
      "keywords": ["Harvard university logo", "scientist in lab", "cortisol chart"]
    },
    {
      "id": 2,
      "text": "Brain scans of regular nappers showed a noticeably larger hippocampus, the region linked to memory.",
      "estimated_duration_seconds": 6.5,
      "source_timestamp_hint": [130.0, 150.0],
      "keywords": ["MRI brain scan", "hippocampus diagram", "medical imaging screen"]
    }
  ]
}
Note: The input segment about the word 'rejuvenate' was intentionally excluded because it is a vocabulary explanation, not factual content about the topic."""
```

#### Change 3: No code logic changes needed

The `_chunk_transcript`, `_generate_with_retry`, and `run` methods are fine as-is. Only the prompt strings need updating.

---

## PART 2: Add Temporal-Guided Retrieval to Phase 4

### Problem
Currently Phase 4 matches each narration sentence to ALL keyframes purely by semantic similarity (cosine). This means:
- A sentence about "chip performance" might match a generic MacBook close-up instead of the benchmark screenshot that was shown at that moment in the video
- The source_timestamp_hint from Phase 2 is generated but NEVER USED in Phase 4

### Concept
We already know approximately WHEN in the original video each summary sentence's content appeared (via source_timestamp_hint). Keyframes also have timestamps from PySceneDetect. We should use this temporal information to BOOST keyframes that are temporally close to where the content originally appeared.

```
final_score = alpha * semantic_score + (1 - alpha) * temporal_score
```

Where temporal_score is high when the keyframe timestamp is close to the source_timestamp_hint, and low when far away.

### Implementation

#### Step 1: Add temporal scoring function

In `src/phase4_retrieve.py` (or wherever retrieval logic lives), add:

```python
import math
import numpy as np

def compute_temporal_scores(sentence_timestamp_hint, keyframe_timestamps, sigma=30.0):
    """
    Compute temporal proximity scores between a sentence's source timestamp
    and all keyframe timestamps.
    
    Args:
        sentence_timestamp_hint: [start, end] from source_timestamp_hint
        keyframe_timestamps: list of floats, midpoint timestamp of each keyframe's scene
        sigma: controls how quickly score decays with distance (in seconds).
               30.0 means keyframes within ~30 seconds get high scores,
               beyond that drops off quickly.
    
    Returns:
        numpy array of scores in [0, 1], one per keyframe
    """
    if sentence_timestamp_hint is None or len(sentence_timestamp_hint) < 2:
        # No timestamp info available, return uniform scores (no temporal bias)
        return np.ones(len(keyframe_timestamps)) / len(keyframe_timestamps)
    
    mid = (sentence_timestamp_hint[0] + sentence_timestamp_hint[1]) / 2.0
    scores = np.array([
        math.exp(-((kf_ts - mid) ** 2) / (2 * sigma ** 2))
        for kf_ts in keyframe_timestamps
    ])
    return scores
```

#### Step 2: Integrate into each retrieval arm

For Arm B (caption + cosine) and Arm C (SigLIP), modify the matching step:

```python
# BEFORE (current):
best_idx = np.argmax(semantic_scores)

# AFTER (temporal-guided):
temporal_scores = compute_temporal_scores(
    sentence.source_timestamp_hint,
    [kf.timestamp_mid for kf in keyframes],
    sigma=config.temporal_sigma  # default 30.0
)

# Normalize both to [0, 1]
semantic_norm = min_max_normalize(semantic_scores)
temporal_norm = min_max_normalize(temporal_scores)

# Weighted combination
beta = config.temporal_weight  # e.g., 0.3
final_scores = (1 - beta) * semantic_norm + beta * temporal_norm

best_idx = np.argmax(final_scores)
```

#### Step 3: Add config parameters

In `configs/default.yaml`:

```yaml
retrieval:
  # Existing settings...
  
  # Temporal guidance (new)
  use_temporal_guidance: true    # set false to disable (for ablation comparison)
  temporal_weight: 0.3           # beta: 0 = pure semantic, 1 = pure temporal
  temporal_sigma: 30.0           # seconds, controls decay width
```

#### Step 4: Keep Arm A (random) unchanged

Random baseline should NOT use temporal guidance — it must stay random for the ablation to be valid.

#### Step 5: Make it toggleable for ablation

The ablation runner should be able to test:
- Arm A: Random (no change)
- Arm B: Caption cosine only (temporal off)
- Arm B+T: Caption cosine + temporal guidance (temporal on)
- Arm C: SigLIP only (temporal off)
- Arm C+T: SigLIP + temporal guidance (temporal on)

This way you can measure: "Does temporal guidance actually improve matching?"

In `run_ablation.py`, the arm configs could look like:

```python
ARMS = [
    {"name": "random", "type": "random", "temporal": False},
    {"name": "caption", "type": "caption", "temporal": False},
    {"name": "caption_temporal", "type": "caption", "temporal": True},
    {"name": "siglip", "type": "siglip", "temporal": False},
    {"name": "siglip_temporal", "type": "siglip", "temporal": True},
]
```

#### Step 6: Access source_timestamp_hint in Phase 4

Phase 4 needs to read the summary_script.json to get source_timestamp_hint for each sentence. It probably already reads the sentences for text — just also read the timestamp hints:

```python
# When loading summary script for Phase 4:
summary = load_json_as_model(summary_path, SummaryScript)

for sentence in summary.sentences:
    # sentence.text -> already used for embedding
    # sentence.source_timestamp_hint -> NEW: use for temporal scoring
    # sentence.keywords -> could also be used for matching (optional future improvement)
    pass
```

Make sure `SummarySentence` in `schemas.py` has `source_timestamp_hint` as a field (it should, based on the JSON output you showed).

---

## Summary of All Changes

| File | Change | Part |
|------|--------|------|
| src/phase2_summarize.py | Update SYSTEM_PROMPT (add rules 6-9) | Part 1 |
| src/phase2_summarize.py | Update FEW_SHOT_EXAMPLE (visual keywords, no vocab) | Part 1 |
| src/phase4_retrieve.py | Add compute_temporal_scores() function | Part 2 |
| src/phase4_retrieve.py | Integrate temporal scoring into Arm B and C matching | Part 2 |
| configs/default.yaml | Add temporal_weight, temporal_sigma, use_temporal_guidance | Part 2 |
| src/eval/run_ablation.py | Add temporal on/off variants for Arm B and C | Part 2 |

## Do NOT Change
- Phase 1, 3, 5 — unaffected
- Arm A (random) — must stay random
- SummaryScript schema — source_timestamp_hint already exists
- clean_for_tts() — still applied after LLM generation

## Success Criteria
- Re-run Phase 2 on a test video → check that output no longer contains vocabulary definitions
- Check that keywords are visual (nouns you can see) not conceptual
- Check that no two sentences cover the same topic
- Run Phase 4 with temporal guidance ON vs OFF → compare CLIPScore
- Temporal-guided arms should produce better visual-text alignment, especially for videos with visual variety (tech reviews, news segments)

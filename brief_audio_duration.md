# Brief: Audio Duration Distribution Per Group

**Task Type:** Execution (Data Measurement)
**Status:** Completed
**Goal:** Measure the distribution of TTS audio duration per group (Phase 4 output) to decide the clip length strategy for Phase 5. This data directly drives decisions on whether to use single-clip, multi-clip, looping, freeze, or playback-speed scaling strategies.

---

## 1. Executive Summary

Empirical measurement was conducted across all **90 sentence groups** spanning **10 review videos** (`review_1` to `review_10`). Sentence voiceovers were generated using the actual production Text-to-Speech (TTS) engine (**Kokoro v1.0 ONNX** @ 24kHz) to guarantee 100% measurement fidelity. 

The primary findings are highly critical for Phase 5:
* **0% of groups (0 out of 90) fit into a single clip of 2.56 seconds.** This applies to both `generate` and `retrieve` actions.
* **89.8% of generated groups require 3 or more clips** (>5.12s) to cover the duration of their spoken narration.
* **Extreme outliers** exist where groups exceed 25 seconds of narration (up to 38.4s), highlighting the necessity of robust temporal strategies (e.g., video freezing or looping) rather than simple linear padding.

---

## 2. Methodology & Hard Rules Verification

1. **WAV File Duration Measurement:** All sentence durations were measured directly from the generated audio files (WAV) using `soundfile.info()`, ensuring that no text-length heuristics or estimations were used.
2. **File Existence Validation:** All audio files were successfully generated and validated. No missing files or duration hallucinations occurred.
3. **Group Boundaries:** Group assignments and sentence-to-group mappings were extracted exactly from Phase 4's `p4_assignments.json` for each review video.

---

## 3. Results: Action = Generate (n = 59 Groups)

This distribution is the most crucial for Phase 5, as these are the groups where new video content must be synthesized (action = `generate`).

### Aggregate Statistics

| Statistic | Value |
|---|---|
| **Total Groups** | 59 |
| **Mean Duration** | 9.26s |
| **Median Duration** | 8.22s |
| **Standard Deviation** | 4.36s |
| **Minimum Duration** | 4.10s |
| **Maximum Duration** | 25.91s |
| **25th Percentile (p25)** | 6.62s |
| **75th Percentile (p75)** | 10.11s |
| **90th Percentile (p90)** | 15.85s |
| **95th Percentile (p95)** | 17.08s |

### Histogram Distribution

| Bucket | Count | Percentage | Clips Needed (@2.56s per clip) |
|---|---|---|---|
| **0 - 2.56s** | 0 | 0.0% | 1 clip |
| **2.56 - 5.12s** | 6 | 10.2% | 2 clips |
| **5.12 - 7.68s** | 20 | 33.9% | 3 clips |
| **7.68 - 10.24s** | 19 | 32.2% | 4 clips |
| **> 10.24s** | 14 | 23.7% | 5+ clips |

### Latency & Pacing Implications
* **Single Clip (≤2.56s):** **0.0%** of groups. A single-clip strategy will fail universally.
* **Double Clip (≤5.12s):** **10.2%** of groups. Stitching two clips together is sufficient for only a small tenth of the output.
* **Multi-Clip (3+ clips, >5.12s):** **89.8%** of groups. Almost nine out of ten generated groups require a multi-clip strategy. Sticking with 2.56s clips will mean orchestrating 3, 4, or even 5+ generated clips per narration segment.

---

## 4. Results: Action = Retrieve (n = 31 Groups)

For sanity check, the retrieval groups (where source video is matched) were also measured.

### Aggregate Statistics

| Statistic | Value |
|---|---|
| **Total Groups** | 31 |
| **Mean Duration** | 13.98s |
| **Median Duration** | 11.05s |
| **Standard Deviation** | 9.33s |
| **Minimum Duration** | 4.51s |
| **Maximum Duration** | 38.44s |
| **25th Percentile (p25)** | 6.66s |
| **75th Percentile (p75)** | 17.16s |
| **90th Percentile (p90)** | 27.13s |
| **95th Percentile (p95)** | 33.96s |

### Histogram Distribution

| Bucket | Count | Percentage |
|---|---|---|
| **0 - 2.56s** | 0 | 0.0% |
| **2.56 - 5.12s** | 1 | 3.2% |
| **5.12 - 7.68s** | 11 | 35.5% |
| **7.68 - 10.24s** | 2 | 6.5% |
| **> 10.24s** | 17 | 54.8% |

### Latency & Pacing Implications
* **Retrieve Alignment:** Retrieval segments are significantly longer than generated segments (mean of 13.98s vs 9.26s). Over 54% of retrieval segments exceed 10.24 seconds, requiring substantial source video reuse or padding strategies.

---

## 5. Extreme Outliers Analysis

Narration segments exceeding 20 seconds are rare edge cases that need specific architectural decisions. The table below lists all groups with `duration > 20s`:

| Video ID | Group ID | Action | N Sentences | Audio Duration (s) | Clips Needed |
|---|---|---|---|---|---|
| **review_8** | 0 | retrieve | 5 | 38.44s | 16 clips |
| **review_8** | 1 | retrieve | 5 | 36.97s | 15 clips |
| **review_9** | 2 | retrieve | 5 | 30.95s | 13 clips |
| **review_9** | 1 | retrieve | 5 | 27.13s | 11 clips |
| **review_1** | 5 | generate | 4 | 25.91s | 11 clips |
| **review_5** | 0 | retrieve | 4 | 25.82s | 11 clips |
| **review_5** | 9 | retrieve | 3 | 22.53s | 9 clips |
| **review_2** | 6 | retrieve | 2 | 20.62s | 9 clips |

> [!WARNING]
> **Outlier Challenge:** `review_1` Group 5 is the longest `generate` group, spanning **25.91 seconds**. Under the 2.56s model constraint, generating 11 distinct clips to cover this single group would result in extremely high latency (11 model forward passes) and visual fragmentation. For such long segments, utilizing frame freezing or looping strategies is highly recommended to limit VRAM usage and model generation overhead.

---

## 6. Project Integration

The complete measurement data has been exported to the root of the repository:
* **CSV File Path:** [audio_duration_per_group.csv](file:///home/wins053/Desktop/SumarizerAI-1-Gemini/video-summarizer/audio_duration_per_group.csv)
* **WAV Sentence Files:** Extracted and saved inside their respective `data/intermediate/review_*/audio/` directories.

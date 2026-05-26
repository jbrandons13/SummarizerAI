# Evaluation Summary Report v1

## 1. Dataset Summary
- **Total Videos:** 10
- **Source Domain:** technology reviews
- **Total Source Duration:** 7487.00 seconds (124.78 minutes)
- **Total Output Duration:** 595.97 seconds (9.93 minutes)
- **Evaluation Wallclock:** 240.35 seconds (4.01 minutes)

### Video Details:
- **review_1:** Samsung Galaxy Buds 4 Pro Review: Better than AirPods! (Marques Brownlee)
- **review_2:** Macbook Neo Review: Better than you Think! (Marques Brownlee)
- **review_3:** I bought a TV with NO 'Smart' Features... (Linus Tech Tips)
- **review_4:** This Video Keyboard Raised $3.8M on Kickstarter (Linus Tech Tips)
- **review_5:** The Samsung TriFold is AWESOME! (Dave2D)
- **review_6:** ROG XBOX Ally X Review (Dave2D)
- **review_7:** Xiaomi 17 Pro Max review - Apple are you seeing this!? (Mrwhosetheboss)
- **review_8:** Oppo Find N5 Review - Samsung just got Demolished!? (Mrwhosetheboss)
- **review_9:** Should You ACTUALLY Upgrade Your Laptop? - Lenovo Yoga 7a 2-in-1 (Austin Evans)
- **review_10:** This Windows Laptop BEATS the MacBook - ASUS Zenbook A16 (Austin Evans)

## 2. Table 1 — Per-Metric Dataset-Level Results
Results are reported as mean ± std across all 10 videos (excluding any failed runs/NaNs).

| Metric | Value | Notes |
|---|---|---|
| CLIPScore (M1) | 0.7822 ± 0.0482 | Visual-text alignment per group (rescaled [0, 2.5]) |
| LLM-Judge Visual: coherence (M2.1) | 3.5667 ± 0.2134 | 1-5 scale |
| LLM-Judge Visual: temporal (M2.2) | 3.6333 ± 0.3145 | 1-5 scale |
| LLM-Judge Visual: quality (M2.3) | 3.5000 ± 0.3727 | 1-5 scale |
| LLM-Judge Narrative: informativeness (M3.1) | 3.9000 ± 0.3000 | 1-5 scale |
| LLM-Judge Narrative: coherence (M3.2) | 4.7000 ± 0.4583 | 1-5 scale |
| LLM-Judge Narrative: faithfulness (M3.3) | 3.6000 ± 0.4899 | 1-5 scale |
| ROUGE-1 F1 (M4) | 0.0963 ± 0.0339 | Summarization overlap [0, 1] |
| ROUGE-2 F1 (M4) | 0.0415 ± 0.0196 | Summarization overlap [0, 1] |
| ROUGE-L F1 (M4) | 0.0645 ± 0.0224 | Summarization overlap [0, 1] |
| BERTScore F1 (M4) | 0.8361 ± 0.0091 | Semantic similarity [0, 1] (roberta-large) |

## 3. Table 2 — Per-Video Breakdown

| Video ID | CLIPScore (M1) | Visual Coh (M2.1) | Temp Cons (M2.2) | Vis Qual (M2.3) | Narr Info (M3.1) | Narr Coh (M3.2) | Narr Faith (M3.3) | ROUGE-1 (M4) | ROUGE-2 (M4) | ROUGE-L (M4) | BERTScore (M4) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| review_1 | 0.7899 | 3.3333333333333335 | 3.6666666666666665 | 3.6666666666666665 | 4 | 5 | 4 | 0.1491 | 0.0878 | 0.1081 | 0.8447 |
| review_2 | 0.8104 | 3.6666666666666665 | 3.6666666666666665 | 3.6666666666666665 | 4 | 5 | 4 | 0.0970 | 0.0326 | 0.0602 | 0.8322 |
| review_3 | 0.7407 | 3.3333333333333335 | 3.6666666666666665 | 3.0 | 4 | 4 | 3 | 0.0602 | 0.0199 | 0.0391 | 0.8324 |
| review_4 | 0.7300 | 3.6666666666666665 | 3.6666666666666665 | 3.6666666666666665 | 4 | 5 | 3 | 0.0531 | 0.0228 | 0.0398 | 0.8473 |
| review_5 | 0.8460 | 3.3333333333333335 | 3.3333333333333335 | 3.0 | 4 | 5 | 4 | 0.1381 | 0.0454 | 0.0819 | 0.8391 |
| review_6 | 0.8659 | 3.3333333333333335 | 3.6666666666666665 | 3.3333333333333335 | 3 | 4 | 3 | 0.0736 | 0.0349 | 0.0511 | 0.8465 |
| review_7 | 0.7378 | 3.6666666666666665 | 3.6666666666666665 | 3.6666666666666665 | 4 | 5 | 4 | 0.0645 | 0.0240 | 0.0459 | 0.8249 |
| review_8 | 0.7759 | 3.6666666666666665 | 3.6666666666666665 | 3.3333333333333335 | 4 | 4 | 3 | 0.1050 | 0.0546 | 0.0760 | 0.8448 |
| review_9 | 0.7174 | 3.6666666666666665 | 3.0 | 3.3333333333333335 | 4 | 5 | 4 | 0.1409 | 0.0568 | 0.0912 | 0.8240 |
| review_10 | 0.8078 | 4.0 | 4.333333333333333 | 4.333333333333333 | 4 | 5 | 4 | 0.0812 | 0.0360 | 0.0514 | 0.8248 |

## 4. Notes & Anomalies
No failures, OOMs, or anomalies were detected. All pipeline metrics executed successfully.

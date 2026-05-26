# Brief: LTX-Video I2V Prompt Refinement Sweep + SigLIP Scoring

**Task type:** Execution (smoke test + objective scoring), bukan research.
**Goal:** 
1. Verify apakah prompt engineering bisa fix masalah di input B (scene drift) dan C (rear screen misinterpretation)
2. Compute SigLIP scores (prompt-output + keyframe-output) untuk semua outputs, termasuk baseline input A, sebagai objective signal untuk decision-making

## Context

- Model: `Lightricks/LTX-Video-0.9.7-distilled` (LTX I2V mode, `LTXImageToVideoPipeline`)
- Setup sama dengan smoke test LTX sebelumnya: seed 42, 8 inference steps, custom timesteps, 768x512, `enable_model_cpu_offload()` (atau sequential offload kalau OOM di 241f, ikuti pola report sebelumnya)
- Conditioning frame untuk semua input: pakai file preprocessed yang sama dengan smoke test sebelumnya (`frame_ltx_preprocessed.jpg` 768x512)
- Monkeypatch `retrieve_timesteps` harus masih applied
- SigLIP encoder: `google/siglip2-so400m-patch16-naflex` (sama dengan Phase 4)

## Test matrix

### Input A — baseline reference (2 runs)

A udah stabil di smoke test sebelumnya, **ga ada variant baru**. Re-run original prompt cuma untuk dapet SigLIP score sebagai baseline reference.

**Original prompt:**
> "A Xiaomi smartphone with a custom gaming case attached, rear screen displaying game controller buttons that glow softly, slow camera pan around the device, tech product review style, soft studio lighting, clean dark background, shallow depth of field"

Lengths: 121f & 241f → 2 outputs.

### Input C (rear screen issue) — 4 variants × 2 lengths = 8 runs

**Original baseline (re-run untuk SigLIP scoring apple-to-apple):**
> "Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V1 — Spatial explicit:**
> "Small secondary display in the camera module area on the back of phone, showing music playback controls and album art, fingers gently tap the small display, rest of phone back is matte aluminum, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V2 — Negative implied:**
> "Phone back with camera bump containing a tiny circular display showing album art and music controls, main body of phone back is plain dark metal, no buttons, no full touchscreen, fingers tap the tiny display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V3 — Reference brand:**
> "Xiaomi 13 Ultra style mini rear display near camera lens showing music album art and playback controls, fingers tap the mini display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V4 — Minimal:**
> "Phone with mini display next to camera lens showing music playback, finger taps the mini display, smooth camera movement, warm lighting, dark background"

Wait — V4 is a fifth variant. Total variants for C = Original + V1 + V2 + V3 + V4 = 5 variants. Confirming intentional: yes, all 5 variants × 2 lengths = 10 runs for input C.

**Correction:** Input C = 5 variants × 2 lengths = **10 runs**.

### Input B (scene drift issue) — 3 variants × 2 lengths = 6 runs

**Original baseline (re-run untuk SigLIP scoring apple-to-apple):**
> "Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field"

**V1 — Static camera:**
> "Two identical phones placed motionless side by side on a clean surface, camera completely static, no zoom no pan, both phones remain unchanged, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field"

**V2 — Single subject focus:**
> "Two identical phones lying side by side on a clean surface, focus on the leftmost phone, both phones remain unchanged throughout, no scene transitions, camera slowly tilts down, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field"

Total input B: 3 variants × 2 lengths = **6 runs**.

### Total runs: 2 (A) + 10 (C) + 6 (B) = **18 runs**

Latency est: ~38s × 9 (121f) + ~67s × 9 (241f) = ~16 menit pure generation, plus scoring ~2 menit.

## SigLIP scoring methodology

After all videos generated, compute two scores per output:

### Score 1: Prompt-output similarity

1. Sample 4 frames uniformly dari output video (at t=0, 1/3·duration, 2/3·duration, end)
2. Encode each frame dengan SigLIP image encoder
3. Encode prompt text dengan SigLIP text encoder
4. Compute cosine similarity per frame, average → final `prompt_score`

### Score 2: Keyframe-output similarity

1. Same 4 frames sampled di atas
2. Encode keyframe (conditioning frame `frame_ltx_preprocessed.jpg`) dengan SigLIP image encoder
3. Compute cosine similarity per frame vs keyframe, average → final `keyframe_score`

Both scores: L2-normalize embeddings sebelum cosine sim (sama dengan Phase 4 convention).

## What to measure per run

1. Peak VRAM
2. Latency wallclock (generation only, exclude scoring)
3. Output mp4 path
4. **Visual notes** singkat per output:
   - **Input A:** masih stabil seperti smoke test sebelumnya? (sanity check)
   - **Input C:** Apakah rear small display direpresentasikan correctly? **Describe what's literally visible, jangan tafsir bebas.**
   - **Input B:** Apakah identity preserved sepanjang clip? Catat di detik berapa drift terjadi kalau ada.
5. SigLIP `prompt_score` dan `keyframe_score`

## Output paths

`~/smoke_tests/ltx_prompt_refine/{input_id}_{variant}_{frames}f.mp4`

Naming:
- `input_a_original_121f.mp4`, `input_a_original_241f.mp4`
- `input_c_original_121f.mp4`, `input_c_v1_121f.mp4`, ..., `input_c_v4_241f.mp4`
- `input_b_original_121f.mp4`, `input_b_v1_121f.mp4`, `input_b_v2_241f.mp4`

## Report format

Main table dengan kolom: `input_id`, `variant`, `num_frames`, `peak_vram_gb`, `latency_s`, `prompt_score`, `keyframe_score`, `visual_notes`, `output_path`.

Plus comparison summary:

### Input A baseline
- Report prompt_score & keyframe_score sebagai "reference level untuk good output"

### Input C analysis
- Variant ranking by prompt_score (descending)
- Variant ranking by keyframe_score (descending)
- **Crucial:** apakah variant dengan prompt_score tinggi juga visually correct (rear small display)? Atau prompt_score gagal capture issue ini?
- Best variant per length? Pattern interaction 121f vs 241f?

### Input B analysis
- Variant ranking by keyframe_score (relevan untuk scene drift)
- Apakah V1 (static) atau V2 (single focus) reduce drift lebih efektif?
- Pattern 121f vs 241f?

### SigLIP validity check
- Apakah ranking SigLIP align dengan visual judgment? Kalau ga align (misal visually best variant punya score rendah), report ini — implies SigLIP ga reliable untuk this domain dan kita harus pakai VLM/LLM-judge nanti.

**JANGAN kasih rekomendasi final variant** — user yang putuskan. Cukup report data + observasi neutral.

## Hard rules

- Conditioning frame, seed, dan setup identik dengan smoke test LTX sebelumnya (apple-to-apple)
- Pause Ollama sebelum run, restore after
- Kalau OOM di config tertentu, fallback ke sequential offload, log perubahan setting
- Verify pipeline class & monkeypatch `retrieve_timesteps` masih applied
- SigLIP scoring pakai model & convention yang sama dengan Phase 4 (`siglip2-so400m-patch16-naflex`, L2-normalize)

## Anti-hallucination

- Visual notes harus dari output mp4 actual, jangan generalisasi
- Quote angka VRAM & latency & scores dari measurement actual
- Kalau hasil variant identik dengan original (zero improvement), report apa adanya
- Kalau variant lebih buruk dari baseline (lower scores atau worse visual), report juga
- **Kalau prompt_score dan keyframe_score conflict (satu tinggi satu rendah), report keduanya — jangan pilih yang lebih flattering**

## Out of scope

- Jangan tweak inference steps, resolution, seed, atau model
- Jangan rekomendasi model lain
- Jangan generate variant tambahan di luar yang di-list
- Jangan implement VLM/LLM-judge scoring — itu defer ke production

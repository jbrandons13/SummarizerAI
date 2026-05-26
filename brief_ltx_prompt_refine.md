# Brief: LTX-Video I2V Prompt Refinement Sweep

**Task type:** Execution (smoke test), bukan research.
**Goal:** Verify apakah prompt engineering bisa fix dua masalah yang teridentifikasi di smoke test LTX sebelumnya:
1. **Input C:** "rear screen" diinterpretasi sebagai full-back touchscreen (harusnya small secondary display di camera module area)
2. **Input B:** Scene change / identity drift di tengah generation (terutama di 241f)

Hasil ini menentukan apakah masalah-nya prompt issue (fixable) atau training data limitation (must accept).

## Context

- Model: `Lightricks/LTX-Video-0.9.7-distilled` (LTX I2V mode, `LTXImageToVideoPipeline`)
- Pakai setup yang sama dengan smoke test LTX sebelumnya: seed 42, 8 inference steps, custom timesteps, 768x512, `enable_model_cpu_offload()` untuk 121f & 241f (kecuali 241f butuh sequential offload, ikuti pola report sebelumnya)
- Conditioning frame untuk input_b & input_c: pakai file preprocessed yang sama dengan smoke test sebelumnya (`frame_ltx_preprocessed.jpg` 768x512)
- Input A skip — udah stabil di kedua length, ga perlu retest

## Test matrix

### Input C (rear screen issue) — 4 variants × 2 lengths = 8 runs

**Original baseline (sudah ada di smoke test sebelumnya, tinggal reference):**
> "Close-up of a smartphone's rear screen lighting up to display music playback controls with album art, fingers gently tap the screen, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V1 — Spatial explicit:**
> "Small secondary display in the camera module area on the back of phone, showing music playback controls and album art, fingers gently tap the small display, rest of phone back is matte aluminum, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V2 — Negative implied:**
> "Phone back with camera bump containing a tiny circular display showing album art and music controls, main body of phone back is plain dark metal, no buttons, no full touchscreen, fingers tap the tiny display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V3 — Reference brand:**
> "Xiaomi 13 Ultra style mini rear display near camera lens showing music album art and playback controls, fingers tap the mini display, smooth camera dolly forward, tech product review style, warm rim lighting, dark studio background, cinematic shallow focus"

**V4 — Minimal:**
> "Phone with mini display next to camera lens showing music playback, finger taps the mini display, smooth camera movement, warm lighting, dark background"

Untuk masing-masing variant, generate di:
- 121f (~4s @ 30fps)
- 241f (~8s @ 30fps)

Total input C: 8 outputs.

### Input B (scene drift issue) — 2 variants × 2 lengths = 4 runs

**Original baseline (sudah ada di smoke test sebelumnya):**
> "Two modern smartphones lying side by side on a clean surface, camera slowly tilts down revealing their identical sleek metal frames and rounded edges, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field"

**V1 — Static camera:**
> "Two identical phones placed motionless side by side on a clean surface, camera completely static, no zoom no pan, both phones remain unchanged, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field"

**V2 — Single subject focus:**
> "Two identical phones lying side by side on a clean surface, focus on the leftmost phone, both phones remain unchanged throughout, no scene transitions, camera slowly tilts down, premium tech review aesthetic, soft diffused lighting, neutral background, shallow depth of field"

Untuk masing-masing variant, generate di:
- 121f
- 241f

Total input B: 4 outputs.

### Total: 12 runs

## What to measure per run

1. Peak VRAM
2. Latency wallclock
3. Output mp4 path
4. **Visual notes** singkat per output:
   - **Untuk input C:** Apakah rear small display direpresentasikan correctly (small element di camera area), atau tetep jadi full-back touchscreen / generic phone back? **JANGAN tafsir bebas — describe what's literally visible.**
   - **Untuk input B:** Apakah identity preserved sepanjang clip (kedua phone tetep sama dari awal sampai akhir), atau ada scene change / morph / object swap? Catat di detik berapa drift terjadi kalau ada.

## Output paths

`~/smoke_tests/ltx_prompt_refine/{input_id}_{variant}_{frames}f.mp4`

Contoh:
- `~/smoke_tests/ltx_prompt_refine/input_c_v1_121f.mp4`
- `~/smoke_tests/ltx_prompt_refine/input_c_v1_241f.mp4`
- `~/smoke_tests/ltx_prompt_refine/input_b_v1_121f.mp4`

## Report format

Markdown table dengan kolom: `input_id`, `variant`, `num_frames`, `peak_vram_gb`, `latency_s`, `visual_notes`, `output_path`.

Plus comparison summary:
- **Input C analysis:** Variant mana (kalau ada) yang berhasil interpret "small rear display" correctly? Apakah ada improvement vs original baseline? Atau semua variant tetap miss?
- **Input B analysis:** Variant mana (kalau ada) yang reduce scene drift? Apakah V1 (static camera) atau V2 (single focus) lebih efektif?
- **Length interaction:** Apakah ada pattern — 121f lebih stabil di variant tertentu vs 241f?

**JANGAN kasih rekomendasi final** — user yang putuskan. Cukup report data + observasi neutral.

## Hard rules

- Conditioning frame & seed identik dengan smoke test sebelumnya (apple-to-apple)
- Pause Ollama sebelum run, restore after
- Kalau OOM di salah satu config, fallback ke sequential offload (sesuai pola smoke test sebelumnya), log perubahan setting
- Verify pipeline class name & monkeypatch `retrieve_timesteps` masih applied (ini critical, tanpa patch crash)

## Anti-hallucination

- Visual notes harus dari output mp4 actual, jangan generalisasi
- Quote angka VRAM & latency dari measurement, bukan estimate
- Kalau hasil sama persis dengan smoke test sebelumnya (e.g. V1 hasilnya identik dengan original), report itu apa adanya — jangan cari beda yang ga ada
- Kalau ada variant yang **lebih buruk** dari baseline, report juga — jangan cuma highlight yang improve

## Out of scope

- Jangan tweak inference steps, resolution, seed, atau model
- Jangan coba input A (udah stabil)
- Jangan rekomendasi model lain
- Jangan generate variant tambahan di luar yang di-list

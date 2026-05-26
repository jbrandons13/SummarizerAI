# Brief: LTX-Video 2B Distilled smoke test — T2V vs I2V comparison

**Task type:** Execution (smoke test), bukan research.
**Goal:** Evaluate LTX-Video 2B distilled sebagai kandidat Phase 5 alternative dari HunyuanVideo 1.5. Bandingkan mode T2V (text-only) vs I2V (keyframe-conditioned) di hardware kita untuk decide arah Phase 5.

## Context

- Hunyuan 1.5 distilled udah terverifikasi max 41 frames @ 2.56s di RTX 3090 24GB. Audio duration distribution shows 0% groups fit single clip, mean 9.26s — strategi multi-clip mahal banget (~10 jam/run).
- Thesis judul: "Summarization-Driven Narrated Video Generation with Diffusion Transformers". LTX-Video adalah DiT, jadi tetep align sama judul.
- LTX 2B distilled (`Lightricks/LTX-Video-0.9.7-distilled`) attractive karena: 8-step inference, ~12GB VRAM, flexible length, native Diffusers support.
- Trade-off vs Hunyuan: 2B vs 8.3B parameter → quality kemungkinan drop.

## What to install/setup

1. Download model: `Lightricks/LTX-Video-0.9.7-distilled` (cek dulu disk space sebelum download)
2. Verify Diffusers version support LTX — minimum `diffusers>=0.32` for LTX-Video. Update kalau perlu.
3. Cache ke `~/models/ltx_video_distilled` (atau path serupa, konsisten dengan model cache lain)
4. **PENTING:** LTX distilled requires `guidance_scale=1.0` dan custom timesteps. Cek dokumentasi resmi:
   - HF page: https://huggingface.co/Lightricks/LTX-Video-0.9.7-distilled
   - Diffusers docs: https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video
   - Use custom timesteps `[1000, 993, 987, 981, 975, 909, 725, 0.03]` untuk base inference (per official docs)

## Test setup

Pakai input **identik dengan smoke test Hunyuan v2 sebelumnya** (apple-to-apple):
- 3 keyframes: `input_a_strong`, `input_b_marginal`, `input_c_generate`
- Prompts yang sama persis dengan smoke test Hunyuan v2 (jangan tweak)
- Seed 42 untuk reproducibility

### Mode 1: Pure T2V (no image conditioning)

Pakai `LTXPipeline` (text-to-video). Input: prompt aja, no image.

Generate clip untuk masing-masing 3 prompts:
- `num_frames=121` (~4s @ 30fps) — match dengan rough mean audio duration shorter groups
- Resolution: 768x512 atau resolusi closest yang divisible by 32 (LTX requirement) dan match aspect ratio Hunyuan test (720x480)
- Output: `~/smoke_tests/ltx/t2v_{input}_121f.mp4`

### Mode 2: I2V keyframe-conditioned

Pakai `LTXImageToVideoPipeline` atau `LTXConditionPipeline` (cek docs mana yang current). Input: prompt + keyframe image.

Generate clip untuk masing-masing 3 inputs dengan keyframe sebagai conditioning:
- `num_frames=121` (~4s @ 30fps)
- Resolution sama dengan Mode 1
- Output: `~/smoke_tests/ltx/i2v_{input}_121f.mp4`

### Stretch test (kalau Mode 1+2 PASS dan VRAM masih lega)

Coba `num_frames=241` (~8s @ 30fps) pada salah satu input untuk verify length scalability. Itu nge-test apakah LTX bisa generate clip yang match max audio duration tanpa multi-clip strategy.

## What to measure per run

1. Peak VRAM (`torch.cuda.max_memory_allocated()`)
2. Latency wallclock (exclude model load)
3. OOM? (kalau iya, log full traceback, STOP arm itu)
4. Output mp4 path
5. **Visual notes** singkat:
   - Identity preservation (untuk I2V): apakah subject di output match keyframe?
   - Prompt fidelity: apakah subject + setting match prompt?
   - Motion quality: smooth/jerky/static?
   - Artifacts: warping, fade, color shift, identity drift?

## Report format

Markdown table:

```
| Mode | Input | num_frames | duration_s | peak_vram_gb | latency_s | oom | visual_verdict | output_path |
```

Plus comparison summary:
- T2V mode: strengths/weaknesses
- I2V mode: strengths/weaknesses vs T2V
- Vs Hunyuan baseline (lihat report sebelumnya `phase5_smoke_outputs/hunyuan_v2/`): better/worse/comparable?
- **JANGAN kasih rekomendasi final** — user yang putuskan. Cukup report data + observasi.

## Hard rules

- Setup mirroring smoke test Hunyuan sebelumnya — same inputs, same prompts, same seed, same resolution target
- Verify HF repo path & model files exist sebelum claim
- Pause Ollama / orchestrator sebelum run (SIGSTOP)
- Restore environment after (SIGCONT)
- Kalau Mode 1 (T2V) OOM atau crash di num_frames=121, STOP dan report — jangan auto-degrade ke lower frames
- Kalau Mode 2 (I2V) gagal load karena pipeline class mismatch, log persis class name yang tersedia dan STOP

## Out of scope

- Jangan coba quantized variant (GGUF Q3/Q4) — itu separate question
- Jangan coba LTX 13B variant — itu butuh way more VRAM
- Jangan coba LTX-2 atau LTX-2.3 — keduanya bukan focus saat ini
- Jangan rekomendasi "pakai Hunyuan saja" atau "pakai Wan" — itu user decision, bukan agent

## Anti-hallucination

- Quote angka VRAM dari `torch.cuda.max_memory_allocated()`, bukan estimasi
- Quote latency dari wallclock actual
- Visual notes harus berdasarkan output file actual, jangan generalisasi dari smoke test orang lain
- Kalau ada konfigurasi yang ambiguous di docs (e.g. pipeline class name berubah antar Diffusers version), report ambiguity-nya — jangan pilih sembarang

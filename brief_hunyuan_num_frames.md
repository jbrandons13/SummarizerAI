# Brief: HunyuanVideo 1.5 num_frames sweep

**Task type:** Execution (smoke test), bukan research.
**Goal:** Cari tahu max viable `num_frames` di RTX 3090 22GB untuk HunyuanVideo 1.5 480p distilled I2V, dan ukur trade-off VRAM/latency/quality.

## Context

- Model sudah locked & cached: `~/models/hunyuanvideo_1.5_distilled` (repo `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled`)
- Smoke test sebelumnya pakai `num_frames=41` @ 16fps = 2.56s, peak VRAM 21.80 GB, 171s/clip
- Default Diffusers `num_frames=121` (~7.5s @ 16fps), Tencent paper claim 13.6 GB di 720p 121-frame dengan offloading+tiling (tapi itu di setup mereka, perlu verify di setup kita)
- Pipeline kita butuh clip yang match audio duration per group (range 3-10s+)

## Hypothesis to test

`num_frames` bisa dinaikin dari 41 → 61 / 81 / 121 untuk dapat clip lebih panjang. Trade-off: VRAM naik linear, latency naik linear, motion coherence bisa degrade.

## What to do

Jalankan 3 sweep run pakai **input yang sama persis** dengan smoke test sebelumnya (`input_a_strong`, atau pilih satu yang representative — minta user pick kalau ragu). Pakai prompt yang sama dengan smoke test 4. Variabel cuma `num_frames`:

| Run | num_frames | Duration @ 16fps |
|---|---|---|
| 1 | 61 | 3.81s |
| 2 | 81 | 5.06s |
| 3 | 121 | 7.56s |

Settings lain sama dengan smoke test sebelumnya (bfloat16, `enable_model_cpu_offload()`, 20 steps karena distilled, VAE tiling kalau sebelumnya udah enabled).

**PENTING:** Cek dulu apakah perlu enable `pipeline.vae.enable_tiling()` (paper bilang ini bagian dari setup 13.6 GB). Kalau sebelumnya belum dipakai dan VRAM tight, enable buat run yang lebih panjang.

## What to measure per run

1. **Peak VRAM** (pakai `torch.cuda.max_memory_allocated()` atau `nvidia-smi`)
2. **Latency** (wallclock end-to-end, exclude model load)
3. **OOM?** (kalau iya, stop di situ, log num_frames-nya)
4. **Output file** (mp4, simpen di `~/smoke_tests/num_frames_sweep/run_{N}.mp4`)

## What to report back

Markdown table dengan kolom: `num_frames`, `duration_s`, `peak_vram_gb`, `latency_s`, `latency_per_frame_ms`, `oom`, `output_path`. Plus catatan kalau ada anomali (motion degrade obvious, identity lost, dll — visual judgment singkat, jangan lebay).

## Hard rules

- Verify HF repo path & file existence sebelum claim apapun (history: Gemini pernah hallucinate paths)
- Kalau OOM di run 1 (num_frames=61), STOP dan report — jangan lanjut ke 81/121
- Kalau ada error yang ga obvious, log full traceback, jangan summarize
- Jangan ubah model, dtype, atau prompt — variabel tunggal cuma `num_frames`
- Ollama / proses GPU lain harus paused dulu sebelum run (SIGSTOP) — handoff v4 nyebut ini udah ada di proof-of-concept

## Out of scope

- Jangan coba model lain
- Jangan tweak inference steps
- Jangan rekomendasi strategi clip length — itu keputusan user setelah liat data ini

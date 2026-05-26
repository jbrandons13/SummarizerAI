# Brief: HunyuanVideo 1.5 VRAM optimization sweep

**Task type:** Execution (smoke test), bukan research.
**Goal:** Verify apakah 41 frames benar-benar ceiling di RTX 3090 22GB, atau bisa di-extend via tiling/offloading/resolution. Sweep sebelumnya OOM di 61 frames — sebelum vonis "maximum", ada beberapa optimization yang harus dicoba dulu.

## Context dari sweep sebelumnya

Hasil run #1 (num_frames=61) OOM:
- 22.62 GB allocated total, request 1.31 GB tambahan → fail
- VRAM headroom hampir habis sebelum attention step
- Output: `report_num_frames_sweep.md`, `results.json`

**Status yang BELUM jelas dari report sebelumnya:**
- Apakah `vae.enable_tiling()` udah enabled? Tencent paper claim 13.6 GB peak di 720p 121-frame justru dengan **kombinasi** offloading + tiling. Kalau tiling belum enabled, hasil 21.80 GB di 41 frames make sense (way above 13.6 GB).
- Apakah resolution udah 720x480 atau lebih tinggi?

Sebelum mulai sweep baru, **verify dulu** setup sebelumnya:
1. Cek script smoke test sebelumnya (kemungkinan di `~/smoke_tests/` atau working dir Phase 5)
2. Confirm: `enable_model_cpu_offload()` ✓ / `vae.enable_tiling()` ? / resolution ?
3. Report finding sebelum lanjut.

## What to test (urut prioritas — STOP kalau ada yang berhasil)

### Test A: Enable VAE tiling

Re-run `num_frames=61` dengan satu perubahan: tambah `pipeline.vae.enable_tiling()` setelah load model. Settings lain identik dengan smoke test sebelumnya (input_a_strong, prompt sama, seed 42, bfloat16, 20 steps, model cpu offload).

Kalau pass → coba `num_frames=81`. Kalau pass lagi → coba `num_frames=121`.
Kalau OOM di 61 → lanjut Test B.

### Test B: Lower resolution

Kalau Test A gagal di 61 frames, coba `num_frames=61` dengan resolution lebih rendah:
- 544x352 (atau aspect ratio terdekat yang divisible by 16)
- Tiling tetap enabled dari Test A

Kalau pass → coba 81, lalu 121 di resolusi yang sama.
Kalau masih OOM → lanjut Test C.

### Test C: Sequential CPU offload

Ganti `enable_model_cpu_offload()` dengan `enable_sequential_cpu_offload()`. Handoff sebelumnya mention ada meta-device bug — kalau bug masih ada, log error full dan STOP. Kalau bisa jalan, test `num_frames=61` di 720x480 dulu.

Trade-off: sequential offload **jauh lebih lambat** (bisa 3-5x), jadi cuma viable kalau quality di frame count yang lebih tinggi worth latency-nya.

## What to measure per run

Sama dengan sweep sebelumnya:
1. Peak VRAM (`torch.cuda.max_memory_allocated()`)
2. Latency wallclock (exclude model load)
3. OOM? (kalau iya, log full traceback)
4. Output mp4 di `~/smoke_tests/vram_optim/{test}_{num_frames}_{resolution}.mp4`
5. **Visual sanity check** singkat — apakah output masih reasonable (identity preserved, motion ada)? Tiling kadang bikin seam artifacts di VAE; resolution turun bisa bikin detail loss. Catat kalau ada degradation obvious.

## What to report back

Markdown table dengan kolom: `test_id`, `num_frames`, `resolution`, `tiling`, `offload_mode`, `peak_vram_gb`, `latency_s`, `oom`, `visual_notes`, `output_path`.

Plus: **rekomendasi konkrit** berdasarkan data — config mana yang ngasih clip terpanjang dengan quality acceptable dan latency wajar (<400s/clip). Kalau ga ada yang lolos, bilang aja "41 frames remains ceiling, alternatives: X/Y/Z".

## Hard rules

- STOP first time sebuah config PASS di num_frames tertentu, lanjut ke num_frames lebih tinggi. JANGAN test semua kombinasi exhaustively — waste GPU time.
- STOP entire sweep kalau Test A+B+C semua OOM di 61 frames.
- Verify setup sebelumnya BEFORE start (cek script, confirm tiling/resolution status). Jangan asumsi.
- Pause Ollama / orchestrator dulu sebelum run (sama dengan sweep sebelumnya).
- Restore environment after (SIGCONT orchestrator).
- Input file & prompt **identik** dengan sweep sebelumnya untuk apple-to-apple.

## Out of scope

- Jangan coba quantization (4-bit/8-bit) — itu separate decision, butuh discussion sama user dulu karena quality trade-off
- Jangan coba model lain
- Jangan rekomendasi "pakai Wan 2.2" — Wan udah fail smoke test sebelumnya (fade-to-brown), itu bukan opsi
- Jangan tweak inference steps (distilled model perlu 20 steps)

## Anti-hallucination reminders

- Verify file path & repo path exist sebelum claim
- Kalau setup sebelumnya ga bisa di-recover dari disk, **bilang aja** — jangan tebak
- Quote angka VRAM dari `torch.cuda.max_memory_allocated()`, bukan estimasi

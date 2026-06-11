# Brief untuk Gemini Agent — Predictive DACA (hilangkan weight sweep)

## Tujuan
DACA versi sekarang harus generate tiap shot di SELURUH weight-grid (sekitar 9x biaya
Phase 4) hanya untuk membaca kurva content preservation sebelum memilih w*. Brief ini
menguji dan (opsional) mendeploy sebuah controller yang MEMPREDIKSI w* per shot TANPA
sweep, memangkas biaya seleksi dari ~9x jadi ~2x (Tier A) atau ~1x (Tier B).

## Ide inti (yang membuat prediksi mungkin)
Untuk tiap shot, kurva content preservation c_s(w) turun dari 1.0 di w=0 menuju asimtot
yang kira-kira sama dengan d_s, namely similarity shot di w=0 terhadap reference kanonik.
Alasannya, di scale IP-Adapter tinggi output didominasi reference, sehingga I_s(w tinggi)
mendekati reference dan sim(I_s(w tinggi), I_s(0)) mendekati sim(reference, I_s(0)) = d_s.
Konsekuensinya, kurva c_s(w) praktis keluarga satu-parameter yang diindeks d_s, dan w*
bisa diprediksi dari d_s saja.

Notasi:
- d_s = sim_to_reference pada w=0 (DINOv2). Sudah ada di collapse_metrics.csv.
- c_s(w) = sim_to_own_w0 pada weight w (content preservation). Sudah ada di CSV.
- tau = 0.7 (TERKUNCI). w* = weight grid terbesar dengan c_s(w) >= tau.

---

## LANGKAH 1 — Validasi offline (TANPA generate apa pun, dari CSV yang sudah ada)
Ini tes yang menentukan. Tidak perlu GPU, cukup pandas/numpy atas CSV lama.

Input:
- `V{n}_{name}_collapse_metrics.csv` (per-shot per-weight: sim_to_reference, sim_to_own_w0).
- `V{n}_{name}_adaptive_anchor.csv` (per shot: w* DACA asli di tau=0.7).

Tugas:
1. Untuk tiap shot, ambil d_s = sim_to_reference di w=0, dan kurva c_s(w) penuh.
2. Konfirmasi asimtot: cek c_s di weight tertinggi mendekati d_s per shot. Laporkan
   gap per-shot dan korelasinya. Kalau lemah per-shot, laporkan apa adanya.
3. Fit predictor, dua framing yang setara, kerjakan keduanya dan bandingkan:
   - (a) Regresi langsung true-w* terhadap d_s lintas semua shot.
   - (b) Model c_s(w) = 1 - (1 - d_s) * g(w), fit g(w) low-dim yang di-pool lintas shot
     (mis. g(w) = w^k, atau monotone spline, g(0)=0 dan g(1)=1). Lalu turunkan
     predicted-w* = weight grid terbesar dengan predicted c_s(w) >= tau.
4. Evaluasi dengan leave-one-VIDEO-out cross-validation (fit di video lain, prediksi shot
   di video yang ditahan). JIKA V6 sampai V11 (run n=10) tersedia, tambahan: fit di
   V1 sampai V5, uji di V6 sampai V11 sebagai held-out test bersih.
5. Metrik wajib:
   - mean |predicted_w* - true_w*| (grid step).
   - **floor-breach rate**: untuk predicted-w*, baca c_s ASLI di weight itu dari CSV, hitung
     berapa shot yang c_s asli-nya < 0.7. Ini metrik terpenting, karena mengukur apakah
     deploy predicted-w* benar-benar menjaga floor.
   - mean content dan mean concept di predicted-w* versus di DACA asli.

Deliverable Langkah 1:
- `predictive_daca_offline.csv` (per shot: video, shot, d_s, true_w*, predicted_w*,
  content_at_pred, breach_flag).
- Ringkasan teks: kualitas fit, breach rate, mean |Δw*|, dan content/concept versus DACA.
- Plot w* versus d_s dengan garis fit-nya.

**Kriteria lolos** (gerbang ke Langkah 2): floor-breach rate rendah (idealnya mendekati 0)
dan mean |Δw*| kecil (dalam 1 grid step). Kalau TIDAK lolos, STOP, laporkan, JANGAN rerun.

---

## LANGKAH 2 — Konfirmasi end-to-end (HANYA jika Langkah 1 lolos, butuh generate)
Tujuan, membuktikan controller jalan di praktik dengan ~2x biaya, bukan 9x.

Tier A (utama), di 1 sampai 2 video saja (mis. Heart dan Sun):
- Untuk tiap shot: generate di w=0, hitung d_s = DINOv2(shot@0, reference), prediksi w*
  dari model yang sudah difit di Langkah 1, lalu generate sekali lagi di w* prediksi.
- Bandingkan dengan pilihan DACA asli yang sudah kamu punya. Laporkan apakah image dan
  content di predicted-w* cocok dengan true-w*, plus biaya aktual (generasi per shot).
- Cek floor: content di predicted-w* harus >= 0.7.

Tier B (opsional, agresif), uji proxy d_s tanpa generate sama sekali:
- Estimasi d_s dari CLIP(prompt_shot, image reference), prediksi w* dari proxy itu, lalu
  ukur seberapa baik proxy memprediksi w* asli. Ini menuju biaya ~1x (sweep hilang total).
  Laporkan akurasinya apa adanya, ini kemungkinan paling kasar.

Deliverable Langkah 2:
- Tabel per-shot predicted versus DACA + angka biaya (generasi per shot, ~2x versus ~9x).
- Contact sheet predicted-w* berdampingan dengan DACA untuk audit visual.
- (Tier B) tabel proxy-d_s versus d_s asli dan w* prediksi versus w* asli.

---

## Aturan konsistensi dan kejujuran
- Pengukuran (DINOv2, CLIP, cosine, preprocessing) dan pipeline (SDXL + cartoon LoRA +
  IP-Adapter, sampler/steps/resolusi/seed) HARUS identik dengan full-run, supaya angka
  sebanding dengan V1 sampai V11.
- Kalau regularitas asimtot atau fit-nya lemah per-shot, laporkan apa adanya. Hasil negatif
  juga temuan, dan dia membatasi seberapa jauh predictor bisa dipakai.
- tau=0.7 terkunci. predicted-w* harus on-grid.

## Sanity-check
- Semua cosine 0..1, c_s(0) = 1 tepat.
- d_s = sim_to_reference di w=0 harus persis dari CSV.
- Floor-breach dihitung dari c_s ASLI di weight terpilih, bukan dari c_s prediksi.

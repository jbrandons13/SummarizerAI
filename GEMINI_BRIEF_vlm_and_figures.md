# Brief untuk Gemini Agent — Reproduce Figures + VLM CSV

Konteks: thesis tentang DACA (divergence-adaptive concept anchoring) untuk video ringkasan edukasi. Kamu (Gemini) yang membuat semua chart/visual data. Sebelumnya kamu sudah memproduksi full-run figures + CSV (collapse_metrics, adaptive_anchor untuk V1-V5). Ini permintaan untuk me-reproduce sebagian figure dengan perubahan, plus 2 figure baru dan 1 CSV.

Video: V1 Geology, V2 Ecology, V3 Sun, V4 Heart, V5 iPhone (contrast control).

## Aturan global (berlaku ke semua output)
- Nama metrik di sumbu Y collapse curve: pakai "content preservation". (Sebelumnya tertulis "content kept" / "similarity to the shot's own original scene" — ganti jadi "content preservation".)
- Untuk perilaku high-w, pakai istilah "content homogenization", JANGAN "copy".
- Pertahankan gaya visual, warna, dan resolusi sama dengan output full-run sebelumnya.

---

## Deliverable 1 — Combined frontier 2x2
Satu figure berisi 4 panel collapse curve (Geology, Ecology, Sun, Heart) dalam grid 2x2.
- Tiap panel: 3 garis (similarity to reference naik, content preservation turun, inter-shot similarity naik) vs anchoring weight.
- Shared legend sekali saja, label panel (a)-(d), rentang sumbu konsisten antar panel bila memungkinkan.
- Sumbu Y: "content preservation" (dan label garis sesuai).
- Output: `combined_frontier_4videos.png`.

## Deliverable 2 — Combined adaptive 2x2
Satu figure berisi 4 panel fair-plane scatter (Geology, Ecology, Sun, Heart) dalam grid 2x2.
- Tiap panel: sumbu X = concept (CLIP-T), sumbu Y = content preservation; garis fixed-weight frontier + bintang adaptive (DACA) + garis floor tau=0.7.
- Shared legend, label panel (a)-(d).
- Output: `combined_adaptive_4videos.png`.

## Deliverable 3 — V5 contrast collapse curve (regenerate)
Reproduce collapse curve V5 iPhone, sama seperti sebelumnya TAPI sumbu Y diganti jadi "content preservation". Tetap figure terpisah (V5 adalah contrast control, tidak masuk 2x2).
- Output: `V5_iPhone_collapse_curve.png` (timpa yang lama).

## Deliverable 4 — Weight-sweep visual proof figure (FIGURE PALING PENTING)
Bukti visual langsung bahwa problem (reward collapse) memang ada. Belum ada. Tiap baris harus memuat KONTEKS + SWEEP penuh:
- Tampilkan gambar reference kanonik (concept image).
- Tampilkan teks prompt shot (sebagai label di samping/atas baris).
- Baris perbandingan: shot yang SAMA di-render pada w = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (langkah penuh 0.1).
- Pilih 1-3 shot representatif dari video dengan collapse paling jelas (rekomendasi Heart atau Geology). Kalau weight tertentu belum pernah di-render, render khusus untuk figure ini.
- Tujuan terlihat: di w=0 shot setia ke prompt-nya sendiri; makin besar w makin berubah jadi near-copy reference di w=1.0.
- Label tiap kolom dengan nilai w. PNG landscape, resolusi cukup untuk full-width.
- Output: mis. `V4_Heart_weight_sweep_proof.png`.

## Deliverable 5 — `vlm_results_aggregate.csv`
VLM-as-judge sudah dijalankan (V3/V4/V5, total 43 pasang). Butuh satu CSV agregat.

Schema (satu baris per video per kondisi):
```
video,condition,n_pairs,content_fidelity,same_concept_rate,content_homogenization_rate
```
- `condition`: `daca` atau `fixed_high_w` (sebutkan w mana yang dipakai).
- `content_fidelity` (0-1): apakah tiap shot setia menggambarkan konten per-shot yang dimaksud.
- `same_concept_rate` (0-1): fraksi pasang shot yang dinilai "scene/konsep sama" (indikator homogenisasi).
- `content_homogenization_rate` (0-1): fraksi pasang near-duplicate/collapsed. Istilah "homogenization", bukan "copy".
- `n_pairs`: jumlah pasang per baris (total semua = 43).

Sanity-check angka yang sudah diketahui:
- content_fidelity (daca): Heart ~0.92, Sun ~0.60, iPhone ~0.53 (DACA menang atas fixed_high_w).
- same_concept_rate: fixed_high_w ~1.0 vs daca ~0.44-0.67.
- content_homogenization_rate: ~0.

Output: `vlm_results_aggregate.csv`.

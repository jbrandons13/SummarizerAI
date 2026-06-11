# Brief untuk Gemini Agent — A3 External Consistency Baseline (StoryDiffusion)

Konteks: thesis tentang DACA (divergence-adaptive concept anchoring) untuk video ringkasan
edukasi. Klaim inti C3 = metode yang memaksimalkan consistency menghomogenkan konten
per-shot (similarity naik, content preservation turun). Sejauh ini bukti C3 datang dari satu
mekanisme saja, yaitu IP-Adapter scale (reference-injection). A3 ingin menguji apakah
mekanisme consistency yang BERBEDA, yaitu attention-sharing tanpa external reference, juga
collapse dengan cara yang sama. Kalau ya, itu konfirmasi C3 yang independen dari IP-Adapter.

PENTING — framing dan kejujuran (baca dulu):
- Ini BUKAN benchmark "kami menang". StoryDiffusion dirancang untuk subject/identity
  consistency, bukan untuk concept-with-variety pada materi edukasi. Tujuan kita bukan
  mempermalukannya, tapi menunjukkan bahwa paradigma consistency-maximizing memang
  mengorbankan content variety pada task ini.
- Jangan men-tune StoryDiffusion supaya terlihat buruk. Pakai setting consistency
  default/rekomendasi dari repo aslinya. Yang kita ukur adalah perilaku DEFAULT paradigma itu.
- Laporkan apa adanya. Kalau ternyata StoryDiffusion menjaga lebih banyak konten dari dugaan,
  itu juga temuan yang harus dilaporkan jujur, bukan disembunyikan.

Baseline: **StoryDiffusion** (Consistent Self-Attention), training-free, berbasis SDXL,
jalan di satu GPU. Kalau repo-nya tidak bisa dijalankan, fallback ke **ConsiStory** (juga
training-free attention-sharing) dengan protokol identik di bawah.

Video yang dipakai: **Sun (16 shot)** dan **Heart (13 shot)** sebagai dua video utama (shot
terbanyak, collapse paling jelas, keduanya video konsep edukasi). Opsional tambah **Geology
(3 shot)** sebagai video ketiga kalau sempat. JANGAN pakai iPhone/V5 (itu contrast control
subject-centric; di situ consistency justru pantas, jadi memasukkannya akan mengaburkan
klaim untuk konten edukasi).

---

## Aturan global (berlaku ke semua output)

1. **Model pengukuran HARUS identik dengan full-run.** Pakai DINOv2 dan CLIP yang SAMA
   persis dengan yang kamu pakai menghasilkan `*_collapse_metrics.csv` dan
   `*_adaptive_anchor.csv`. Pakai jalur kode pengukuran yang sama (definisi cosine,
   preprocessing image, dst), supaya semua angka berada di skala yang sama dan bisa
   ditumpuk di plane yang sudah ada.
2. **Prompt per-shot dan reference kanonik HARUS sama dengan full-run** untuk video yang
   dipilih. Jangan menulis prompt baru. Konten yang diminta harus task yang sama.
3. **Istilah metrik:** pakai "content preservation" dan "content homogenization" (jangan
   "copy"), konsisten dengan figure dan CSV sebelumnya.

---

## Protokol generasi (bagian paling penting)

StoryDiffusion tidak punya external reference dan tidak punya weight knob seperti IP-Adapter.
Maka kita bangun analог yang adil terhadap thesis. Untuk SETIAP video terpilih:

**Hasilkan dua set gambar dari prompt per-shot yang sama:**

- **VANILLA** = base SDXL, tiap shot di-generate INDEPENDEN (Consistent Self-Attention OFF).
  Ini anchor "konten milik shot sendiri" dan merupakan analog dari w=0 di thesis.
- **CONSISTENT** = StoryDiffusion dengan Consistent Self-Attention ON, shot-shot satu video
  di-generate sebagai satu batch (mode konsisten bawaan repo).

**Kontrol yang wajib:**
- **Seed per-shot SAMA** antara VANILLA dan shot yang bersesuaian di set CONSISTENT, supaya
  selisih antar keduanya mengisolasi efek mekanisme consistency (persis seperti content
  preservation di thesis mengisolasi kekuatan anchoring). Kalau parity seed tidak bisa di
  mode batch StoryDiffusion, tetap jalan tapi catat bahwa content preservation jadi memuat
  sedikit noise seed (masih bisa ditafsir secara agregat).
- **Checkpoint SDXL sama** dengan pipeline thesis. Idealnya pakai juga cartoon LoRA yang sama
  supaya gaya visual sebanding. Kalau LoRA tidak bisa dikomposisikan dengan StoryDiffusion,
  pakai base SDXL dan catat perbedaan gaya ini sebagai caveat (metrik content
  preservation, inter-shot, CLIP-T tetap valid karena bersifat relatif di dalam tiap metode).
- **Resolusi, sampler, jumlah langkah** sebisa mungkin sama dengan full-run. Catat kalau beda.

---

## Metrik yang diukur (semua dengan model yang sama seperti aturan global #1)

Per shot di set CONSISTENT:
- `content_preservation` = DINOv2 cosine( shot_consistent , shot_vanilla_prompt_sama ).
  Analog langsung dari `sim_to_own_w0` di thesis. Rentang harus 0..1.
- `concept_clip_t` = CLIP( shot_consistent , teks konsep ). Sama persis dengan kolom concept
  di `*_adaptive_anchor.csv`.

Per video (agregat):
- `inter_shot_sim_consistent` = rata-rata pairwise DINOv2 cosine antar shot di set CONSISTENT.
  Ini sinyal homogenisasi.
- `inter_shot_sim_vanilla` = rata-rata pairwise DINOv2 cosine antar shot di set VANILLA.
  Ini baseline. SELISIH (consistent minus vanilla) menunjukkan homogenisasi yang diinduksi.
- `mean_content_preservation`, `mean_concept_clip_t` (rata-rata kolom per-shot di atas).
- (OPSIONAL, hanya kalau murah) `sim_to_reference` = DINOv2 cosine( shot_consistent ,
  reference kanonik yang sama dengan thesis ). Catat bahwa StoryDiffusion tidak
  mengoptimalkan ini, jadi nilainya hanya untuk menempatkannya di sumbu yang sama.

---

## Yang kita harapkan (untuk orientasi saja, JANGAN dipaksakan)

Dugaan: set CONSISTENT punya `inter_shot_sim` jauh lebih tinggi dari VANILLA, dan
`content_preservation` rendah (shot ditarik ke penampilan bersama sehingga konten masing-masing
hilang). Di plane (concept CLIP-T, content preservation), titik StoryDiffusion diperkirakan
jatuh di wilayah content-rendah yang sama dengan fixed IP-Adapter ber-w tinggi, yaitu di atas
atau di bawah collapse frontier. Itulah konfirmasi C3 lintas-paradigma.

Tapi tetap laporkan angka apa adanya. Kalau hasilnya beda dari dugaan, tulis apa adanya.

---

## Deliverable

### Deliverable 1 — CSV metrik StoryDiffusion
Satu file `storydiffusion_baseline_metrics.csv`.

Bagian per-shot (satu baris per shot per video):
```
video,shot,concept_clip_t_consistent,content_preservation_consistent
```
Bagian agregat (satu baris per video, dipisah blank line, gaya sama seperti collapse_metrics):
```
video,mean_concept_clip_t,mean_content_preservation,inter_shot_sim_consistent,inter_shot_sim_vanilla
```

### Deliverable 2 — Koordinat untuk overlay di fair-plane
Untuk tiap video, satu titik StoryDiffusion = ( mean_concept_clip_t , mean_content_preservation ).
Sertakan ini eksplisit (boleh di akhir CSV atau file teks pendek) supaya bisa ditumpuk sebagai
satu marker tambahan di panel fair-plane yang sudah ada (Sun dan Heart).

### Deliverable 3 — Gambar untuk visual audit
Simpan gambar yang dihasilkan (set CONSISTENT dan set VANILLA) untuk tiap video, minimal satu
contact sheet per video yang menaruh shot CONSISTENT berdampingan dengan shot VANILLA dari
prompt yang sama, supaya homogenisasi terlihat mata.
- Output: mis. `storydiffusion_Sun_contactsheet.png`, `storydiffusion_Heart_contactsheet.png`.

### Deliverable 4 (OPSIONAL) — Fair-plane dengan titik StoryDiffusion
Kalau memungkinkan, hasilkan versi panel fair-plane Sun dan Heart yang menambahkan satu marker
StoryDiffusion (beda bentuk/warna dari bintang DACA), supaya thesis bisa menampilkannya visual.
Pertahankan gaya, warna, dan resolusi sama dengan `combined_adaptive_4videos.png`.

### Deliverable 5 — Ringkasan teks + config
File teks pendek berisi, per video: mean content preservation, mean concept CLIP-T,
inter_shot_sim consistent vs vanilla, dan VRAM peak. Plus laporan config:
- checkpoint SDXL, LoRA dipakai atau tidak, sampler/steps/resolusi, skema seed,
- setting StoryDiffusion (jumlah langkah dengan consistent attention, parameter id_length /
  sa32 / sa64 atau ekuivalennya), batch berapa shot sekaligus,
- deviasi apa pun dari protokol di atas, ditulis sebagai caveat eksplisit.

---

## Sanity-check
- `content_preservation` dan semua cosine harus di rentang 0..1.
- `concept_clip_t` diperkirakan berada di kisaran yang mirip thesis (concept CLIP-T sekitar
  0.24..0.37). Kalau jauh di luar itu, kemungkinan model CLIP atau preprocessing beda dari
  full-run, perbaiki dulu.
- Kalau homogenisasi terjadi, `inter_shot_sim_consistent` harus lebih tinggi dari
  `inter_shot_sim_vanilla`. Laporkan keduanya apa pun hasilnya.

## Risiko dan fallback
- VRAM 24GB: StoryDiffusion di SDXL semestinya muat. Kalau OOM, kurangi jumlah shot per batch
  atau turunkan resolusi, lalu catat. Jangan diam-diam mengubah parameter penting tanpa lapor.
- Kalau repo StoryDiffusion tidak bisa jalan dalam waktu wajar, fallback ke ConsiStory dengan
  protokol yang sama. Kalau dua-duanya gagal, laporkan blocker, jangan paksakan output
  setengah jadi atau janky.

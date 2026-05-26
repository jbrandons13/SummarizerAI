# Brief: Diagnose Freeze & Missing Scenes in Final Output

**Task type:** Diagnostic (read-only investigation), bukan fix.
**Goal:** Identify root cause kenapa final output `data/output/review_1/summary_grouping_gate.mp4` punya symptom: "cuma beberapa scene yang muncul, beberapa scene freeze lama banget" (per user observation setelah Phase 5 LTX integration).

---

## Investigation 1: Segments audit

*Note: Pipeline run dengan Groq LLM backend menghasilkan 5 groups (2 generate, 3 retrieve) karena perbedaan panjang dan pembagian kalimat summarization dibanding run sebelumnya yang menghasilkan 7 groups. Namun, karakteristik bug and underlying assembly logic tetap identik.*

| group_id | action | audio_duration | source segment exists? | source duration | generated clip exists? | generated duration | resolution |
|---|---|---|---|---|---|---|---|
| 0 | generate | 7.752s | No | 5.138s (Scene 4) | Yes | 7.767s | 768x512 |
| 1 | retrieve | 10.952s | Yes | 10.952s (Scene 8-10) | No | N/A | 640x320 |
| 2 | retrieve | 11.016s | Yes | 11.016s (Scene 15-16) | No | N/A | 640x320 |
| 3 | generate | 13.277s | No | 11.245s (Scene 38) | Yes | 8.033s | 768x512 |
| 4 | retrieve | 14.557s | Yes | 14.557s (Scene 50) | No | N/A | 640x320 |

---

## Investigation 2: Assembler logic

- **clip < audio handling:** `none`. Di `src/phase5_assemble.py`, assembler mengasumsikan durasi video segment (`v_dur`) sama dengan target durasi audio (`a_dur`):
  ```python
  # Line 222 in src/phase5_assemble.py:
  v_dur = seg_metadata.source_time_range[1] - seg_metadata.source_time_range[0]
  a_dur = seg_metadata.group_audio_duration
  ```
  Assembler tidak mem-probe file video hasil generasi LTX (`group_003.mp4`) untuk mengetahui durasi aslinya (8.033s). Akibatnya, spacer video yang digenerate hanya sebesar `padding_s` (0.15s), menyisakan gap durasi video stream sebesar `13.277 - 8.033 = 5.244s` yang tidak tertangani.
- **Resolution mismatch handling:** `missing`. Di `src/phase5_assemble.py`, video segments disalin langsung dari folder generated (jika aksi = generate) atau dipotong dari video asli tanpa scaling/resize:
  ```python
  # Lines 197-201 in src/phase5_assemble.py:
  if use_generated:
      # Copy generated video to segment path
      shutil.copy(generated_path, video_seg_path)
  else:
      # Cut video segment from source video (re-encoded, silent)
      cut_video_segment(original_video_path, cut_start, cut_end, video_seg_path)
  ```
  Fungsi `cut_video_segment` di `src/utils/ffmpeg_ops.py` (lines 49-83) mere-encode video segment tanpa scaling/resize filter, sehingga output segment tetap mengikuti resolusi video asli (`640x320`). Di sisi lain, LTX generator menghasilkan video dengan resolusi `768x512` (hardcoded di `src/phase5_ltx_runner.py` lines 229-230).
- **ffmpeg concat method:** `demuxer`. Concat dilakukan menggunakan FFmpeg concat demuxer dengan stream copy (`-c copy`) di `src/utils/ffmpeg_ops.py` (lines 85-116):
  ```python
  (
      ffmpeg
      .input(temp_file_path, format='concat', safe=0)
      .output(str(out_path), c='copy')
      .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
  )
  ```
  Stream copy pada concat demuxer menggabungkan file beresolusi berbeda secara mentah tanpa re-encoding.
- **Critical findings:**
  1. Concat demuxer menggabungkan file `768x512` (segment 0) dan `640x320` (segment 1) ke dalam satu file container.
  2. Saat subtitle burning terjadi di `mux_video_audio` (lines 117-133), FFmpeg dipaksa melakukan decode-encode ulang menggunakan `libx264`.
  3. Ketika H.264 decoder mendeteksi perubahan resolusi dari `768x512` ke `640x320` pada timestamp `8.33s`, decoder mengalami error/stuck, sehingga hanya merender frame terakhir dari segment pertama berulang-ulang (freeze) sepanjang sisa durasi video.

---

## Investigation 3: ffprobe final output

- **Duration (Format):** 59.955000s
- **Duration (Video Stream):** 58.933333s
- **Duration (Audio Stream):** 59.954000s
- **Resolution:** 768x512 (mengikuti metadata stream pertama)
- **Scene cuts (I-frames):** 9
- **I-frame timestamps:**
  - `0.000000`
  - `8.333333`
  - `16.666667`
  - `25.000000`
  - `33.333333`
  - `41.666667`
  - `50.000000`
  - `58.333333`
  - `58.833333`
- **Expected scene cuts:** 5 (satu per group boundary)
- **Match:** `No`. I-frame hanya berada pada interval GOP default (250 frame @ 30fps = setiap ~8.333 detik) karena tidak ada scene cut visual yang terdeteksi setelah video mengalami freeze di awal.

---

## Investigation 4: Frame uniqueness

- **Total frames sampled:** 1768
- **Unique frames:** 199 (11.3%)
- **Frozen frames:** 1569 (88.7%)
- **Freeze regions detected:**
  - `7.63s` to `8.33s` (Duration: `0.70s`) — static spacer di akhir segment 0.
  - `8.33s` to `58.33s` (Duration: `50.00s`) — freeze total akibat kegagalan decoder H.264 saat transisi ke segment 1 (`640x320`).
  - `58.33s` to `58.83s` (Duration: `0.50s`) — sisa static frame di akhir video stream.

---

## Investigation 5: Pre-LTX comparison

- **Pre-LTX accessible:** `Yes`. Kita meregenerasi baseline retrieval-only dengan merubah `gate_threshold` ke `-1.0` untuk memaksa bypass LTX generation.
- **Differences in duration, scene count, freeze regions:**
  - **Resolution:** Pre-LTX output secara konsisten beresolusi `640x320` (tidak ada mismatch).
  - **Duration:** Pre-LTX video berdurasi `58.26s` (video stream) / `59.95s` (audio stream), sedangkan post-LTX video berdurasi `58.93s` (video stream).
  - **Freeze regions:** Pre-LTX memiliki `0` freeze regions berdurasi >= 0.5s. Persentase frame aktif/unik mencapai `99.4%` (hanya `0.6%` frozen frames), sedangkan post-LTX memiliki `88.7%` frozen frames.
  - **Scene cuts (I-frames):** Pre-LTX memiliki keyframe yang pas sejajar dengan batas transisi segmen (misal di `7.91s`, `19.02s`, `30.20s`, `43.61s`), membuktikan transisi visual berjalan lancar tanpa terputus.

---

## Root cause hypothesis

- **Most likely cause:**
  1. **Decoder Crash Akibat Resolution Mismatch:** Concat demuxer menggunakan `-c copy` menggabungkan segmen video hasil LTX (`768x512`) dengan segmen video asli (`640x320`) tanpa resizing. Ketika FFmpeg melakukan re-encode saat proses burning subtitle, decoder mengalami error saat transisi resolusi di timestamp `8.33s`, mengakibatkan sisa frame video membeku (freeze) di frame terakhir segmen 0.
  2. **Audio-Video Drift Akibat Mismatch Durasi LTX:** LTX generator dibatasi untuk memproduksi video maksimal 241 frame (8.033s), namun assembler mengasumsikan durasi video cocok dengan durasi audio (13.277s) tanpa memvalidasi durasi file video secara terprogram. Hal ini menyebabkan desinkronisasi audio dan video.
- **Supporting evidence:**
  - Probing individual segment menunjukkan resolusi bercampur antara `768x512` dan `640x320`.
  - Deteksi programmatic mengonfirmasi video freeze total dari `8.33s` hingga `58.33s` (durasi freeze 50.00s).
  - Output run retrieval-only (tanpa LTX) berjalan sempurna tanpa freeze frames dan memiliki resolusi `640x320` yang seragam.
- **Recommended fix area:**
  - **Unifikasi Resolusi:** Modifikasi `src/phase5_assemble.py` dan `src/utils/ffmpeg_ops.py` agar semua segmen video di-resize/scale ke target resolusi yang sama (misalnya ke resolusi target default atau resolusi video input) sebelum proses concat dilakukan.
  - **Validasi Durasi Video Segmen:** Modifikasi `src/phase5_assemble.py` untuk mengukur durasi riil file video segmen menggunakan `ffprobe` sebelum mengkalkulasi durasi spacer, dan menangani gap durasi (misal dengan looping atau freeze frame lokal pada tingkat segmen) jika clip lebih pendek dari audio.

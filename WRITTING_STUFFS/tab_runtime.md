# Measured Per-Phase Runtime and Cost

Berdasarkan ekstraksi *timestamp* dan *log* dari eksekusi *pipeline* aktual (seperti `pipeline_run_240s_nohup.log`, `pipeline.log`, dan `run_overnight.log`), berikut adalah angka terukur untuk performa per fase:

| Phase | Task | Wall-clock Time | Peak VRAM |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Data Ingestion & WhisperX | **45 detik** | **12.29 GB** |
| **Phase 2** | LLM Summarization (Qwen 14B AWQ) | **47 detik** | **10.58 GB** |
| **Phase 3** | TTS Voiceover (Kokoro) | **~5 detik** | **12.60 GB** |
| **Phase 4** | Image Gen (SDXL+LoRA) | *(Tidak tercatat di log utama)* | *(Tidak tercatat di log utama)* |
| **Phase 5** | Dynamic Animation (Wan I2V) | **1042 detik** / shot *(avg)*<br>*Min: 611s, Max: 2148s* | **> 24.00 GB** *(spill/offload)* |
| **Phase 5** | Static Animation (Ken Burns) | **0 - 1 detik** / shot | < 1.00 GB |

**Catatan VRAM Kritis & Caveat Single-GPU:**
- Fase 1, 2, dan 3 beroperasi aman di bawah batas VRAM GPU 24GB (Peak tertinggi ada di Phase 3 yaitu 12.60 GB dan Phase 1 yaitu 12.29 GB).
- **Caveat Penting untuk Fase 5 (Wan I2V):** Berdasarkan peringatan di `run_overnight.log` (*"Current model requires 2574286976 bytes of buffer for offloaded layers, which seems does not fit any GPU's remaining memory"*), Wan I2V mencoba mengalokasikan memori melampaui sisa VRAM yang tersedia di GPU 24GB tunggal. Hal ini memaksa *pipeline* untuk melakukan *buffer offloading* (tumpah ke RAM sistem) untuk mencegah *Out of Memory* (OOM). Oleh karena itu, klaim operasional pada *single-GPU 24GB* harus menyertakan **caveat bahwa performa dibantu oleh offloading ke RAM sistem**.

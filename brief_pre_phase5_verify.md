# Brief: Pre-Phase-5 Quick Verification

**Task type:** Audit / verification (read-only + minimal checks), bukan execution.
**Goal:** Verify 3 asumsi kritis sebelum Phase 5 LTX integration. Setiap asumsi yang salah berpotensi bikin rework di Phase 5. Total target time: 10 menit.

## Context

Foundation untuk Phase 5 LTX integration udah siap:
- Phase 4 grouping + gate working (timestamp bug fixed)
- Phase 5 assembler consume per-group format dengan fallback
- Pipeline end-to-end runnable di review_1

Tapi 3 asumsi belum eksplisit verified. Brief ini quick-check itu.

**Important:** No fixes, no implementations. Just report findings.

## Verification 1: Phase 2 keywords field

Handoff v4 claim Phase 2 LLM summary output punya field `keywords` (3-5 visual nouns per sentence). Field ini akan dipake LLM prompt rewriter untuk Phase 5 LTX.

### What to check

1. Open existing Phase 2 output (e.g. `data/intermediate/review_1/summary.json` atau equivalent path).
2. Verify struktur tiap sentence punya field `keywords`:
   ```json
   {
     "id": 0,
     "text": "...",
     "estimated_duration_seconds": ...,
     "source_timestamp_hint": [...],
     "keywords": ["xiaomi", "smartphone", "gaming case"]    ← THIS
   }
   ```
3. Sampling: lihat 3 sentences random dari review_1 — apakah keywords-nya quality (visual nouns) atau garbage/empty?

### Schema cross-check

Cross-reference dengan `src/schemas.py`:
- `SummarySentence` model punya `keywords: List[str]`? (handoff said yes)
- Field required atau optional?

### What to report

- File path Phase 2 output
- Keywords field present: yes/no
- Sample 3 entries dengan id + text (first 50 chars) + keywords list
- Quality assessment singkat: "looks good" / "keywords too generic" / "many empty"

## Verification 2: Phase 3 audio per sentence

Phase 5 LTX module akan butuh audio duration per sentence (lalu di-aggregate per group). Audio files harus exist per sentence di disk.

### What to check

1. List files di `data/intermediate/review_1/audio/`:
   ```bash
   ls -la data/intermediate/review_1/audio/
   ```
2. Verify struktur naming sesuai dengan referensi di `Phase5SegmentMetadata.audio_path` (e.g. `audio/sentence_000.wav`).
3. Count: jumlah .wav files vs jumlah sentences di Phase 2 output. Should match.
4. Cek 1 file: pakai `ffprobe` atau Python (`librosa.get_duration` / `soundfile.info`) untuk verify file readable dan punya audio data valid (bukan 0-byte file).

### What to report

- Folder path Phase 3 audio output
- N audio files vs N sentences: <num> vs <num>
- Naming pattern: <e.g. sentence_NNN.wav>
- Sample file duration: <value seconds>
- Any anomalies (missing files, 0-byte files, etc.)

## Verification 3: VRAMManager state

Phase 5 LTX butuh model loading/unloading orchestration. VRAMManager harus bisa unload model sepenuhnya (free GPU memory) supaya LTX bisa di-load dengan VRAM yang available.

### What to check

1. Find `src/utils/vram.py` (atau wherever `VRAMManager` defined).
2. Review methods yang ada:
   - `load_model()` / `unload_current_model()` / `unload_all()` ?
   - GPU memory freeing logic (`torch.cuda.empty_cache()` etc.) ?
   - Tracking of loaded model ?
3. **Light test**: write a small Python snippet that:
   - Import VRAMManager
   - Instantiate
   - Load a small model (e.g. SigLIP atau yang udah ada di pipeline)
   - Check `torch.cuda.memory_allocated()` before & after
   - Call `unload_current_model()`
   - Check memory again
   - Verify memory dropped significantly after unload

   Don't write to disk, just print results.

4. **Ollama SIGSTOP/SIGCONT check**: search codebase untuk pattern terkait kontrol process Ollama (e.g. `subprocess.run`, `os.kill`, `SIGSTOP`, `SIGCONT`). Apakah ada implementasi-nya?

### What to report

- VRAMManager location: <file path>
- Methods available: <list>
- Light test result:
  - Memory before load: X GB
  - Memory after load: X GB
  - Memory after unload: X GB
  - Unload working: yes/no
- Ollama orchestration: <found / not found, with location>

## Hard rules

- **Read-only audit + light test.** Jangan modify code untuk fix.
- **Sampling not exhaustive.** 3 sentences for Phase 2, 1 audio file for Phase 3 — enough untuk verify, bukan benchmark.
- **STOP and report kalau audit reveal critical issue** (e.g. Phase 2 keywords field missing entirely). Jangan auto-implement workaround.

## Anti-hallucination

- Quote actual file contents (paths, snippets, sample values).
- Memory measurements harus dari `torch.cuda.memory_allocated()`, bukan estimasi.
- Kalau field/file/method ga ditemukan, bilang "NOT FOUND" — jangan asumsi.

## Out of scope

- Phase 5 LTX implementation
- Fixes untuk apa yang ditemukan
- Detailed performance benchmarks
- Re-running pipeline

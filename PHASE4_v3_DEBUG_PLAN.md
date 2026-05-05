# TASK: Phase 4 v3 — Debug & Verification Plan

## Kenapa plan ini ada

Hasil ablation pertama di video Apple Macbook review (10 menit) menunjukkan
indikasi serius bahwa **v3 tidak bekerja sebagaimana mestinya**:

1. **Temporal Accuracy = 0.000 untuk SEMUA config** → metric atau pipeline rusak
2. **Visual Coherence = 0.000 untuk SEMUA config** → metric atau propagation rusak
3. **CLIPScore counter-intuitive:**
   - SigLIP Direct (0.570) **lebih tinggi** dari SigLIP+Temporal (0.548)
   - DP (0.522) **lebih rendah** dari Greedy (0.548)
   - Random (0.477) terlalu dekat dengan SigLIP (0.570) — gap cuma 0.093

Implementasi v3 yang sudah selesai **belum bisa dipercaya** sampai dibuktikan
benar. Plan ini dirancang untuk debug sistematis, bukan tambah fitur.

**Aturan utama: jangan lanjut ke v4 sebelum semua check di plan ini PASS.**

---

## DIAGNOSA AWAL (hipotesis yang harus dibuktikan)

Tiga skenario kemungkinan, dan tujuan plan ini adalah menentukan mana yang benar:

**Skenario A: Bug di metrics implementation**
Pipeline berjalan benar, hasil retrieval valid, tapi 2 metrik baru
(Temporal Accuracy, Visual Coherence) implementasinya salah dan return 0.
**Probabilitas: tinggi** karena keduanya 0 untuk semua config — kalau pipeline
yang rusak, biasanya tidak persis 0.

**Skenario B: Bug di propagation `best_frame_path` / `best_frame_timestamp`**
Multi-frame extraction jalan, tapi field `best_frame_path` di SceneMatch
tidak terisi. Akibatnya:
- Visual coherence tidak bisa lookup embedding → return 0
- Temporal alignment pakai fallback timestamp yang salah → return 0
**Probabilitas: tinggi** karena di v3 ini perubahan baru.

**Skenario C: Bug fundamental di matching algorithm**
DP atau Hungarian sebenarnya tidak mempengaruhi hasil retrieval — assignment
identik dengan Greedy. Itu menjelaskan kenapa CLIPScore tidak meningkat.
**Probabilitas: medium** — sanity test v3 dulu lulus, tapi tidak menjamin
algoritma berjalan dalam pipeline lengkap.

**Skenario D: Talking-head failure mode**
Pipeline benar, metrics benar, tapi video Macbook review memang sulit karena
talking-head. Random ≈ SigLIP karena semua scene mirip secara visual.
**Probabilitas: rendah** untuk menjelaskan SEMUA observasi (terutama 0.000
untuk Temporal Accuracy dan Visual Coherence).

---

## STRUKTUR DEBUGGING

5 step. **Jalankan berurutan**, setiap step PASS sebelum lanjut.

```
STEP 1 → Inspeksi data flow (cek output JSON)
   ↓ PASS
STEP 2 → Verifikasi metric implementation
   ↓ PASS
STEP 3 → Verifikasi matching algorithm aktual
   ↓ PASS
STEP 4 → Re-run ablation single video, validasi hasil
   ↓ PASS
STEP 5 → Multi-video sanity check (1 talking-head + 1 diverse content)
   ↓ PASS
GO ke v4 plan
```

---

## STEP 1: Inspeksi Data Flow

**Tujuan:** Konfirmasi bahwa output pipeline punya semua field yang dibutuhkan
metric. Ini diagnose Skenario B (propagation bug).

### 1.1 Inspeksi output SceneMatch

Buat script diagnostic, jangan cuma lihat sekilas:

```python
# scripts/debug_v3_inspect_output.py

import json
from pathlib import Path

def inspect_match_output(output_dir: Path, config_name: str):
    """Print struktur SceneMatch untuk verify field propagation."""
    matches_path = output_dir / config_name / "matches.json"

    if not matches_path.exists():
        print(f"❌ FILE TIDAK ADA: {matches_path}")
        return

    with open(matches_path) as f:
        data = json.load(f)

    matches = data.get("matches", data) if isinstance(data, dict) else data
    print(f"\n=== Config: {config_name} ===")
    print(f"Total matches: {len(matches)}")

    if not matches:
        print("❌ MATCHES KOSONG")
        return

    # Inspeksi 3 match pertama
    for i, m in enumerate(matches[:3]):
        print(f"\nMatch {i}:")
        print(f"  sentence_id: {m.get('sentence_id')}")
        print(f"  matched_scene_id: {m.get('matched_scene_id')}")
        print(f"  score: {m.get('score')}")
        print(f"  best_frame_path: {m.get('best_frame_path', 'MISSING')!r}")
        print(f"  best_frame_timestamp: {m.get('best_frame_timestamp', 'MISSING')}")

    # Aggregate check
    has_best_path = sum(1 for m in matches if m.get("best_frame_path"))
    has_best_ts = sum(1 for m in matches if m.get("best_frame_timestamp", 0) > 0)
    print(f"\nAggregate:")
    print(f"  Matches with best_frame_path: {has_best_path}/{len(matches)}")
    print(f"  Matches with best_frame_timestamp > 0: {has_best_ts}/{len(matches)}")

    return matches


# Run untuk semua configs
configs = ["A_random", "B_caption", "C_siglip", "CT_siglip",
           "CT_hungarian", "CT_dp"]
for c in configs:
    inspect_match_output(Path("outputs/macbook_review"), c)
```

**Expected output kalau v3 benar:**
- `best_frame_path` non-empty untuk SEMUA matches
- `best_frame_timestamp > 0` untuk SEMUA matches

**Kalau gagal:** Lokasi bug ada di `phase4_retrieve.py` saat conversion
assignment → SceneMatch. Cek apakah `best_frames` lookup dipanggil dengan benar.

### 1.2 Inspeksi output summary (sentences)

```python
def inspect_summary_output(output_dir: Path):
    """Verify source_timestamp_hint terisi di summary sentences."""
    summary_path = output_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    sentences = summary.get("sentences", [])
    print(f"\n=== Summary inspection ===")
    print(f"Total sentences: {len(sentences)}")

    has_hint = sum(1 for s in sentences if s.get("source_timestamp_hint"))
    print(f"Sentences with source_timestamp_hint: {has_hint}/{len(sentences)}")

    for i, s in enumerate(sentences[:5]):
        hint = s.get("source_timestamp_hint")
        text = s.get("text", "")[:60]
        print(f"  Sentence {i}: hint={hint}, text={text!r}")
```

**Expected:** Hampir semua sentences punya `source_timestamp_hint = [start, end]`.
**Kalau gagal:** Bug di Phase 3 (LLM summarizer), bukan Phase 4.

### 1.3 Inspeksi embeddings cache

```python
import joblib

def inspect_embeddings_cache(cache_path: Path):
    """Verify embeddings cache structure dan content."""
    if not cache_path.exists():
        print(f"❌ CACHE TIDAK ADA: {cache_path}")
        return

    embs = joblib.load(cache_path)
    print(f"\n=== Embeddings cache: {cache_path.name} ===")
    print(f"Type: {type(embs)}")

    if isinstance(embs, dict):
        print(f"Total entries: {len(embs)}")
        sample_keys = list(embs.keys())[:5]
        print(f"Sample keys: {sample_keys}")

        if sample_keys:
            sample_val = embs[sample_keys[0]]
            print(f"Sample value shape: {sample_val.shape}")
            print(f"Sample value dtype: {sample_val.dtype}")
    else:
        print(f"❌ CACHE BUKAN DICT — visual coherence pasti gagal")
```

**Expected:** Dict dengan keys `(scene_id, frame_timestamp)` dan value np.ndarray.

### 1.4 Acceptance criteria STEP 1

- [ ] Semua matches punya `best_frame_path` non-empty
- [ ] Semua matches punya `best_frame_timestamp > 0`
- [ ] Hampir semua sentences punya `source_timestamp_hint`
- [ ] Embeddings cache adalah dict dengan struktur yang benar

**Kalau ada yang fail → fix dulu sebelum lanjut. Plan section "Fixes" di bawah.**

---

## STEP 2: Verifikasi Metric Implementation

**Tujuan:** Test 2 metrik baru di-isolasi dari pipeline. Diagnose Skenario A.

### 2.1 Unit test temporal_alignment_score

Buat test yang **harus pass** sebelum lanjut:

```python
# tests/test_metrics_temporal.py

import pytest
from src.eval.metrics import temporal_alignment_score
from src.schemas import SceneMatch, KeyframeScene, SummarySentence

def test_perfect_alignment_zero_error():
    """Match jatuh persis di tengah hint window → error 0."""
    matches = [SceneMatch(
        sentence_id=0, matched_scene_id=0, score=1.0,
        best_frame_path="x", best_frame_timestamp=15.0,
    )]
    summary = type("S", (), {"sentences": [SummarySentence(
        id=0, text="x", source_timestamp_hint=[10.0, 20.0]
    )]})()
    manifest = type("M", (), {"scenes": [KeyframeScene(
        id=0, start_seconds=10, end_seconds=20,
        keyframe_path="x", keyframe_timestamp=15.0,
    )]})()

    result = temporal_alignment_score(matches, summary, manifest)
    print(f"Result: {result}")
    assert result["mean_temporal_error_seconds"] == 0.0
    assert result["temporal_accuracy_within_5s"] == 1.0


def test_outside_window_correct_error():
    """Match di luar window → error = jarak ke edge terdekat."""
    matches = [SceneMatch(
        sentence_id=0, matched_scene_id=0, score=1.0,
        best_frame_path="x", best_frame_timestamp=30.0,  # window [10,20], match di 30
    )]
    summary = type("S", (), {"sentences": [SummarySentence(
        id=0, text="x", source_timestamp_hint=[10.0, 20.0]
    )]})()
    manifest = type("M", (), {"scenes": [KeyframeScene(
        id=0, start_seconds=25, end_seconds=35,
        keyframe_path="x", keyframe_timestamp=30.0,
    )]})()

    result = temporal_alignment_score(matches, summary, manifest)
    print(f"Result: {result}")
    assert result["mean_temporal_error_seconds"] == 10.0  # 30 - 20
    assert result["temporal_accuracy_within_5s"] == 0.0
    assert result["temporal_accuracy_within_15s"] == 1.0


def test_no_hint_skipped():
    """Sentence tanpa hint harus di-skip, bukan crash."""
    matches = [SceneMatch(
        sentence_id=0, matched_scene_id=0, score=1.0,
        best_frame_path="x", best_frame_timestamp=15.0,
    )]
    summary = type("S", (), {"sentences": [SummarySentence(
        id=0, text="x", source_timestamp_hint=None
    )]})()
    manifest = type("M", (), {"scenes": [KeyframeScene(
        id=0, start_seconds=10, end_seconds=20,
        keyframe_path="x", keyframe_timestamp=15.0,
    )]})()

    result = temporal_alignment_score(matches, summary, manifest)
    print(f"Result: {result}")
    # n_evaluated=0, semua metric -1 atau 0
    assert result.get("n_evaluated", 0) == 0
```

**Run:**
```bash
pytest tests/test_metrics_temporal.py -v -s
```

**Kalau gagal:** Lihat traceback. Kemungkinan bugs:
1. `match.best_frame_timestamp` tidak dipakai (masih pakai `scene.keyframe_timestamp`)
2. Loop `for match in matches` skip semua karena conditional salah
3. Window check `hint[0] <= ts <= hint[1]` ada off-by-one atau type mismatch

### 2.2 Unit test visual_coherence_score

```python
# tests/test_metrics_coherence.py

import numpy as np
from src.eval.metrics import visual_coherence_score
from src.schemas import SceneMatch

def test_identical_frames_high_coherence():
    """2 match dengan embedding sama → coherence ≈ 1.0"""
    same_emb = np.random.randn(128).astype(np.float32)
    matches = [
        SceneMatch(sentence_id=0, matched_scene_id=0, score=1.0,
                   best_frame_path="a", best_frame_timestamp=5.0),
        SceneMatch(sentence_id=1, matched_scene_id=1, score=1.0,
                   best_frame_path="b", best_frame_timestamp=15.0),
    ]
    embeddings = {
        (0, 5.0): same_emb,
        (1, 15.0): same_emb,
    }

    result = visual_coherence_score(matches, embeddings)
    print(f"Result: {result}")
    assert result["visual_coherence_mean"] > 0.99


def test_orthogonal_frames_zero_coherence():
    """2 match dengan embedding orthogonal → coherence ≈ 0"""
    matches = [
        SceneMatch(sentence_id=0, matched_scene_id=0, score=1.0,
                   best_frame_path="a", best_frame_timestamp=5.0),
        SceneMatch(sentence_id=1, matched_scene_id=1, score=1.0,
                   best_frame_path="b", best_frame_timestamp=15.0),
    ]
    embeddings = {
        (0, 5.0): np.array([1.0, 0.0, 0.0], dtype=np.float32),
        (1, 15.0): np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    result = visual_coherence_score(matches, embeddings)
    print(f"Result: {result}")
    assert abs(result["visual_coherence_mean"]) < 0.01


def test_missing_embedding_keys_handled():
    """Kalau embedding key tidak ada di dict, function harus skip pasangan
    itu, bukan return 0 untuk semua."""
    matches = [
        SceneMatch(sentence_id=0, matched_scene_id=0, score=1.0,
                   best_frame_path="a", best_frame_timestamp=5.0),
        SceneMatch(sentence_id=1, matched_scene_id=1, score=1.0,
                   best_frame_path="b", best_frame_timestamp=15.0),
        SceneMatch(sentence_id=2, matched_scene_id=2, score=1.0,
                   best_frame_path="c", best_frame_timestamp=25.0),
    ]
    same_emb = np.random.randn(128).astype(np.float32)
    embeddings = {
        (0, 5.0): same_emb,
        (1, 15.0): same_emb,
        # (2, 25.0) sengaja missing
    }

    result = visual_coherence_score(matches, embeddings)
    print(f"Result: {result}")
    # Pasangan (0,1) jalan; pasangan (1,2) skip karena missing
    # n_pairs harus 1, bukan 0
    assert result.get("n_pairs", 0) >= 1
```

### 2.3 Test apa yang sebenarnya di-pass ke metric

Tambahkan logging temporary di `run_ablation.py`:

```python
# Sebelum panggil metric
print(f"\n=== Calling temporal_alignment_score for {config_name} ===")
print(f"Number of matches: {len(matches)}")
print(f"First match: {matches[0] if matches else 'NONE'}")
print(f"Number of sentences: {len(summary.sentences)}")
print(f"First sentence hint: {summary.sentences[0].source_timestamp_hint if summary.sentences else 'NONE'}")

temporal_result = temporal_alignment_score(matches, summary, manifest)
print(f"Temporal result: {temporal_result}")

print(f"\n=== Calling visual_coherence_score for {config_name} ===")
print(f"Frame embeddings keys count: {len(frame_embeddings)}")
print(f"Sample embedding key: {list(frame_embeddings.keys())[0] if frame_embeddings else 'NONE'}")

coherence_result = visual_coherence_score(matches, frame_embeddings)
print(f"Coherence result: {coherence_result}")
```

**Run sekali ablation pendek (1 video, 1 config) dengan logging ini.** Lihat
output. Bug akan terlihat jelas dari sini.

### 2.4 Acceptance criteria STEP 2

- [ ] Semua test temporal pass
- [ ] Semua test coherence pass
- [ ] Logging di run_ablation menunjukkan input metric tidak kosong/None
- [ ] Saat run mini-ablation, metric return value > 0 untuk minimal 1 config

---

## STEP 3: Verifikasi Matching Algorithm

**Tujuan:** Konfirmasi DP, Hungarian, Greedy benar-benar **menghasilkan
assignment yang berbeda**. Diagnose Skenario C.

### 3.1 Cetak assignment per config

```python
# scripts/debug_v3_compare_assignments.py

def compare_assignments(output_dir: Path):
    """Bandingkan assignment antar config untuk 1 video."""
    configs = ["CT_siglip", "CT_hungarian", "CT_dp"]
    assignments = {}

    for c in configs:
        with open(output_dir / c / "matches.json") as f:
            data = json.load(f)
        matches = data.get("matches", data) if isinstance(data, dict) else data
        assignments[c] = [m["matched_scene_id"] for m in matches]

    # Print side-by-side
    print(f"\n{'sent_id':<10}{'Greedy':<12}{'Hungarian':<12}{'DP':<12}")
    print("-" * 46)
    for i in range(len(assignments["CT_siglip"])):
        g = assignments["CT_siglip"][i]
        h = assignments["CT_hungarian"][i]
        d = assignments["CT_dp"][i]
        marker = "" if g == h == d else " ← DIFFERENT"
        print(f"{i:<10}{g:<12}{h:<12}{d:<12}{marker}")

    # Aggregate
    diff_gh = sum(1 for i in range(len(assignments["CT_siglip"]))
                  if assignments["CT_siglip"][i] != assignments["CT_hungarian"][i])
    diff_gd = sum(1 for i in range(len(assignments["CT_siglip"]))
                  if assignments["CT_siglip"][i] != assignments["CT_dp"][i])
    diff_hd = sum(1 for i in range(len(assignments["CT_hungarian"]))
                  if assignments["CT_hungarian"][i] != assignments["CT_dp"][i])

    print(f"\nDifferent assignments:")
    print(f"  Greedy vs Hungarian: {diff_gh}")
    print(f"  Greedy vs DP: {diff_gd}")
    print(f"  Hungarian vs DP: {diff_hd}")
```

**Expected:**
- Greedy vs Hungarian: minimal beberapa berbeda (Hungarian re-allocate untuk
  total skor lebih tinggi)
- Greedy vs DP: BANYAK yang berbeda (DP optimasi temporal coherence)
- Hungarian vs DP: ada yang berbeda

**Kalau semua identik:** Bug. Salah satu kemungkinan:
1. Algoritma return assignment greedy by default (config tidak di-baca)
2. `jump_penalty=0.3` terlalu rendah, DP de-facto = greedy
3. `linear_sum_assignment` di Hungarian return sama karena reuse_penalty = 0

### 3.2 Cetak DP transition cost matrix

```python
# Ditambahkan ke dp_sequence_align untuk debug
def dp_sequence_align(self, sim_matrix, scenes, video_duration, ...):
    # ... existing code sampai transition_matrix dibuat ...

    # DEBUG: print transition cost summary
    print(f"\n=== DP transition matrix stats ===")
    print(f"Shape: {transition_matrix.shape}")
    print(f"Diagonal (same-scene): mean={np.diag(transition_matrix).mean():.4f}")
    print(f"Off-diagonal: mean={transition_matrix[~np.eye(len(transition_matrix), dtype=bool)].mean():.4f}")
    print(f"Min: {transition_matrix.min():.4f}, Max: {transition_matrix.max():.4f}")

    # ... lanjut existing code ...
```

**Expected output:**
- Diagonal mean: negatif (~-0.3, karena reuse_bonus)
- Off-diagonal mean: positif (~0.1-0.5, karena jump_penalty)
- Range tidak boleh nol-semua

**Kalau diagonal dan off-diagonal sama:** Bug, transition tidak diaplikasikan.

### 3.3 Verify DP changes assignment vs greedy via internal test

```python
# tests/test_dp_changes_assignment.py

import numpy as np

def test_dp_differs_from_greedy_when_jumps_costly():
    """
    Kasus dirancang sehingga greedy akan loncat tapi DP tidak.

    sim_matrix (3 sentences x 4 scenes), times = [0, 30, 60, 90]:
        Sentence 0: prefer scene 0 (0.9)
        Sentence 1: prefer scene 3 (0.8) — jauh dari sentence 0
        Sentence 2: prefer scene 1 (0.7) — kembali mundur

    Greedy: [0, 3, 1] — total skor 0.9+0.8+0.7 = 2.4, banyak loncat
    DP dengan jump penalty: should pick path lebih monoton meskipun skor
    individual lebih rendah.
    """
    sim = np.array([
        [0.9, 0.5, 0.4, 0.3],
        [0.2, 0.4, 0.6, 0.8],
        [0.3, 0.7, 0.5, 0.4],
    ])
    scenes = [
        type("S", (), {"keyframe_timestamp": 0,  "end_seconds": 10})(),
        type("S", (), {"keyframe_timestamp": 30, "end_seconds": 40})(),
        type("S", (), {"keyframe_timestamp": 60, "end_seconds": 70})(),
        type("S", (), {"keyframe_timestamp": 90, "end_seconds": 100})(),
    ]

    from src.phase4_retrieve import dp_sequence_align, greedy_assign

    greedy_a = greedy_assign(sim)
    dp_a = dp_sequence_align(sim, scenes, video_duration=100,
                            jump_penalty=1.0, reuse_bonus=0.0,
                            backward_penalty=0.5)

    print(f"Greedy: {greedy_a}")
    print(f"DP:     {dp_a}")
    assert greedy_a != dp_a, "DP should differ from greedy when jumps are penalized"


def test_dp_equals_greedy_when_penalties_zero():
    """jump_penalty=0, reuse_bonus=0, backward_penalty=0 → DP = argmax."""
    sim = np.random.rand(5, 8)
    scenes = [type("S", (), {"keyframe_timestamp": i*10, "end_seconds": i*10+10})()
             for i in range(8)]

    from src.phase4_retrieve import dp_sequence_align

    dp_a = dp_sequence_align(sim, scenes, video_duration=80,
                            jump_penalty=0.0, reuse_bonus=0.0,
                            backward_penalty=0.0)
    expected = [int(np.argmax(sim[i])) for i in range(5)]
    assert dp_a == expected, f"With zero penalties, DP must reduce to argmax. Got {dp_a}, expected {expected}"
```

**Run:**
```bash
pytest tests/test_dp_changes_assignment.py -v -s
```

### 3.4 Acceptance criteria STEP 3

- [ ] Greedy vs DP: minimum 30% assignment berbeda di mini-ablation
- [ ] DP transition matrix punya diagonal negatif, off-diagonal positif
- [ ] Test `test_dp_differs_from_greedy_when_jumps_costly` PASS
- [ ] Test `test_dp_equals_greedy_when_penalties_zero` PASS

**Kalau gagal:** ada 2 kemungkinan:
1. Config `dp_jump_penalty` tidak terbaca di runtime → cek YAML loading
2. DP recurrence salah → review `dp_sequence_align` line by line

---

## STEP 4: Re-run Single Video, Validasi Hasil

**Tujuan:** Setelah STEP 1-3 lulus, jalankan ablation lengkap di Macbook
review video ulang dan validasi hasil masuk akal.

### 4.1 Re-run dengan logging verbose

```python
# Modifikasi run_ablation.py temporary
LOG_VERBOSE = True

if LOG_VERBOSE:
    # Print sim_matrix stats per arm
    print(f"\n=== {config_name} sim_matrix stats ===")
    print(f"Shape: {sim_matrix.shape}")
    print(f"Mean: {sim_matrix.mean():.4f}")
    print(f"Std: {sim_matrix.std():.4f}")
    print(f"Min: {sim_matrix.min():.4f}, Max: {sim_matrix.max():.4f}")
```

### 4.2 Sanity invariants

```python
# tests/test_v3_invariants_real_data.py

def test_random_lower_than_siglip(results):
    """Random arm CLIPScore < SigLIP+T arm CLIPScore."""
    assert results["A_random"]["clipscore"] < results["CT_siglip"]["clipscore"], \
        "Random arm tidak boleh lebih tinggi dari SigLIP+T"


def test_temporal_accuracy_nonzero(results):
    """Minimal 1 config harus punya temporal accuracy > 0."""
    accs = [r.get("temporal_acc_30s", 0) for r in results.values()]
    assert max(accs) > 0, "Semua temporal_acc_30s = 0 — pasti bug metric"


def test_dp_higher_coherence_than_greedy(results):
    """DP harus punya visual coherence lebih tinggi dari Greedy
    (kalau DP berfungsi mempenalti loncat)."""
    dp_coh = results["CT_dp"]["visual_coherence_mean"]
    greedy_coh = results["CT_siglip"]["visual_coherence_mean"]
    assert dp_coh >= greedy_coh - 0.02, \
        f"DP coherence ({dp_coh}) harus >= Greedy ({greedy_coh}) - small tolerance"


def test_temporal_prior_helps_temporal_accuracy(results):
    """SigLIP+T harus punya temporal_acc > SigLIP (tanpa T)."""
    ct_acc = results["CT_siglip"]["temporal_acc_30s"]
    c_acc = results["C_siglip"]["temporal_acc_30s"]
    assert ct_acc >= c_acc - 0.05, \
        f"Temporal prior tidak membantu: CT={ct_acc}, C={c_acc}"
```

**Kalau invariants gagal di hasil real:** ini diagnostic info.

- `test_random_lower_than_siglip` gagal → talking-head failure mode confirmed
- `test_temporal_accuracy_nonzero` gagal → metric masih bug, balik ke STEP 2
- `test_dp_higher_coherence_than_greedy` gagal → DP gagal lakukan tugasnya
- `test_temporal_prior_helps_temporal_accuracy` gagal → temporal prior code path
  tidak jalan, atau hint quality buruk

### 4.3 Tonton output video

**Wajib lakukan ini, bukan optional.**

Buka 3 video output: `CT_siglip` (Greedy), `CT_hungarian`, `CT_dp`. Tonton
side-by-side. Pertanyaan kualitatif:

1. Apakah **video DP terlihat lebih smooth** (transisi antar B-roll lebih
   masuk akal) dibanding Greedy?
2. Apakah ada **scene yang jelas salah** di salah satu config tapi benar di
   yang lain?
3. Apakah **mostly identik**? (Indikasi semua config sebenarnya pilih scene
   yang sama → bug.)

Catat observasi di `notes/v3_qualitative_observations.md`. Ini akan jadi
material untuk thesis section "qualitative analysis."

### 4.4 Acceptance criteria STEP 4

- [ ] Minimal 3 dari 4 invariant test PASS
- [ ] Sudah tonton 3 video output side-by-side
- [ ] Sudah catat qualitative observations
- [ ] Tabel hasil punya range yang masuk akal (Random < SigLIP, ada perbedaan
      antar matching algorithm)

---

## STEP 5: Multi-Video Sanity Check

**Tujuan:** Konfirmasi bahwa hasil bukan artifact dari Macbook video saja.
Diagnose Skenario D (talking-head failure mode).

### 5.1 Pilih 2 video kontras

- **Video 1: Macbook review (existing).** Talking-head heavy. Expected:
  small differences antar config.
- **Video 2: Tutorial atau B-roll heavy video.** Misal: cooking video, travel
  vlog, atau video tutorial dengan banyak screenshot/screen recording.
  Expected: large differences antar config.

Kalau di Video 2 hasil-nya juga flat/0/random, **bug fundamental**. Kalau
Video 2 menunjukkan diferensiasi, **Macbook adalah talking-head failure mode**.

### 5.2 Bandingkan hasil

```python
# scripts/debug_v3_multi_video_comparison.py

def compare_videos(results: Dict[str, Dict[str, Dict]]):
    """
    results structure: {video_name: {config_name: {metric: value}}}
    """
    for video, configs in results.items():
        print(f"\n=== {video} ===")
        print(f"{'Config':<20}{'CLIPScore':<12}{'TempAcc30':<12}{'VisCoh':<12}")
        print("-" * 56)
        for c, metrics in configs.items():
            print(f"{c:<20}"
                  f"{metrics.get('clipscore', 0):<12.4f}"
                  f"{metrics.get('temporal_acc_30s', 0):<12.4f}"
                  f"{metrics.get('visual_coherence_mean', 0):<12.4f}")

        # Range analysis
        clipscores = [m.get('clipscore', 0) for m in configs.values()]
        print(f"\nCLIPScore range: {max(clipscores) - min(clipscores):.4f}")
```

**Expected differences:**
- Macbook (talking-head): CLIPScore range 0.05-0.15 (narrow)
- Tutorial/B-roll: CLIPScore range 0.10-0.30 (wider)

Kalau Tutorial juga narrow, bug. Kalau Tutorial wider, talking-head failure
mode confirmed.

### 5.3 Acceptance criteria STEP 5

- [ ] Sudah run di minimal 2 video (1 talking-head + 1 diverse)
- [ ] Range CLIPScore lebih wide di video diverse
- [ ] Visual coherence dan temporal accuracy bervariasi antar config
- [ ] Pola hasil **konsisten dengan teori** — DP > Greedy untuk coherence,
      SigLIP+T > SigLIP untuk temporal accuracy

---

## FIXES — Intervention Berdasarkan Hasil Debug

Section ini berisi fix yang akan diaplikasikan sesuai temuan STEP 1-5.

### Fix 1: best_frame_path tidak terisi (kalau STEP 1.1 gagal)

**Lokasi bug:** `src/phase4_retrieve.py`, conversion assignment → SceneMatch.

**Cek:** Apakah `best_frames[(sent_idx, scene_idx)]` di-build dan di-lookup
benar? Cek apakah greedy path juga propagate (di v3 plan disebut harus
diseragamkan).

**Fix template:**

```python
# Pastikan best_frames dibangun SEBELUM assignment, di scoring loop:
best_frames: Dict[Tuple[int, int], Tuple[str, float]] = {}
for sent_idx in range(len(sentences)):
    for scene_idx in range(len(scenes)):
        scene = scenes[scene_idx]
        frame_paths = scene.multi_frame_paths or [scene.keyframe_path]
        frame_ts = scene.multi_frame_timestamps or [scene.keyframe_timestamp]
        frame_scores = compute_frame_scores(...)  # per-frame similarity
        best_idx = int(np.argmax(frame_scores))
        best_frames[(sent_idx, scene_idx)] = (frame_paths[best_idx], frame_ts[best_idx])

# Lalu saat conversion ke SceneMatch:
for sent_idx, scene_idx in enumerate(assignment):
    best_path, best_ts = best_frames.get((sent_idx, scene_idx), ("", 0.0))
    matches.append(SceneMatch(
        ...,
        best_frame_path=best_path,
        best_frame_timestamp=best_ts,
    ))
```

### Fix 2: Metric implementation menggunakan `keyframe_timestamp` alih-alih `best_frame_timestamp`

**Lokasi bug:** `src/eval/metrics.py`.

**Cek:** Apakah `temporal_alignment_score` masih pakai `scene.keyframe_timestamp`?

**Fix:**

```python
# BEFORE:
retrieved_mid = scene.keyframe_timestamp

# AFTER:
retrieved_ts = match.best_frame_timestamp or scene.keyframe_timestamp
```

### Fix 3: Visual coherence lookup salah key

**Lokasi bug:** `src/eval/metrics.py`, `visual_coherence_score`.

**Cek:** Function expect `Dict[Tuple[int, float], np.ndarray]` tapi dipanggil
dengan `Dict[int, np.ndarray]` (scene-level, bukan frame-level)?

**Fix:**

```python
# Pastikan caller pass frame-level dict:
frame_embeddings = load_cache("siglip_embeddings.joblib")  # (scene_id, ts) -> emb

# Di metric:
key_a = (matches[i].matched_scene_id, matches[i].best_frame_timestamp)
emb_a = frame_embeddings.get(key_a)
if emb_a is None:
    continue  # skip pasangan ini, JANGAN return 0
```

### Fix 4: DP / Hungarian tidak dipanggil

**Lokasi bug:** `src/eval/run_ablation.py` atau retrieval class.

**Cek:** Apakah config `matching_algorithm` di-pass ke retrieval method?

**Debug print:**

```python
print(f"Config matching_algorithm: {self.config.get('retrieval', {}).get('matching_algorithm')}")
print(f"Branch taken: {matching_algo}")
```

### Fix 5: jump_penalty terlalu kecil sehingga DP ≈ Greedy

**Cek:** Run DP dengan jump_penalty {0.1, 0.3, 0.5, 1.0, 2.0} pada 1 video,
hitung berapa banyak assignment yang berbeda dari greedy untuk setiap value.

**Sweet spot:** jump_penalty di mana 30-50% assignment berbeda dari greedy.

---

## TIMELINE

**Total: 3-4 hari kerja.** Tidak boleh lebih, kalau lebih berarti masalah
fundamental yang butuh diskusi ulang.

| Day | Task |
|-----|------|
| 1   | STEP 1 (inspeksi data) + STEP 2 (verifikasi metric) |
| 2   | STEP 3 (verifikasi matching algo) + Fix berdasarkan STEP 1-3 |
| 3   | STEP 4 (re-run single video, validasi) |
| 4   | STEP 5 (multi-video) + writeup observation |

---

## DELIVERABLES

Setelah plan ini selesai, harus ada:

1. **Test files** di `tests/` yang pass:
   - `test_metrics_temporal.py`
   - `test_metrics_coherence.py`
   - `test_dp_changes_assignment.py`
   - `test_v3_invariants_real_data.py`

2. **Debug scripts** di `scripts/`:
   - `debug_v3_inspect_output.py`
   - `debug_v3_compare_assignments.py`
   - `debug_v3_multi_video_comparison.py`

3. **Document** `notes/v3_debug_findings.md` berisi:
   - Bugs yang ditemukan
   - Fix yang diaplikasikan
   - Hasil ablation Macbook + 1 video lain
   - Qualitative observation dari menonton output video

4. **Status report** ke supervisor (kalau perlu): 1 paragraf executive
   summary apakah v3 working atau ada problem.

---

## DECISION POINT

Setelah plan ini selesai, ada 3 kemungkinan:

**Outcome A: Semua pass, hasil masuk akal**
→ Lanjut ke v4 plan (`PHASE4_v4_BRANCH_PLAN.md`)

**Outcome B: Sebagian pass, ada limitations**
→ Update v4 plan dengan scope yang lebih realistis. Misal kalau talking-head
videos memang tidak bisa di-improve, fokus v4 ke video diverse-content.

**Outcome C: Banyak gagal, v3 fundamentally broken**
→ Stop, diskusi sama supervisor/aku tentang strategi. Mungkin perlu
re-implement bagian inti.

**Jangan paksakan v4 sebelum ada clarity di v3.**

---

## END OF PLAN

Implementasikan oleh coding agent atau manually. Setiap STEP punya acceptance
criteria yang jelas — jangan lanjut sebelum criteria terpenuhi.

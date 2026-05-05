# TASK: Debug Hungarian Algorithm + Tune DP jump_penalty

## Context

I'm working on Phase 4 of a video summarization pipeline (master's thesis,
single RTX 3090). The v3 implementation is complete and metrics are working,
but a 3-video ablation revealed two specific problems that need fixing
**before** any new features are added.

Plan v3 sudah selesai diimplementasikan dan metrics berjalan. Tapi hasil
ablation di 3 video menunjukkan 2 masalah spesifik yang harus di-fix
**sebelum** menambah fitur apapun.

### Ablation results (3 gadget review videos)

```
Video 1:
  random              | ClipScore: 0.5105 | TempAcc: 0.286 | VisCoher: 0.661
  caption_temporal    | ClipScore: 0.6063 | TempAcc: 0.571 | VisCoher: 0.693
  siglip_direct       | ClipScore: 0.5171 | TempAcc: 0.000 | VisCoher: 0.627
  siglip_temporal     | ClipScore: 0.5268 | TempAcc: 0.714 | VisCoher: 0.697
  siglip_temporal_hung| ClipScore: 0.5268 | TempAcc: 0.714 | VisCoher: 0.697  ← IDENTIK ke greedy
  siglip_temporal_dp  | ClipScore: 0.5583 | TempAcc: 0.143 | VisCoher: 0.975  ← TempAcc anjlok

Video 2:
  random              | ClipScore: 0.4490 | TempAcc: 0.250 | VisCoher: 0.681
  caption_temporal    | ClipScore: 0.5449 | TempAcc: 1.000 | VisCoher: 0.760
  siglip_direct       | ClipScore: 0.6226 | TempAcc: 0.000 | VisCoher: 0.814
  siglip_temporal     | ClipScore: 0.5112 | TempAcc: 1.000 | VisCoher: 0.763
  siglip_temporal_hung| ClipScore: 0.5112 | TempAcc: 1.000 | VisCoher: 0.763  ← IDENTIK
  siglip_temporal_dp  | ClipScore: 0.5262 | TempAcc: 0.500 | VisCoher: 0.923  ← TempAcc turun 50%

Video 3:
  random              | ClipScore: 0.4214 | TempAcc: 0.600 | VisCoher: 0.782
  caption_temporal    | ClipScore: 0.5934 | TempAcc: 0.800 | VisCoher: 0.773
  siglip_direct       | ClipScore: 0.4084 | TempAcc: 1.000 | VisCoher: 0.626
  siglip_temporal     | ClipScore: 0.4786 | TempAcc: 1.000 | VisCoher: 0.670
  siglip_temporal_hung| ClipScore: 0.4786 | TempAcc: 1.000 | VisCoher: 0.670  ← IDENTIK
  siglip_temporal_dp  | ClipScore: 0.4767 | TempAcc: 0.800 | VisCoher: 0.682
```

### Two problems identified

**Problem 1: Hungarian = Greedy persis sama di 3 video.**
Hungarian algorithm tidak mempengaruhi assignment. Hasil identik 100% dengan
Greedy untuk CLIPScore, TempAcc, dan VisCoher.

**Problem 2: DP terlalu agresif.**
DP berhasil meningkatkan VisCoher dramatis (V1: +0.278, V2: +0.160) tapi
mengorbankan TempAcc parah (V1: -0.571, V2: -0.500). `jump_penalty` saat ini
terlalu tinggi sehingga DP "stick" di scene yang sama atau scene berdekatan
dan lupa mengikuti hint temporal.

---

## Your tasks

Lakukan **2 task** berurutan. Task 1 dulu sampai selesai, baru Task 2.

---

## TASK 1: Debug Hungarian

### Goal
Cari tahu kenapa Hungarian menghasilkan assignment **persis identik** dengan
Greedy. Fix masalahnya kalau itu bug, atau dokumentasikan kalau itu kondisi
degenerate yang inherent.

### Step 1.1: Print assignment side-by-side

Buat script `scripts/debug_hungarian_vs_greedy.py`:

```python
"""
Compare assignment dari Greedy vs Hungarian untuk 1 video.
Print side-by-side per sentence.
"""
import json
from pathlib import Path

def load_assignments(output_dir: Path, config_name: str):
    """Load matched scene IDs from results."""
    matches_path = output_dir / config_name / "matches.json"
    with open(matches_path) as f:
        data = json.load(f)
    matches = data.get("matches", data) if isinstance(data, dict) else data
    return [(m["sentence_id"], m["matched_scene_id"], m.get("score", 0))
            for m in sorted(matches, key=lambda x: x["sentence_id"])]


def compare(video_output_dir: Path):
    greedy = load_assignments(video_output_dir, "siglip_temporal")
    hungarian = load_assignments(video_output_dir, "siglip_temporal_hungarian")

    print(f"\n{'sent':<6}{'Greedy scene':<15}{'Hungarian scene':<18}{'Match?':<10}{'G score':<10}{'H score':<10}")
    print("-" * 70)

    diff_count = 0
    for (s_id, g_scene, g_score), (_, h_scene, h_score) in zip(greedy, hungarian):
        match = "SAME" if g_scene == h_scene else "DIFF"
        if g_scene != h_scene:
            diff_count += 1
        print(f"{s_id:<6}{g_scene:<15}{h_scene:<18}{match:<10}{g_score:<10.4f}{h_score:<10.4f}")

    print(f"\nTotal differences: {diff_count} / {len(greedy)}")
    print(f"Identical assignments: {len(greedy) - diff_count} / {len(greedy)}")

    # Check apakah scene reuse terjadi di greedy
    greedy_scenes = [g[1] for g in greedy]
    unique_scenes = len(set(greedy_scenes))
    print(f"\nGreedy unique scenes used: {unique_scenes} / {len(greedy_scenes)}")
    print(f"Greedy reused scenes: {len(greedy_scenes) - unique_scenes}")


if __name__ == "__main__":
    import sys
    compare(Path(sys.argv[1]))  # pass video output dir
```

Run untuk 3 video. Catat:
- Berapa banyak assignment yang berbeda?
- Apakah Greedy reuse scenes (penting karena Hungarian seharusnya prevent reuse)?
- Apakah skor Hungarian dan Greedy sama persis?

### Step 1.2: Inspect Hungarian implementation

Cek `src/phase4_retrieve.py` fungsi `hungarian_align()`. Verifikasi:

```python
# Hal yang harus diverify:

# 1. Apakah reuse_penalty != 0?
# Jika reuse_penalty=0, semua kolom tile punya cost identik,
# dan scenes bisa di-reuse tanpa biaya → behavior = greedy.
print(f"reuse_penalty: {self.config['retrieval'].get('hungarian_reuse_penalty')}")

# 2. Apakah sim_matrix punya banyak ties?
# Kalau scenes >> sentences DAN tidak ada ties, Hungarian dan greedy bisa
# converge ke assignment yang sama (greedy juga optimal kalau resource >> demand).
print(f"sim_matrix shape: {sim_matrix.shape}")
print(f"sim_matrix unique values: {len(np.unique(sim_matrix.round(4)))}")

# 3. Apakah linear_sum_assignment beneran dipanggil?
# Print baris/kolom hasil:
row_idx, col_idx = linear_sum_assignment(cost_matrix)
print(f"row_idx: {row_idx}")
print(f"col_idx: {col_idx}")
print(f"col_idx % M (mapped): {[c % M for c in col_idx]}")
```

### Step 1.3: Diagnose & fix

Berdasarkan output Step 1.2, ada 3 kemungkinan:

**Diagnosis A: `reuse_penalty=0` di config.**
Fix: Set `hungarian_reuse_penalty: 0.2` di `configs/default.yaml` dan re-run.
Verifikasi assignment berubah.

**Diagnosis B: Greedy tidak reuse scenes (jadi Hungarian tidak punya kesempatan
re-allocate).**
Cek apakah `len(scenes) >> len(sentences)`. Kalau iya:
- Untuk skenario ini, Greedy sudah optimal (setiap sentence dapat scene
  unik karena scenes berlimpah)
- Hungarian == Greedy itu **mathematically expected**
- Ini BUKAN bug
- Dokumentasikan: "Hungarian degenerates to Greedy when scenes >> sentences
  and no scene reuse is needed"

**Diagnosis C: Bug di tile/cost matrix construction.**
Print intermediate values:
```python
print(f"K (number of tiles): {K}")
print(f"cost_matrix shape after tile: {cost_matrix.shape}")
print(f"cost_matrix[:, 0:5]: {cost_matrix[:5, 0:5]}")  # tile pertama
print(f"cost_matrix[:, M:M+5]: {cost_matrix[:5, M:M+5]}")  # tile kedua
# Tile kedua harus = tile pertama + reuse_penalty
```

Fix berdasarkan apa yang salah.

### Step 1.4: Verify fix

Re-run Hungarian config di 1 video. Pastikan:
- Assignment **berbeda** dari Greedy di minimal beberapa sentence
- Atau, kalau confirmed degenerate (Diagnosis B), dokumentasikan dan tambahkan
  log warning saat scenes >> sentences

### Deliverable Task 1
- File `scripts/debug_hungarian_vs_greedy.py`
- File `notes/hungarian_diagnosis.md` berisi:
  - Output dari step 1.1 (3 video)
  - Diagnosis (A/B/C)
  - Fix yang diaplikasikan (atau justifikasi kenapa tidak fix)
  - Hasil re-run setelah fix

---

## TASK 2: Tune DP jump_penalty

### Goal
Cari nilai `jump_penalty` yang memberi trade-off baik antara VisCoher (smoothness)
dan TempAcc (temporal accuracy). Saat ini DP terlalu agresif — VisCoher melonjak
tapi TempAcc anjlok.

### Step 2.1: Sweep jump_penalty values

Buat script `scripts/sweep_dp_jump_penalty.py`:

```python
"""
Sweep jump_penalty values for DP, run on 1 video, plot trade-off.
"""
import numpy as np
from pathlib import Path
import yaml
import subprocess
import json

JUMP_PENALTIES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
TEST_VIDEO = "video_1"  # ganti dengan path video yang available
ARM_NAME = "siglip_temporal_dp"


def run_with_jump_penalty(jp: float, video_id: str):
    """Run pipeline dengan jump_penalty tertentu, return metrics."""
    # Modifikasi config temporarily
    config_path = Path("configs/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    original_jp = config["retrieval"]["dp_jump_penalty"]
    config["retrieval"]["dp_jump_penalty"] = jp

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        # Run pipeline (sesuaikan dengan command kamu)
        result = subprocess.run(
            ["python", "-m", "src.run_pipeline", "--video", video_id, "--arm", ARM_NAME],
            capture_output=True, text=True, check=True
        )

        # Load results
        results_path = Path(f"results/{video_id}/{ARM_NAME}/metrics.json")
        with open(results_path) as f:
            metrics = json.load(f)
        return metrics
    finally:
        # Restore config
        config["retrieval"]["dp_jump_penalty"] = original_jp
        with open(config_path, "w") as f:
            yaml.dump(config, f)


def main():
    results = []
    for jp in JUMP_PENALTIES:
        print(f"\n=== Running with jump_penalty={jp} ===")
        metrics = run_with_jump_penalty(jp, TEST_VIDEO)
        results.append({
            "jump_penalty": jp,
            "clipscore": metrics.get("clipscore"),
            "temp_acc": metrics.get("temporal_acc_30s"),
            "vis_coher": metrics.get("visual_coherence_mean"),
        })

    # Print trade-off table
    print(f"\n{'JP':<8}{'CLIPScore':<12}{'TempAcc':<12}{'VisCoher':<12}")
    print("-" * 44)
    for r in results:
        print(f"{r['jump_penalty']:<8}{r['clipscore']:<12.4f}"
              f"{r['temp_acc']:<12.4f}{r['vis_coher']:<12.4f}")

    # Save raw
    with open("notes/dp_jump_penalty_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

**Note:** Sesuaikan command pipeline (`python -m src.run_pipeline ...`) dengan
yang ada di project. Kalau command-nya beda, adjust.

### Step 2.2: Analisa trade-off

Setelah sweep selesai, lihat tabel hasil. Cari "elbow point":
- VisCoher tetap meningkat dari Greedy (improvement preserved)
- TempAcc tidak turun lebih dari 10-15% dari Greedy baseline

**Greedy baselines (untuk referensi):**
- V1: TempAcc=0.714, VisCoher=0.697
- V2: TempAcc=1.000, VisCoher=0.763
- V3: TempAcc=1.000, VisCoher=0.670

**Target untuk DP yang sehat:**
- TempAcc ≥ 0.6 (V1) atau ≥ 0.85 (V2, V3) — penurunan max 15%
- VisCoher ≥ Greedy (improvement preserved)

### Step 2.3: Verify di 3 video

Setelah pilih `jump_penalty` baru, re-run full ablation di 3 video. Compare
dengan hasil sebelumnya.

**Expected pattern setelah tuning:**
- DP TempAcc tidak lagi anjlok ke 0.143 / 0.500
- DP VisCoher masih lebih tinggi dari Greedy (tapi mungkin tidak 0.975 lagi —
  yang itu memang artifact dari over-aggressive DP)
- DP CLIPScore stabil

### Deliverable Task 2
- File `scripts/sweep_dp_jump_penalty.py`
- File `notes/dp_jump_penalty_sweep.json` (raw sweep results)
- File `notes/dp_tuning_decision.md` berisi:
  - Tabel sweep results
  - Nilai jump_penalty yang dipilih dengan justifikasi
  - Hasil re-run 3 video dengan jump_penalty baru
  - Comparison: before tuning vs after tuning

### Update config

Setelah keputusan final, update `configs/default.yaml`:

```yaml
retrieval:
  dp_jump_penalty: <new_value>  # tuned from sweep
```

---

## CRITICAL CONSTRAINTS

1. **Don't add new features.** Plan ini purely debug + tune. Kalau ada
   improvement idea muncul, catat di notes tapi jangan implement.

2. **Don't modify metric implementations.** Metrics sekarang sudah jalan.
   Hands off.

3. **Don't run full 12-config ablation yet.** Cukup arm yang relevan (Greedy,
   Hungarian, DP) untuk debugging.

4. **Save all intermediate logs.** Print output, comparison tables, dan
   keputusan harus terdokumentasi di `notes/`. Kita perlu paper trail untuk
   thesis writeup.

5. **Stop dan report kalau ada hasil yang tidak terduga.** Misal:
   - Hungarian setelah fix menjadi sangat berbeda dari Greedy → bagus, lapor
   - DP setelah tuning tidak menunjukkan improvement coherence sama sekali
     → bug lain mungkin ada, lapor
   - Pipeline error/crash → lapor

---

## SUCCESS CRITERIA

Task selesai kalau:

**Hungarian:**
- [ ] Diagnosis jelas (A/B/C dari Step 1.3)
- [ ] Kalau bug: fixed dan assignment Hungarian berbeda dari Greedy
- [ ] Kalau degenerate: didokumentasikan dengan justifikasi matematis

**DP jump_penalty:**
- [ ] Sweep done, results saved
- [ ] New `jump_penalty` value chosen and committed to config
- [ ] Re-run di 3 video menunjukkan TempAcc tidak anjlok parah lagi
- [ ] VisCoher tetap > Greedy

**Documentation:**
- [ ] `notes/hungarian_diagnosis.md`
- [ ] `notes/dp_tuning_decision.md`
- [ ] `notes/dp_jump_penalty_sweep.json`

---

## END OF PROMPT

Mulai dari Task 1 (debug Hungarian). Setelah selesai dan dokumentasi lengkap,
baru mulai Task 2 (tune DP). Jangan kerjakan paralel — Task 1 mungkin reveal
isu yang ngubah cara approach Task 2.

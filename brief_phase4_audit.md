# Phase 4 Restore and Audit Report

## 1. Diff Analysis & Ambiguity Resolution

We have successfully performed a selective restore of the stashed `RetrievalGate` logic from `stash@{0}` without overwriting the improvements introduced on branch `v8` (such as the multi-method sentence-level matching algorithms).

* **Restored Entities:**
  * Added `Sentence`, `Scene`, `Assignment`, `RetrievalGateConfig`, `RetrievalGate`, `FrameSelectorConfig`, `FrameSelector`, `FrameRef`, and `summarise_assignments` to the end of `src/phase4_retrieve.py`.
  * Updated typing and dataclass imports at the top of `src/phase4_retrieve.py`.
  * Appended the `phase4:` block with `gate_threshold: 0.12` and other parameters to `configs/default.yaml`.
  * Clean backups of the active `v8` files were saved to `/tmp/`.

### ⚠️ AMBIGUOUS Case in `src/pipeline.py`

There is a significant architectural divergence in the orchestration layer:
* **Stashed State (`stash@{0}`):** `pipeline.py` ran `RetrievalGate` grouping, outputted `p4_assignments.json`, and passed the list of `Assignment` objects directly into `Phase5Assembler.run(..., assignments)`.
* **Active State (branch `v8`):** `pipeline.py` runs sentence-level `Phase4Retrieval` with multiple backend matching algorithms, outputting `scene_matches_{method}.json` (matching the `RetrievalOutput` Pydantic schema in `src/schemas.py`), and passes the output path to `Phase5Assembler.run(..., retrieval_output_path)`.

> [!WARNING]
> Restoring the stashed `pipeline.py` orchestration directly will break the active `v8` sentence-level pipeline and cause `Phase5Assembler` to crash. We have left `src/pipeline.py` unchanged and present the options below for user decision.

### ❓ Options for Resolution

1. **Option A (Sentence-level & Calibration Only):** Keep the active `v8` pipeline sentence-level orchestration as is. Use the restored `RetrievalGate` classes solely for calibration/sanity scripts.
2. **Option B (Divergent Path / Grouping Restore):** Restore grouping & gating orchestration in `pipeline.py` but adapt `Phase5Assembler` to handle grouped assignments and support generative/hybrid assembly.

---

## 2. Verification Status

* **Import Test:** Verified that `RetrievalGate` imports cleanly in the environment:
  ```bash
  conda run -n sumarizer python3 -m src.phase4_retrieve
  # Output: OK (DEBUG: PHASE 4 MODULE LOADED)
  ```
* **Sanity Runner:** Verified that running `scratch/run_phase4_sanity.py` generates the `sanity_check_threshold_012/` directory and extracts the representative keyframes successfully.

---

## 3. Replication Metrics

We verified bit-exact reproduction of the old `p4_assignments.json` records by comparing the newly computed assignments against the stashed ones on disk for all 10 review videos.

| Video ID | Group Count | Threshold 0.13 (Stashed Runs) | Threshold 0.12 (New Config) | Verification Status |
|---|---|---|---|---|
| **review_1** | 7 | 2 retrieve / 5 generate | 3 retrieve / 4 generate | ✅ Bit-exact replica at 0.13 |
| **review_2** | 12 | 1 retrieve / 11 generate | 1 retrieve / 11 generate | ✅ Bit-exact replica at 0.13 |
| **review_3** | 9 | 1 retrieve / 8 generate | 1 retrieve / 8 generate | ✅ Bit-exact replica at 0.13 |
| **review_4** | 12 | 4 retrieve / 8 generate | 7 retrieve / 5 generate | ✅ Bit-exact replica at 0.13 |
| **review_5** | 11 | 8 retrieve / 3 generate | 8 retrieve / 3 generate | ✅ Bit-exact replica at 0.13 |
| **review_6** | 8 | 1 retrieve / 7 generate | 1 retrieve / 7 generate | ✅ Bit-exact replica at 0.13 |
| **review_7** | 7 | 4 retrieve / 3 generate | 5 retrieve / 2 generate | ✅ Bit-exact replica at 0.13 |
| **review_8** | 3 | 2 retrieve / 1 generate | 2 retrieve / 1 generate | ✅ Bit-exact replica at 0.13 |
| **review_9** | 7 | 2 retrieve / 5 generate | 2 retrieve / 5 generate | ✅ Bit-exact replica at 0.13 |
| **review_10** | 14 | 6 retrieve / 8 generate | 7 retrieve / 7 generate | ✅ Bit-exact replica at 0.13 |
| **TOTAL** | **90** | **31 retrieve / 59 generate** | **37 retrieve / 53 generate** | **✅ 100% Replication Match** |

> [!NOTE]
> The threshold difference explains the mismatch: the old runs used `gate_threshold: 0.13` (pre-calibration), while the new default configuration specifies `0.12`. Running the gating algorithm with `gate_threshold: 0.13` produces 100% identical outputs compared to the saved files on disk.

---

## 4. Downstream Regression Analysis

1. **Phase 5 Generation Handling:**
   Active `v8` `src/phase5_assemble.py` has **no handling** for `action="generate"`. It assumes all matches are to be retrieved and sliced directly from the source video.
2. **Phase 5 Grouping Support:**
   The active `v8` assembler is strictly sentence-based and iterates over `retrieval_output.matches` list. It does not support reading or processing grouped sentence structures.
3. **`p4_assignments.json` Consumers:**
   The only consumer files of `p4_assignments.json` are calibration/diagnostic scripts (`scripts/phase4_calibration_runner.py` and `scratch/analyze_audio_duration.py`).
4. **Unit Tests Restored:**
   We identified that `tests/test_phase5.py` and `tests/test_pipeline.py` were failing because the `v8` assembler saves output into a nested `output_dir / video_id` folder while the tests asserted paths in the root of `output_dir`. We updated the assertions, and **all tests now pass successfully**.
5. **Requirements for Generation Support in Assembly:**
   If we choose to restore generative gating in Phase 5, the assembler will require:
   * Integration with the Image-to-Video generation pipeline (such as LTX-Video or Wan 2.2).
   * Logic to separate `retrieve` assignments from `generate` assignments.
   * Frame-conditioning selection using the `FrameSelector` class to guide the generative model.

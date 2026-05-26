# Brief: Phase 4 RetrievalGate Selective Restore + Post-Restore Audit

**Task type:** Execution (selective code restoration + verification + audit).
**Goal:** Restore `RetrievalGate` (grouping + decision gate) ke active branch v8 dari `stash@{0}`, **secara selektif** (cuma yang hilang, jangan overwrite improvement v8 yang lain). Verify via existing calibration script. Lalu audit codebase untuk catch regression lain yang mungkin ada.

## Context

Audit sebelumnya (`phase4_audit_report.md`) confirmed:
- `RetrievalGate` class + config + pipeline integration exists di `stash@{0}` tapi missing di branch v8
- Branch v8 cuma punya sentence-by-sentence retrieval (no grouping, no decision gate)
- Old output data (`p4_assignments.json`) dari run 13 May masih ada — bisa jadi reference untuk verify post-restore correctness

**Critical concern:** User ga inget kenapa RetrievalGate di-revert. Mungkin ga sengaja. Tapi v8 mungkin punya improvement lain (bug fixes, new algorithms) yang harus DIPERTAHANKAN. Blind `git stash pop` bisa rusak ini.

## Stage 1: Selective restore

### Step 1.1: Pre-restore inventory

Sebelum apa-apa, dump current state of relevant files:
```bash
cp src/phase4_retrieve.py /tmp/v8_phase4_retrieve.py
cp src/pipeline.py /tmp/v8_pipeline.py
cp configs/default.yaml /tmp/v8_default.yaml
# Save schemas too if exists
[ -f src/schemas.py ] && cp src/schemas.py /tmp/v8_schemas.py
```

Document: list semua classes/functions yang ada di v8 `phase4_retrieve.py` (jangan modify).

### Step 1.2: Identify what to restore

Dari `stash@{0}`, identify komponen yang **missing di v8**:

1. **Classes yang harus ditambah ke `src/phase4_retrieve.py`:**
   - `Sentence` dataclass/model
   - `Scene` dataclass/model
   - `Assignment` dataclass/model
   - `RetrievalGateConfig`
   - `RetrievalGate` (main class with grouping + decision gate logic)
   
2. **Schemas yang mungkin perlu ditambah** ke `src/schemas.py` (kalau Assignment/Group schemas di sini)

3. **Config block** di `configs/default.yaml`:
   ```yaml
   phase4:
     gate_threshold: 0.12
     extend_epsilon: 0.03
     max_group_size: 5
     join_sep: " "
     temporal_sigma: 30.0
     enable_temporal_prior: true
   ```

4. **Pipeline orchestration** di `src/pipeline.py`:
   - Code yang call `RetrievalGate` dan write `p4_assignments.json`

### Step 1.3: Identify what NOT to restore

**JANGAN restore atau overwrite:**
- Existing classes di v8 yang ada di kedua versi (e.g. `SigLIP2DirectRetrieval`, `CaptionCosineRetrieval`, `KeyframeExtractor`)
- Matching algorithms baru di v8 yang ga ada di stash (misal `cv_align`, `ccma` — cek apakah ini ada di stash atau cuma di v8)
- ARM_CONFIGS expansion di v8 (kalau v8 punya lebih banyak ablation arms)
- Phase 1-3 changes (kalau ada di stash, jangan touch)

### Step 1.4: Diff first, apply selectively

Untuk setiap file yang akan di-modify, generate diff dulu:
```bash
git diff stash@{0}:src/phase4_retrieve.py src/phase4_retrieve.py > /tmp/p4_diff.txt
git diff stash@{0}:src/pipeline.py src/pipeline.py > /tmp/pipeline_diff.txt
git diff stash@{0}:configs/default.yaml configs/default.yaml > /tmp/config_diff.txt
```

Review diff manually (in your output to user). Identify which hunks are:
- **MUST restore** (RetrievalGate logic missing from v8)
- **MUST NOT restore** (v8 has improvements stash doesn't have)
- **AMBIGUOUS** (both versions differ, unclear which is better)

**Stop and report ambiguous cases.** Don't make architectural decisions about which version "wins" — user decides.

### Step 1.5: Apply restoration

After diff review and resolving ambiguous cases:
- Append missing classes to `src/phase4_retrieve.py` (don't overwrite existing classes)
- Add missing schemas to `src/schemas.py`
- Add `phase4:` block to `configs/default.yaml` (append, don't replace)
- Add orchestration to `src/pipeline.py`

After each file modification, show a brief summary of what was added.

## Stage 2: Verification

### Step 2.1: Import test
```bash
python -c "from src.phase4_retrieve import RetrievalGate, RetrievalGateConfig; print('OK')"
```

Should succeed without ImportError.

### Step 2.2: Run sanity script
Run `scratch/run_phase4_sanity.py` atau `scripts/phase4_calibration_runner.py` (yang masih ada di repo). Should generate `p4_assignments.json` output successfully.

### Step 2.3: Compare with old output

Output baru harus reproducible dari old output di `data/intermediate/review_*/p4_assignments.json` (assuming same input dan same config).

For each video di `data/intermediate/`:
- Old `p4_assignments.json` exists?
- New run produces same number of groups? Same group-to-scene assignments? Same actions?
- If different, report differences — kemungkinan ada bug residual atau improvement v8 yang affect output.

**Acceptance criteria:** Old vs new output should be **identical or near-identical** (small floating-point differences OK). If significantly different, STOP and report.

## Stage 3: Post-restore audit

Setelah restore success, audit codebase untuk regression LAIN yang mungkin ada.

### Search for:
1. **Phase 5 generate handling**: ada ga code yang handle `action="generate"`? Atau cuma `action="retrieve"`?
2. **Phase 6 (assembly) supports groups**: code di `src/phase5_assemble.py` (current name) bisa baca group structure, atau cuma per-sentence?
3. **`p4_assignments.json` consumer**: siapa yang baca file ini? Phase 5? Phase 6? Atau cuma calibration script?
4. **Test files broken**: handoff v4 mention "broken tests since Phase 4 refactor" — cek `tests/` dir untuk failing imports

Report findings, JANGAN fix. User decide priority untuk Phase 5 build.

## What to report back

Markdown report:

```
## Stage 1: Restore
- Files modified: <list>
- What was added: <summary per file>
- Ambiguous decisions encountered: <list, with how resolved>

## Stage 2: Verification
- Import test: PASS/FAIL
- Sanity script: PASS/FAIL (output: <path>)
- Comparison with old output:
  - Identical: <N videos>
  - Different: <N videos with details>

## Stage 3: Post-restore audit
- Phase 5 generate handling: <status>
- Phase 6 group support: <status>
- p4_assignments.json consumers: <list>
- Broken tests: <list>

## Summary
- Restore success: yes/no
- Remaining gaps blocking Phase 5 production
- Recommended next steps for user
```

## Hard rules

- **Selective merge, JANGAN `git stash pop`.** Apply changes selectively berdasarkan diff review.
- **Backup current state before any modification** (Step 1.1).
- **Show diffs before applying** — user perlu inspect kalau ada doubt.
- **STOP and ask** kalau ketemu ambiguous case (e.g. v8 punya class yang berbeda implementasi dari stash). Jangan auto-decide.
- **Verify after each stage.** Don't proceed to Stage 2 if Stage 1 has unresolved issues.
- **Don't touch Phase 1-3 code** unless explicitly part of restoration scope.

## Anti-hallucination

- Quote file paths, line numbers, dan actual code snippets
- Kalau diff command fails atau stash inaccessible, report error verbatim
- Comparison numbers (e.g. "5 videos identical") harus dari actual file inspection, jangan estimate
- Kalau old `p4_assignments.json` ga ada di salah satu video, bilang "missing" — jangan asumsi

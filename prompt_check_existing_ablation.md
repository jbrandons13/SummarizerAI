# Prompt: Inspect Existing Ablation Results

**Task:** Determine whether `final_ablation_results.csv` exists, its contents, and whether it reflects pre-fix or post-fix CCMA state. Report only — do NOT re-run anything, do NOT modify files.

## Steps

```bash
# 1. Find the canonical results CSV
ls -la results/ 2>/dev/null
find . -name "final_ablation_results.csv" 2>/dev/null
find . -name "*ablation*results*" -type f 2>/dev/null | head -20
find . -name "*ablation*" -name "*.csv" 2>/dev/null | head -20

# 2. If found, show structure
RESULTS_CSV="results/final_ablation_results.csv"
if [ -f "$RESULTS_CSV" ]; then
  echo "=== HEADER ==="
  head -1 "$RESULTS_CSV"
  echo "=== ROW COUNT ==="
  wc -l "$RESULTS_CSV"
  echo "=== UNIQUE ARMS ==="
  python -c "import pandas as pd; df = pd.read_csv('$RESULTS_CSV'); print(df['arm'].value_counts() if 'arm' in df.columns else df.columns.tolist())"
  echo "=== UNIQUE VIDEOS ==="
  python -c "import pandas as pd; df = pd.read_csv('$RESULTS_CSV'); print(df['video_id'].nunique() if 'video_id' in df.columns else 'no video_id column')"
fi

# 3. Check file modification date vs CCMA fix commit date
stat results/final_ablation_results.csv 2>/dev/null | grep -E "Modify|Change"
git log -1 --format="%ai %s" 2c51ea1  # CCMA fix commit
git log --oneline -- results/final_ablation_results.csv 2>/dev/null | head -5

# 4. Check if there are CCMA-specific result files (intermediate)
ls data/intermediate/review_1/ | grep -iE "ccma|scene_match" | head -10
find data/intermediate -name "scene_matches_*_ccma.json" 2>/dev/null | head -10
find data/intermediate -name "eval_results_*_ccma*" 2>/dev/null | head -10

# 5. Check ccma_fix_report.md (per ccma_fix_prompt.md, this should exist if fix was executed)
ls -la ccma_fix_report.md 2>/dev/null
cat ccma_fix_report.md 2>/dev/null | head -100

# 6. Look for any tabular summary docs (markdown reports comparing arms)
find . -name "*.md" -exec grep -l "caption_temporal_ccma\|siglip_temporal_ccma" {} \; 2>/dev/null | head -10
```

## Report format

```
## Existing Ablation Results

**final_ablation_results.csv:**
- Path: <full path, or "NOT FOUND">
- Last modified: <date>
- Total rows: <n>
- Unique arms present: <list with counts>
- Unique videos: <n>
- Sample row (first non-header line): <paste>

**CCMA fix commit date:** <date>
**CSV last modified vs CCMA fix:** BEFORE / AFTER / SAME DAY

**CCMA intermediate cache files:**
- scene_matches_*_ccma.json: <count, sample paths>
- eval_results_*_ccma*: <count, sample paths>
- Last modified dates: <range>

**ccma_fix_report.md exists?** YES / NO
**If YES, first 100 lines:**
<paste>

**Other markdown reports mentioning CCMA arms:**
<list with paths and 1-line description each>

## Key answers

1. Does `final_ablation_results.csv` exist with usable data?
   YES (post-fix) / YES (pre-fix, may be stale) / NO

2. Does the CSV contain BOTH caption_temporal_ccma AND siglip_temporal_ccma rows for all 10 videos?
   YES / PARTIAL / NO
   If PARTIAL: list which arms × videos are missing

3. Does the CSV contain grouping_gate as one of the arms?
   YES / NO

4. If YES to Q3: how does grouping_gate compare numerically to ccma arms?
   Paste mean values per arm for key metrics (clipscore_mean, scene_diversity, viscoher_strict, max_consecutive_reuse if present).
   DO NOT interpret. Just report numbers.

5. Was the fix from ccma_fix_prompt.md ever executed?
   Evidence: presence of ccma_fix_report.md, intermediate cache dates post-fix, etc.
```

DO NOT re-run ablation. DO NOT modify CSVs. DO NOT interpret quality. Report what files contain.

End of prompt.

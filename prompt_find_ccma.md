# Prompt: Locate CCMA Implementation

**Task:** Find any existing CCMA (Cosine Composite Min-max Aggregation, or Min-Max per-sentence normalization) ranking implementation in the codebase. Do NOT implement anything new. Report findings only.

## Steps

Run these searches and report results verbatim:

```bash
# 1. Direct CCMA references
grep -rn "ccma\|CCMA" src/ --include="*.py"
grep -rn "ccma\|CCMA" configs/ --include="*.yaml"

# 2. Min-Max normalization patterns
grep -rn "MinMaxScaler\|minmax_scale\|min_max\|MinMax" src/ --include="*.py"
grep -rn "(row.max\|row.min\|.max(axis\|.min(axis)" src/ --include="*.py" | head -20

# 3. Per-sentence normalization patterns (likely in retrieval/Phase 4 code)
ls src/phase4/ src/retrieval/ 2>/dev/null
grep -rn "normalize\|normalise" src/phase4/ src/retrieval/ 2>/dev/null --include="*.py" | head -20

# 4. Git history
git log --all --oneline | grep -iE "ccma|minmax|normaliz" | head -20
git branch -a | head -20

# 5. Old/deprecated/legacy folders
find src/ -type d -iname "*old*" -o -iname "*legacy*" -o -iname "*deprecated*" -o -iname "*v1*" -o -iname "*archive*" 2>/dev/null
```

## Report format

```
## CCMA Search Results

**Direct CCMA matches:**
<paste grep output, or "NONE">

**Min-Max patterns:**
<paste grep output, or "NONE">

**Per-sentence normalization in retrieval code:**
<paste grep output, or "NONE">

**Git history mentions:**
<paste log output, or "NONE">

**Legacy/old folders:**
<paste find output, or "NONE">

## Files identified as potentially CCMA-related

For each candidate file, show:
- Path
- Function name(s) that look CCMA-like
- 10-20 line code excerpt of the relevant function

## Verdict

One of:
- FOUND COMPLETE: working CCMA implementation exists at <path>, function `<name>`, looks usable as-is
- FOUND PARTIAL: code fragments exist but not a complete CCMA pipeline (describe what's missing)
- NOT FOUND: no CCMA-related code in current codebase or git history

DO NOT implement new CCMA code. DO NOT modify any files. Report only.
```

End of prompt.

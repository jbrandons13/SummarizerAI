# Prompt: Confirm CCMA Usage History

**Task:** Determine which Phase 4 retrieval method was used in past experiments where "CCMA gave better output." Report findings only — do NOT implement, modify, or run anything.

## Steps

```bash
# 1. List all available Phase 4 methods
grep -n "method_configs\|method_map\|RETRIEVAL_METHODS" src/phase4_retrieve.py | head -10
sed -n '970,990p' src/phase4_retrieve.py    # show the method registry around line 977

# 2. Check current default method
grep -n "method\|matching_algo" configs/default.yaml | head -10
grep -n "method\|matching_algo" configs/smoke_test.yaml | head -10

# 3. Check git history for method/config changes
git log --all --oneline -- configs/default.yaml | head -20
git log --all --oneline -- src/phase4_retrieve.py | head -20

# 4. Look for any test/experiment outputs that mention CCMA
find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.json" \) -exec grep -l -i "ccma" {} \; 2>/dev/null | head -10

# 5. Check archive/ folder for old configs
ls -la archive/ 2>/dev/null
find archive/ -name "*.yaml" -o -name "*.py" 2>/dev/null | head -20

# 6. Check run_ccma_tests.py — what method does it actually call?
cat run_ccma_tests.py 2>/dev/null | head -80
```

## Report format

```
## CCMA Usage History

**Available Phase 4 methods (from method registry):**
<paste list, e.g.:>
- siglip_direct: (siglip, False, "greedy")
- siglip_temporal: (siglip_temporal, True, "greedy")
- siglip_temporal_ccma: (siglip_temporal, True, "ccma")
- caption_cosine: ...
- caption_temporal: ...
- caption_temporal_ccma: ...
- grouping_gate: ...
- [others]

**Current default method in configs/default.yaml:**
<paste value>

**Smoke test method:**
<paste value>

**Git log: when was each method introduced / last modified:**
<list with dates>

**Files mentioning CCMA results:**
<list with brief excerpt of context>

**run_ccma_tests.py contents (first 80 lines):**
<paste>

## Key answers

1. What does `matching_algo == "ccma"` actually do?
   <one-sentence explanation, citing line numbers in phase4_retrieve.py>

2. Are there TWO different "CCMA"s in the codebase, or just one?
   - "CCMA" as assignment algorithm (capacity-constrained DP) — YES/NO
   - "CCMA" or "Min-Max" as ranking normalization method — YES/NO
   - If both exist: are they used together or separately?

3. Based on git history and run_ccma_tests.py, what method was likely used in past "CCMA experiments"?
   - Best guess: <method name>
   - Confidence: HIGH / MEDIUM / LOW
   - Evidence: <bullet points>

4. What's the OPPOSITE/baseline of CCMA in this codebase?
   - i.e., if I want to ablate "with CCMA" vs "without CCMA", what's the non-CCMA variant?
   - <method name and brief description>
```

DO NOT implement anything. DO NOT modify files. Report only.

End of prompt.

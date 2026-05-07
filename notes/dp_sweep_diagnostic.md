# DP Sweep Diagnostic Report

## Test 1: Extreme Value (backward_penalty = 0.0)

- **Assignment**: [2, 4, 11, 1, 1, 4]
- **Reuse Rate**: 0.333
- **TempAcc (15s)**: 0.333
- **VisCoher**: 0.770

## Test 2: Negative Value (backward_penalty = -1.0)

- **Assignment**: [38, 11, 7, 4, 3, 1]
- **Reuse Rate**: 0.000
- **TempAcc (15s)**: 0.333
- **VisCoher**: 0.687

## Test 3: Print Verification

(Check terminal output for 'DIAGNOSTIC: DP called with backward_penalty=...') 

## Verdict

**H1 CONFIRMED**: The assignment changed when moving to a reward for backward jumps. This means the parameter is wired correctly, and the previous 'identical' results were due to the semantic signal being too strong for the 0.05-0.5 penalty range to overcome.

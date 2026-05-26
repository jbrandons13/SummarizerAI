import csv
import math

with open('results/final_ablation_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

metrics = [
    'clipscore_mean',
    'blipscore_mean',
    'llm_judge_coherence',
    'llm_judge_consistency',
    'llm_judge_quality',
    'scene_diversity'
]

print(f'Total rows read: {len(rows)}')

for m in metrics:
    vals = []
    for r in rows:
        val_str = r.get(m, '')
        if val_str and val_str != 'NaN':
            try:
                vals.append(float(val_str))
            except ValueError:
                pass
    if not vals:
        print(f'{m}: No valid numerical values')
        continue
    
    m_min = min(vals)
    m_max = max(vals)
    m_mean = sum(vals) / len(vals)
    m_variance = sum((x - m_mean) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0
    m_std = math.sqrt(m_variance)
    
    print(f'Metric: {m}')
    print(f'  Range observed: ({m_min:.6f}, {m_max:.6f})')
    print(f'  Mean: {m_mean:.6f}')
    print(f'  Std Dev: {m_std:.6f}')

print('\n=== BLIPSCORE DETAILED ANALYSIS ===')
add_on = ['raw_hybrid_retrieval_ccma_grouping_gating_prompt_expanded', 'raw_hybrid_retrieval_ccma_grouping_gating_cascade_verified']
orig_blips = []
prompt_expanded_blips = []
cascade_verified_blips = []

for r in rows:
    arm = r['arm']
    val_str = r.get('blipscore_mean', '')
    if val_str and val_str != 'NaN':
        val = float(val_str)
        if arm == 'raw_hybrid_retrieval_ccma_grouping_gating_prompt_expanded':
            prompt_expanded_blips.append(val)
        elif arm == 'raw_hybrid_retrieval_ccma_grouping_gating_cascade_verified':
            cascade_verified_blips.append(val)
        else:
            orig_blips.append(val)

if orig_blips:
    print(f'Original 16 arms BLIPScore range: ({min(orig_blips):.6f}, {max(orig_blips):.6f})')
if prompt_expanded_blips:
    print(f'Prompt expanded BLIPScore range: ({min(prompt_expanded_blips):.6f}, {max(prompt_expanded_blips):.6f})')
if cascade_verified_blips:
    print(f'Cascade verified BLIPScore range: ({min(cascade_verified_blips):.6f}, {max(cascade_verified_blips):.6f})')

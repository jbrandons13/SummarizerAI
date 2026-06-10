import json
import os
import torch
import sys
import hashlib

sys.modules['gptqmodel'] = None
sys.modules['awq'] = None
sys.modules['bitsandbytes'] = None
import peft.import_utils
peft.import_utils.is_auto_awq_available = lambda: False
peft.import_utils.is_gptqmodel_available = lambda: False
peft.import_utils.is_auto_gptq_available = lambda: False

sys.path.append(os.path.abspath("."))
from pipeline.facet.scoring_wrap import ScoringWrap

def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]

def main():
    stamp = "G0_A0_geology"
    run_dir = os.path.join("runs", stamp)
    records_file = os.path.join(run_dir, "records.jsonl")
    
    records = []
    with open(records_file, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    # Need storyboard to get the actual text prompt for clip_t
    video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    sb_path = f"data/intermediate/{video}/phase4/storyboard.json"
    with open(sb_path, "r") as f:
        sb = json.load(f)["shots"]
        
    text_map = {shot["shot_id"]: shot["visual_description"] for shot in sb}
    ref_path = "runs/geology/reference.png"
    
    if not os.path.exists(ref_path):
        print(f"Warning: {ref_path} not found.")
        ref_path = None
        
    scorer = ScoringWrap()
    
    # Pre-find w=0.0 paths
    w0_paths = {}
    for r in records:
        w = r["knobs"]["w"]
        if abs(w - 0.0) < 0.01:
            w0_paths[r["shot_id"]] = r["paths"]["image"]
            
    out_records = []
    for r in records:
        w = r["knobs"]["w"]
        img_path = r["paths"]["image"]
        w0_path = w0_paths.get(r["shot_id"])
        
        text_prompt = text_map.get(r["shot_id"], "")
        
        global_concept = "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones"
        scores = scorer.score_shot(img_path, w0_path, ref_path, text_prompt, global_concept)
        r["metrics"] = scores
        out_records.append(r)
        
    # Write scored records
    scored_file = os.path.join(run_dir, "scored_records.jsonl")
    with open(scored_file, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
            
    # Adaptive logic
    tau = 0.70
    W_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    
    # Organize by shot
    shot_data = {}
    for r in out_records:
        sid = r["shot_id"]
        w = r["knobs"]["w"]
        if sid not in shot_data:
            shot_data[sid] = {}
        shot_data[sid][w] = r
        
    print("\n## Reproduction Table: DACA on Geology")
    print("\n### 1. Fixed-scale frontier")
    print("| w | c_bar (sim to w=0) | mean_concept (CLIP) | ref_sim (diag) |")
    print("|---|---|---|---|")
    
    means_fixed = {}
    for w in [0.0] + W_grid:
        c_s_vals = [r["metrics"]["c_s"] for sid, d in shot_data.items() for rw, r in d.items() if abs(rw - w) < 0.01]
        concept_vals = [r["metrics"]["mean_concept"] for sid, d in shot_data.items() for rw, r in d.items() if abs(rw - w) < 0.01]
        ref_vals = [r["metrics"]["ref_sim"] for sid, d in shot_data.items() for rw, r in d.items() if abs(rw - w) < 0.01]
        
        c_bar = sum(c_s_vals) / len(c_s_vals) if c_s_vals else float('nan')
        mean_concept = sum(concept_vals) / len(concept_vals) if concept_vals else float('nan')
        ref_sim = sum(ref_vals) / len(ref_vals) if ref_vals else float('nan')
        means_fixed[w] = (c_bar, mean_concept, ref_sim)
        
        print(f"| {w:.2f} | {c_bar:.4f} | {mean_concept:.4f} | {ref_sim:.4f} |")
        
    print("\n### 2. Adaptive Selection (tau = 0.70)")
    adaptive_c = []
    adaptive_concept = []
    for sid, d in shot_data.items():
        w_star = min(W_grid)
        valid_ws = [w for w in W_grid if w in d and d[w]["metrics"]["c_s"] >= tau]
        if valid_ws:
            w_star = max(valid_ws)
        
        adaptive_c.append(d[w_star]["metrics"]["c_s"])
        adaptive_concept.append(d[w_star]["metrics"]["mean_concept"])
        
    ad_c_bar = sum(adaptive_c) / len(adaptive_c)
    ad_concept = sum(adaptive_concept) / len(adaptive_concept)
    
    print(f"**Adaptive (tau=0.70)**: c_bar = {ad_c_bar:.4f}, mean_concept = {ad_concept:.4f}")
    
    # Check qualitative reproduction criteria based on primary axes (mean_concept, c_bar)
    best_fixed = min([w for w in W_grid if 0.3 <= w <= 0.5], key=lambda w: abs(means_fixed[w][1] - ad_concept))
    bf_c_bar, bf_concept, _ = means_fixed[best_fixed]
    
    print(f"\n**Qualitative check:**")
    print(f"Nearest fixed scale (w={best_fixed}): c_bar = {bf_c_bar:.4f}, mean_concept = {bf_concept:.4f}")
    print(f"Adaptive advantage: c_bar +{ad_c_bar - bf_c_bar:.4f} at mean_concept diff {ad_concept - bf_concept:.4f}")
    if (ad_c_bar >= bf_c_bar + 0.08) and abs(ad_concept - bf_concept) <= 0.02:
        print("-> QUALITATIVE REPRODUCTION: PASS")
    else:
        print("-> QUALITATIVE REPRODUCTION: FAIL")

    print("\n### 3. Post-hoc frontier (tau sweep)")
    print("| tau | c_bar | mean_concept |")
    print("|---|---|---|")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        ac, ar = [], []
        for sid, d in shot_data.items():
            w_star = min(W_grid)
            valid_ws = [w for w in W_grid if w in d and d[w]["metrics"]["c_s"] >= t]
            if valid_ws:
                w_star = max(valid_ws)
            ac.append(d[w_star]["metrics"]["c_s"])
            ar.append(d[w_star]["metrics"]["mean_concept"])
        tc = sum(ac)/len(ac) if ac else float('nan')
        tr = sum(ar)/len(ar) if ar else float('nan')
        print(f"| {t:.2f} | {tc:.4f} | {tr:.4f} |")


if __name__ == "__main__":
    main()

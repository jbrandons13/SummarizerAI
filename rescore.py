import json, os
from pipeline.facet.scoring_wrap import ScoringWrap

def get_hash(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]

def main():
    video = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    sb_path = f"data/intermediate/{video}/phase4/storyboard.json"
    with open(sb_path, "r") as f:
        sb = json.load(f)["shots"]
        
    text_map = {shot["shot_id"]: shot["visual_description"] for shot in sb}
    ref_path = "runs/geology/reference.png"
    global_concept = "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones"
    
    scorer = ScoringWrap()
    
    # We will score runs/geology/sweep/
    W_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    shot_ids = [f"shot_{i:03d}" for i in range(1, 15)]
    
    shot_data = {}
    for sid in shot_ids:
        shot_data[sid] = {}
        text_prompt = text_map.get(sid, "")
        
        for w in [0.0] + W_grid:
            w0_path = f"runs/geology/sweep/{sid}_w0.0.png"
            img_path = f"runs/geology/sweep/{sid}_w{w:.1f}.png"
            if w == 0.0 or w == 1.0: img_path = f"runs/geology/sweep/{sid}_w{w:.1f}.png" # Some are w0.0 some might be missing
            
            if not os.path.exists(img_path):
                img_path = f"runs/geology/sweep/{sid}_w{w:.2f}.png" # Try w0.20 format if needed, but it's w0.2
            
            if os.path.exists(img_path):
                scores = scorer.score_shot(img_path, w0_path, ref_path, text_prompt, global_concept)
                shot_data[sid][w] = {"metrics": scores}

    print("\n## Reproduction Table: DACA on Geology (Re-scored thesis images)")
    print("\n### 1. Fixed-scale frontier")
    print("| w | c_bar (sim to w=0) | mean_concept (CLIP) | ref_sim (diag) |")
    print("|---|---|---|---|")
    
    means_fixed = {}
    for w in [0.0] + W_grid:
        c_s_vals = [d[w]["metrics"]["c_s"] for sid, d in shot_data.items() if w in d]
        concept_vals = [d[w]["metrics"]["mean_concept"] for sid, d in shot_data.items() if w in d]
        ref_vals = [d[w]["metrics"]["ref_sim"] for sid, d in shot_data.items() if w in d]
        
        c_bar = sum(c_s_vals) / len(c_s_vals) if c_s_vals else float('nan')
        mean_concept = sum(concept_vals) / len(concept_vals) if concept_vals else float('nan')
        ref_sim = sum(ref_vals) / len(ref_vals) if ref_vals else float('nan')
        means_fixed[w] = (c_bar, mean_concept, ref_sim)
        
        print(f"| {w:.2f} | {c_bar:.4f} | {mean_concept:.4f} | {ref_sim:.4f} |")
        
    tau = 0.70
    print(f"\n### 2. Adaptive Selection (tau = {tau})")
    adaptive_c = []
    adaptive_concept = []
    for sid, d in shot_data.items():
        w_star = min(W_grid)
        valid_ws = [w for w in W_grid if w in d and d[w]["metrics"]["c_s"] >= tau]
        if valid_ws:
            w_star = max(valid_ws)
        
        if w_star in d:
            adaptive_c.append(d[w_star]["metrics"]["c_s"])
            adaptive_concept.append(d[w_star]["metrics"]["mean_concept"])
        
    ad_c_bar = sum(adaptive_c) / len(adaptive_c) if adaptive_c else float('nan')
    ad_concept = sum(adaptive_concept) / len(adaptive_concept) if adaptive_concept else float('nan')
    
    print(f"**Adaptive (tau={tau})**: c_bar = {ad_c_bar:.4f}, mean_concept = {ad_concept:.4f}")
    
    best_fixed = min([w for w in W_grid if 0.3 <= w <= 0.5], key=lambda w: abs(means_fixed[w][1] - ad_concept))
    bf_c_bar, bf_concept, _ = means_fixed[best_fixed]
    
    print(f"\n**Qualitative check:**")
    print(f"Nearest fixed scale (w={best_fixed}): c_bar = {bf_c_bar:.4f}, mean_concept = {bf_concept:.4f}")
    print(f"Adaptive advantage: c_bar +{ad_c_bar - bf_c_bar:.4f} at mean_concept diff {ad_concept - bf_concept:.4f}")
    if (ad_c_bar >= bf_c_bar + 0.08) and abs(ad_concept - bf_concept) <= 0.02:
        print("-> QUALITATIVE REPRODUCTION: PASS")
    else:
        print("-> QUALITATIVE REPRODUCTION: FAIL")

if __name__ == "__main__":
    main()

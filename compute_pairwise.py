import os
import sys
import json
import numpy as np
from pipeline.facet.scoring_wrap import ScoringWrap
from scipy.stats import spearmanr

def get_concept_map(video):
    if video == "geo":
        path = "data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4/storyboard.json"
    else:
        path = "data/intermediate/2D7hZpIYlCA_hydrologic-carbon-cycles-crash-course-ecology/phase4/storyboard.json"
        
    with open(path, "r") as f:
        sb = json.load(f)["shots"]
        
    cmap = {}
    for s in sb:
        cmap[s["shot_id"]] = s["topic_tag"]
        
    return cmap

def main():
    scorer = ScoringWrap()
    
    geo_cmap = get_concept_map("geo")
    eco_cmap = get_concept_map("eco")
    
    def get_pairs(cmap, exclude=[]):
        tag2shots = {}
        for s, tag in cmap.items():
            if s in exclude: continue
            if tag not in tag2shots: tag2shots[tag] = []
            tag2shots[tag].append(s)
            
        pairs = []
        for tag, shots in tag2shots.items():
            if len(shots) >= 2:
                for i in range(len(shots)):
                    for j in range(i+1, len(shots)):
                        pairs.append((shots[i], shots[j]))
        return pairs
        
    geo_pairs = get_pairs(geo_cmap)
    eco_pairs = get_pairs(eco_cmap, exclude=["shot_011"])
    
    sweeps = [
        {
            "name": "Geology Legacy",
            "pairs": geo_pairs,
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "dir_fn": lambda s, w: f"runs/geology/sweep/collapse_evidence/{s}_w0.0.png" if w == 0.0 else next((m["path"] for m in legacy_man if m["shot_id"] == s and abs(m["weight"] - w) < 0.01), None)
        },
        {
            "name": "Geology Collided",
            "pairs": geo_pairs,
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "dir_fn": lambda s, w: f"runs/G0_A0_geology_collided/images/A0/w{w:.2f}/{s}.png"
        },
        {
            "name": "Geology v2",
            "pairs": geo_pairs,
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "dir_fn": lambda s, w: f"runs/G0_A0_geology/images/A0/w{w:.2f}/{s}.png"
        },
        {
            "name": "Ecology v2",
            "pairs": eco_pairs,
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "dir_fn": lambda s, w: f"runs/G0_A0_ecology/images/A0/w{w:.2f}/{s}.png"
        }
    ]
    
    with open("runs/geology/sweep/manifest.json", "r") as f:
        legacy_man = json.load(f)
        
    print("### Pairwise Evaluation Results")
    
    output_md = ["\n### Final G0 Addition: Pairwise Alignment"]
    output_md.append("The pre-registered axis-promotion rule: if `Spearman(w, pairwise) >= 0.8` AND `range(pairwise) >= 0.05` for $w>0$, pairwise is promoted to the primary consistency axis for Stages 2-6. The ecology `mc` table (non-monotone, range 0.016) motivated this rule.")
    
    for sweep in sweeps:
        name = sweep["name"]
        pairs = sweep["pairs"]
        dir_fn = sweep["dir_fn"]
        weights = sweep["weights"]
        
        print(f"\n{name} ({len(pairs)} pairs)")
        output_md.append(f"\n**{name}** ({len(pairs)} pairs)")
        output_md.append("| w | pairwise |")
        output_md.append("|---|---|")
        
        pw_scores = []
        for w in weights:
            sims = []
            for s1, s2 in pairs:
                p1 = dir_fn(s1, w)
                p2 = dir_fn(s2, w)
                if p1 and p2 and os.path.exists(p1) and os.path.exists(p2):
                    emb1 = scorer.embed_dino(p1)
                    emb2 = scorer.embed_dino(p2)
                    sims.append(float((emb1 * emb2).sum().item()))
            avg_sim = sum(sims)/len(sims) if sims else float("nan")
            print(f"w={w:.1f} | pairwise={avg_sim:.4f}")
            output_md.append(f"| {w:.2f} | {avg_sim:.4f} |")
            if w > 0:
                pw_scores.append(avg_sim)
                
        # Calculate spearman and range
        w_nz = [w for w in weights if w > 0]
        sp_corr, _ = spearmanr(w_nz, pw_scores)
        pw_range = max(pw_scores) - min(pw_scores)
        
        promoted = (sp_corr >= 0.8) and (pw_range >= 0.05)
        
        stat_str = f"Spearman: {sp_corr:.4f}, Range: {pw_range:.4f} -> Promoted: {promoted}"
        print(stat_str)
        output_md.append(f"*{stat_str}*")

    with open("runs/G0_REPORT.md", "a") as f:
        f.write("\n".join(output_md) + "\n")

if __name__ == "__main__":
    main()

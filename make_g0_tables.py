import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pipeline.facet.scoring_wrap import ScoringWrap

def interpolate_c_bar(mc_target, fixed_points):
    # fixed_points is a list of (mc, c_bar) tuples
    # sort by mc
    pts = sorted(fixed_points, key=lambda x: x[0])
    
    # if target is outside, return the nearest bound's c_bar (or extrapolate? linear interp)
    mcs = [p[0] for p in pts]
    cbs = [p[1] for p in pts]
    
    return np.interp(mc_target, mcs, cbs)

def main():
    scorer = ScoringWrap()
    
    sweeps = [
        {
            "name": "Geology Legacy",
            "global_text": "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones",
            "ref_path": "runs/geology/reference.png",
            "shots": [f"shot_{i:03d}" for i in range(1, 15)],
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "is_legacy": True
        },
        {
            "name": "Geology Collided",
            "global_text": "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones",
            "ref_path": "runs/geology/reference.png",
            "shots": [f"shot_{i:03d}" for i in range(1, 15)],
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "is_legacy": False
        },
        {
            "name": "Geology v2",
            "global_text": "a colorful cartoon illustration of rocks, rocky terrain, boulders and stones",
            "ref_path": "runs/geology/reference.png",
            "shots": [f"shot_{i:03d}" for i in range(1, 15)],
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "is_legacy": False
        },
        {
            "name": "Ecology v2",
            "global_text": "a colorful cartoon illustration of a dripping water cave, the water cycle",
            "ref_path": "runs/ecology/reference.png",
            "shots": [f"shot_{i:03d}" for i in range(1, 17)],
            "weights": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
            "is_legacy": False
        }
    ]
    
    with open("runs/geology/sweep/manifest.json", "r") as f:
        legacy_man = json.load(f)
        
    def get_path(sweep_name, shot, w):
        if sweep_name == "Geology Legacy":
            if w == 0.0: return f"runs/geology/sweep/collapse_evidence/{shot}_w0.0.png"
            return next((m["path"] for m in legacy_man if m["shot_id"] == shot and abs(m["weight"] - w) < 0.01), None)
        elif sweep_name == "Geology Collided":
            return f"runs/G0_A0_geology_collided/images/A0/w{w:.2f}/{shot}.png"
        elif sweep_name == "Geology v2":
            return f"runs/G0_A0_geology/images/A0/w{w:.2f}/{shot}.png"
        elif sweep_name == "Ecology v2":
            return f"runs/G0_A0_ecology/images/A0/w{w:.2f}/{shot}.png"
            
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    tau = 0.7
    summary_rows = []
    output_md = []
    
    for i, sweep in enumerate(sweeps):
        name = sweep["name"]
        shots = sweep["shots"]
        weights = sweep["weights"]
        global_text = sweep["global_text"]
        
        # Load reference embedding
        ref_path = sweep["ref_path"]
        emb_ref = scorer.embed_dino(ref_path) if os.path.exists(ref_path) else None
        
        data = {} # shot -> {w: (mc, cs, refsim)}
        for shot in shots:
            w0_path = get_path(name, shot, 0.0)
            
            data[shot] = {}
            for w in weights:
                img_path = get_path(name, shot, w)
                if img_path and os.path.exists(img_path):
                    mc = scorer.get_clip_concept(img_path, global_text)
                    
                    if w0_path and os.path.exists(w0_path):
                        emb1 = scorer.embed_dino(img_path)
                        emb2 = scorer.embed_dino(w0_path)
                        cs = float((emb1 * emb2).sum().item())
                    else:
                        cs = float('nan')
                        
                    if emb_ref is not None:
                        emb1 = scorer.embed_dino(img_path)
                        refsim = float((emb1 * emb_ref).sum().item())
                    else:
                        refsim = float('nan')
                        
                    data[shot][w] = (mc, cs, refsim)
                    
        # Fixed frontier
        output_md.append(f"### {name} - Fixed Frontier")
        output_md.append("| w | mean_concept | c̄ | ref_sim |")
        output_md.append("|---|---|---|---|")
        
        fixed_pts = []
        for w in weights:
            mcs = [data[s][w][0] for s in shots if w in data[s] and not np.isnan(data[s][w][0])]
            css = [data[s][w][1] for s in shots if w in data[s] and not np.isnan(data[s][w][1])]
            refsims = [data[s][w][2] for s in shots if w in data[s] and not np.isnan(data[s][w][2])]
            
            mc_val = sum(mcs)/len(mcs) if mcs else float('nan')
            cs_val = sum(css)/len(css) if css else float('nan')
            refsim_val = sum(refsims)/len(refsims) if refsims else float('nan')
            
            c_str = f"{cs_val:.4f}" + (" (n=3)" if sweep["is_legacy"] and len(css) == 3 else "")
            
            output_md.append(f"| {w:.2f} | {mc_val:.4f} | {c_str} | {refsim_val:.4f} |")
            if w > 0:
                fixed_pts.append((mc_val, cs_val))
                
        # Methods
        methods = {
            "Original DACA (max w)": {"mcs": [], "css": [], "n": 0},
            "Benefit-gated (δ=0.01)": {"mcs": [], "css": [], "n": 0},
            "Pure argmax": {"mcs": [], "css": [], "n": 0}
        }
        
        w_min = 0.2
        for s in shots:
            d = data[s]
            if w_min not in d or np.isnan(d[w_min][1]):
                continue # Skip if no c_s at w_min (e.g. legacy missing w0)
                
            methods["Original DACA (max w)"]["n"] += 1
            methods["Benefit-gated (δ=0.01)"]["n"] += 1
            methods["Pure argmax"]["n"] += 1
            
            feasible_ws = [w for w in weights if w > 0 and w in d and not np.isnan(d[w][1]) and d[w][1] >= tau]
            
            # Original DACA
            w_star_orig = max(feasible_ws) if feasible_ws else w_min
            methods["Original DACA (max w)"]["mcs"].append(d[w_star_orig][0])
            methods["Original DACA (max w)"]["css"].append(d[w_star_orig][1])
            
            # Benefit-gated
            mc_min = d[w_min][0]
            qualifying_ws = [w for w in feasible_ws if w > w_min and d[w][0] - mc_min > 0.01]
            w_star_bg = max(qualifying_ws, key=lambda w: d[w][0]) if qualifying_ws else w_min
            methods["Benefit-gated (δ=0.01)"]["mcs"].append(d[w_star_bg][0])
            methods["Benefit-gated (δ=0.01)"]["css"].append(d[w_star_bg][1])
            
            # Pure argmax
            if feasible_ws:
                w_star_pa = feasible_ws[0]
                for w in feasible_ws[1:]:
                    if d[w][0] > d[w_star_pa][0]:
                        w_star_pa = w
            else:
                w_star_pa = w_min
            methods["Pure argmax"]["mcs"].append(d[w_star_pa][0])
            methods["Pure argmax"]["css"].append(d[w_star_pa][1])
            
        for m_name, m_data in methods.items():
            if m_data["n"] == 0: continue
            mc_avg = sum(m_data["mcs"])/len(m_data["mcs"])
            cs_avg = sum(m_data["css"])/len(m_data["css"])
            interp_cs = interpolate_c_bar(mc_avg, fixed_pts)
            delta = cs_avg - interp_cs
            
            n_str = f"n={m_data['n']}"
            if sweep["is_legacy"]: n_str = f"n=3 (anchors exist only for shots 005/011/013)"
            
            summary_rows.append(f"| {name} | {m_name} | {mc_avg:.4f} | {cs_avg:.4f} | {delta:+.4f} | {n_str} |")
            
            methods[m_name]["avg"] = (mc_avg, cs_avg)
            
        output_md.append("")
        
        # Plotting
        ax = axes[i]
        fx = [p[0] for p in fixed_pts]
        fy = [p[1] for p in fixed_pts]
        
        # Sort fixed pts for line plotting
        sort_idx = np.argsort(fx)
        fx_s = np.array(fx)[sort_idx]
        fy_s = np.array(fy)[sort_idx]
        
        ax.plot(fx_s, fy_s, 'ko-', label="Fixed Frontier")
        
        # Plot methods
        colors = {"Original DACA (max w)": "blue", "Benefit-gated (δ=0.01)": "red", "Pure argmax": "green"}
        markers = {"Original DACA (max w)": "s", "Benefit-gated (δ=0.01)": "^", "Pure argmax": "d"}
        
        for m_name, m_data in methods.items():
            if "avg" in m_data:
                ax.plot(m_data["avg"][0], m_data["avg"][1], color=colors[m_name], marker=markers[m_name], markersize=10, label=m_name)
                
        ax.set_title(name)
        ax.set_xlabel("mean_concept")
        ax.set_ylabel("c̄")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig("runs/g0_frontiers.png", dpi=300)
    
    summary_md = [
        "### Summary: Method Advantage vs Interpolated Fixed Frontier",
        "| Sweep | Method | mc | c̄ | Δc̄ vs Frontier | n |",
        "|---|---|---|---|---|---|"
    ]
    summary_md.extend(summary_rows)
    
    with open("runs/g0_tables.md", "w") as f:
        f.write("\n".join(output_md) + "\n" + "\n".join(summary_md))
        
if __name__ == "__main__":
    main()

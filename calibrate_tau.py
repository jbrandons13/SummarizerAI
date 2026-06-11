import json
import numpy as np
from scipy.stats import spearmanr

def main():
    with open("runs/G0_A0_geology/scored_records.jsonl", "r") as f:
        ground_truth = [json.loads(line) for line in f]
        
    true_curves = {}
    for r in ground_truth:
        s = "geo_" + r["shot_id"].split("_")[1]
        w = r["knobs"]["w"]
        if s not in true_curves:
            true_curves[s] = {"c": {}, "mc": {}}
        true_curves[s]["c"][w] = r["metrics"]["c_s"]
        true_curves[s]["mc"][w] = r["metrics"]["mean_concept"]
        
    W_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    w_min = 0.2
    tau = 0.70
    delta = 0.01
    
    shots = list(true_curves.keys())
    
    # Identify non-monotone subset
    # shot is non-monotone if c_bar(0.8) > c_bar(0.6)
    non_monotone = [s for s in shots if true_curves[s]["c"][0.8] > true_curves[s]["c"][0.6]]
    
    # True selections
    true_selections = {"max_w": {}, "bg": {}}
    for s in shots:
        feasible = [w for w in W_grid if true_curves[s]["c"][w] >= tau]
        # max w
        w_max = max(feasible) if feasible else w_min
        true_selections["max_w"][s] = w_max
        
        # bg
        mc_min = true_curves[s]["mc"][w_min]
        qual = [w for w in feasible if w > w_min and true_curves[s]["mc"][w] - mc_min > delta]
        w_bg = max(qual, key=lambda w: true_curves[s]["mc"][w]) if qual else w_min
        true_selections["bg"][s] = w_bg
        
    results = {}
    
    for k in [9, 12]:
        with open(f"runs/G1_Alt1_v2/matrices_k{k}.json", "r") as f:
            matrices = json.load(f)
            
        k_results = {"calibration": {}, "evaluation": {}}
        
        # 1. Calibrate tau_prev
        # Sweep tau_prev from 0.0 to 1.0, find max agreement for max-w
        best_tau = 0.0
        best_agrm = -1
        
        for t_p in np.arange(0.0, 1.01, 0.01):
            agrm = 0
            for s in shots:
                pred_c = matrices[s]
                # preview feasible
                # matrices use string keys for w!
                p_feas = [w for w in W_grid if pred_c[f"{w:.1f}"]["c_hat"] >= t_p]
                p_w = max(p_feas) if p_feas else w_min
                if abs(W_grid.index(p_w) - W_grid.index(true_selections["max_w"][s])) <= 1:
                    agrm += 1
            if agrm > best_agrm:
                best_agrm = agrm
                best_tau = t_p
                
        k_results["calibration"]["tau_prev"] = best_tau
        
        # 2. Evaluate with calibrated tau_prev
        pred_selections = {"max_w": {}, "bg": {}}
        
        for s in shots:
            pred_c = matrices[s]
            p_feas = [w for w in W_grid if pred_c[f"{w:.1f}"]["c_hat"] >= best_tau]
            p_max = max(p_feas) if p_feas else w_min
            pred_selections["max_w"][s] = p_max
            
            mc_min = pred_c[f"{w_min:.1f}"]["preview_mc"]
            qual = [w for w in p_feas if w > w_min and pred_c[f"{w:.1f}"]["preview_mc"] - mc_min > delta]
            p_bg = max(qual, key=lambda w: pred_c[f"{w:.1f}"]["preview_mc"]) if qual else w_min
            pred_selections["bg"][s] = p_bg
            
        # Spearman
        c_hat_spearman = []
        mc_spearman = []
        for s in shots:
            true_c_list = [true_curves[s]["c"][w] for w in W_grid]
            pred_c_list = [matrices[s][f"{w:.1f}"]["c_hat"] for w in W_grid]
            sp, _ = spearmanr(true_c_list, pred_c_list)
            if not np.isnan(sp): c_hat_spearman.append(sp)
            
            true_mc_list = [true_curves[s]["mc"][w] for w in W_grid]
            pred_mc_list = [matrices[s][f"{w:.1f}"]["preview_mc"] for w in W_grid]
            sp, _ = spearmanr(true_mc_list, pred_mc_list)
            if not np.isnan(sp): mc_spearman.append(sp)
            
        k_results["evaluation"]["c_hat_spearman"] = np.median(c_hat_spearman)
        k_results["evaluation"]["mc_spearman"] = np.median(mc_spearman)
        
        # Agreement
        def calc_agrm(rule, subset=shots):
            agrm = 0
            for s in subset:
                t_idx = W_grid.index(true_selections[rule][s])
                p_idx = W_grid.index(pred_selections[rule][s])
                if abs(t_idx - p_idx) <= 1:
                    agrm += 1
            return agrm / len(subset) if subset else 0
            
        k_results["evaluation"]["max_w_agrm"] = calc_agrm("max_w")
        k_results["evaluation"]["max_w_agrm_non_mono"] = calc_agrm("max_w", non_monotone)
        k_results["evaluation"]["bg_agrm"] = calc_agrm("bg")
        k_results["evaluation"]["bg_agrm_non_mono"] = calc_agrm("bg", non_monotone)
        
        results[k] = k_results

    with open("runs/G1_Alt1_v2/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))
    print("Non-monotone subset:", non_monotone)

if __name__ == "__main__":
    main()

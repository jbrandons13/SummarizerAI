import os
import csv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def load_data():
    videos = {
        "V1_Geology": "fullrun_results/data/V1_Geology",
        "V2_Ecology": "fullrun_results/data/V2_Ecology",
        "V3_Sun": "fullrun_results/data/V3_Sun",
        "V4_Heart": "fullrun_results/data/V4_Heart",
        "V5_iPhone": "fullrun_results/data/V5_iPhone"
    }
    
    dataset = []
    
    for vid, path_prefix in videos.items():
        # Load adaptive anchors (true w*)
        true_w_star = {}
        true_concept = {}
        true_content = {}
        try:
            with open(f"{path_prefix}_adaptive_anchor.csv") as f:
                reader = csv.reader(f)
                in_shots = False
                for row in reader:
                    if not row: continue
                    if row[0] == "shot":
                        in_shots = True
                        continue
                    if in_shots:
                        if row[0] == "scheme":
                            break # done with shots
                        # shot,adaptive_w*,content_at_w*,concept_at_w*
                        sid = row[0].replace('"', '')
                        true_w_star[sid] = float(row[1])
                        true_content[sid] = float(row[2])
                        true_concept[sid] = float(row[3])
        except Exception as e:
            print(f"Error loading {vid} adaptive: {e}")
            continue
            
        # Load metrics (c_s, d_s)
        try:
            with open(f"{path_prefix}_collapse_metrics.csv") as f:
                reader = csv.reader(f)
                in_shots = True
                
                shot_data = {}
                for row in reader:
                    if not row: continue
                    if row[0] == "weight":
                        in_shots = False
                        break
                    if row[0] == "shot": continue
                    if in_shots:
                        sid = row[0].replace('"', '')
                        w = float(row[1])
                        sim_ref = float(row[2])
                        sim_own = float(row[3])
                        
                        if sid not in shot_data:
                            shot_data[sid] = {"w": [], "sim_ref": [], "sim_own": []}
                        shot_data[sid]["w"].append(w)
                        shot_data[sid]["sim_ref"].append(sim_ref)
                        shot_data[sid]["sim_own"].append(sim_own)
        except Exception as e:
            print(f"Error loading {vid} metrics: {e}")
            continue
            
        for sid, sdata in shot_data.items():
            if sid not in true_w_star: continue
            
            # Find d_s (sim_to_ref at w=0)
            idx0 = sdata["w"].index(0.0)
            d_s = sdata["sim_ref"][idx0]
            
            # Find c_s(max_w)
            max_w = max(sdata["w"])
            idx1 = sdata["w"].index(max_w)
            c_s_1 = sdata["sim_own"][idx1]
            
            # c_s(w) curve
            w_grid = sdata["w"]
            c_s = sdata["sim_own"]
            sim_ref_curve = sdata["sim_ref"]
            
            dataset.append({
                "video": vid,
                "shot": sid,
                "d_s": d_s,
                "c_s_1": c_s_1,
                "w_grid": np.array(w_grid),
                "c_s": np.array(c_s),
                "sim_ref_curve": np.array(sim_ref_curve),
                "true_w": true_w_star[sid],
                "true_concept": true_concept[sid],
                "true_content": true_content[sid]
            })
            
    return pd.DataFrame(dataset)

def step1_analysis():
    df = load_data()
    print(f"Loaded {len(df)} shots across {df['video'].nunique()} videos.")
    
    # 1. Asymptote analysis
    gap = df["c_s_1"] - df["d_s"]
    print(f"\nAsymptote Analysis: c_s(w_max) vs d_s")
    print(f"Mean Gap: {gap.mean():.4f}")
    print(f"Std Gap: {gap.std():.4f}")
    corr, p = pearsonr(df["d_s"], df["c_s_1"])
    print(f"Correlation: {corr:.4f} (p={p:.4g})")
    
    # 2. Cross-validation
    videos = df["video"].unique()
    
    results_a = []
    results_b = []
    
    for test_vid in videos:
        train_df = df[df["video"] != test_vid]
        test_df = df[df["video"] == test_vid]
        
        # Method A: Direct linear regression of true_w on d_s
        # true_w = m * d_s + b
        m, b = np.polyfit(train_df["d_s"], train_df["true_w"], 1)
        
        # Method B: Global g(w)
        # g(w) = (1 - c_s(w)) / (1 - d_s)
        g_w = {}
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            g_vals = []
            for _, row in train_df.iterrows():
                idx = np.where(row["w_grid"] == w)[0]
                if len(idx) > 0:
                    c = row["c_s"][idx[0]]
                    d = row["d_s"]
                    if d < 0.99: # avoid div by zero
                        g_vals.append((1 - c) / (1 - d))
            if g_vals:
                g_w[w] = np.mean(g_vals)
                
        # Ensure monotonic increasing g(w)
        prev_g = 0
        sorted_w = sorted(g_w.keys())
        for w in sorted_w:
            g_w[w] = max(prev_g, g_w[w])
            prev_g = g_w[w]
            
        # Predict on test
        for _, row in test_df.iterrows():
            d_s = row["d_s"]
            
            # Pred A
            pred_w_a_raw = m * d_s + b
            # Snap to grid
            grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
            pred_w_a = grid[np.argmin(np.abs(grid - pred_w_a_raw))]
            
            # Pred B
            pred_w_b = 0.0
            for w in sorted_w:
                c_pred = 1 - (1 - d_s) * g_w[w]
                if c_pred >= 0.7:
                    pred_w_b = w
                else:
                    break
                    
            # Get actual content at pred_w
            def get_metrics_at_w(w_target):
                idx = np.where(row["w_grid"] == w_target)[0]
                if len(idx) > 0:
                    return row["c_s"][idx[0]], row["sim_ref_curve"][idx[0]]
                return 0, 0
                
            content_a, concept_a = get_metrics_at_w(pred_w_a)
            content_b, concept_b = get_metrics_at_w(pred_w_b)
            
            res_row = {
                "video": row["video"],
                "shot": row["shot"],
                "d_s": d_s,
                "true_w": row["true_w"],
                "true_content": row["true_content"],
                "true_concept": row["true_concept"],
                "pred_w_a": pred_w_a,
                "content_a": content_a,
                "concept_a": concept_a,
                "breach_a": content_a < 0.7,
                "pred_w_b": pred_w_b,
                "content_b": content_b,
                "concept_b": concept_b,
                "breach_b": content_b < 0.7,
            }
            results_b.append(res_row)
            
    res_df = pd.DataFrame(results_b)
    res_df.to_csv("predictive_daca_offline.csv", index=False)
    
    print("\n--- RESULTS METHOD A (Direct Regression) ---")
    print(f"Mean |Delta w*|: {np.abs(res_df['pred_w_a'] - res_df['true_w']).mean():.4f}")
    print(f"Floor Breach Rate: {res_df['breach_a'].mean():.2%}")
    print(f"Mean Content (Pred vs True): {res_df['content_a'].mean():.4f} vs {res_df['true_content'].mean():.4f}")
    print(f"Mean Concept (Pred vs True): {res_df['concept_a'].mean():.4f} vs {res_df['true_concept'].mean():.4f}")
    
    print("\n--- RESULTS METHOD B (Parametric Curve Fit) ---")
    print(f"Mean |Delta w*|: {np.abs(res_df['pred_w_b'] - res_df['true_w']).mean():.4f}")
    print(f"Floor Breach Rate: {res_df['breach_b'].mean():.2%}")
    print(f"Mean Content (Pred vs True): {res_df['content_b'].mean():.4f} vs {res_df['true_content'].mean():.4f}")
    print(f"Mean Concept (Pred vs True): {res_df['concept_b'].mean():.4f} vs {res_df['true_concept'].mean():.4f}")
    
    # Plot true_w vs d_s
    plt.figure(figsize=(8,6))
    plt.scatter(df["d_s"], df["true_w"], alpha=0.5, label="True w*")
    
    d_s_range = np.linspace(df["d_s"].min(), df["d_s"].max(), 100)
    # Fit globally just for plot
    m_glob, b_glob = np.polyfit(df["d_s"], df["true_w"], 1)
    plt.plot(d_s_range, m_glob * d_s_range + b_glob, 'r-', label=f"Linear Fit (Method A)")
    
    # Method B global curve
    g_w_glob = {}
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        g_vals = []
        for _, row in df.iterrows():
            idx = np.where(row["w_grid"] == w)[0]
            if len(idx) > 0:
                c = row["c_s"][idx[0]]
                d = row["d_s"]
                if d < 0.99:
                    g_vals.append((1 - c) / (1 - d))
        if g_vals:
            g_w_glob[w] = np.mean(g_vals)
    prev_g = 0
    for w in sorted(g_w_glob.keys()):
        g_w_glob[w] = max(prev_g, g_w_glob[w])
        prev_g = g_w_glob[w]
        
    w_b_curve = []
    for d in d_s_range:
        best_w = 0.0
        for w in sorted(g_w_glob.keys()):
            if 1 - (1 - d) * g_w_glob[w] >= 0.7:
                best_w = w
            else:
                break
        w_b_curve.append(best_w)
        
    plt.plot(d_s_range, w_b_curve, 'g--', label="Parametric Fit (Method B)")
    
    plt.xlabel("d_s (Similarity to Reference at w=0)")
    plt.ylabel("w* (Optimal Anchor Weight)")
    plt.title("w* vs d_s with Predictive Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("predictive_daca_fit.png", dpi=150)

if __name__ == "__main__":
    step1_analysis()

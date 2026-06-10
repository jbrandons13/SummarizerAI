import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def make_frontier_ecology():
    with open("runs/ecology/sweep/collapse_metrics.csv", "r") as f:
        lines = f.read().splitlines()
    
    # Find empty line
    empty_idx = lines.index("")
    # Read the means part
    means_lines = lines[empty_idx+1:]
    import io
    df = pd.read_csv(io.StringIO("\n".join(means_lines)))
    
    weights = df['weight'].values
    m_ref = df['mean_sim_to_reference'].values
    m_self = df['mean_sim_to_own_w0'].values
    m_inter = df['mean_inter_shot_sim'].values

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(weights, m_ref, "o-", color="#c0392b",
             label="similarity to reference  (the score that rewards copying)  \u2191")
    plt.plot(weights, m_self, "s-", color="#2471a3",
             label="similarity to the shot's own original scene  (content kept)  \u2193")
    plt.plot(weights, m_inter, "^--", color="#7d8c00",
             label="similarity among the shots  (they converge to one)  \u2191")
    
    # Mark crossover (~0.3)
    plt.axvline(x=0.3, color='gray', linestyle=':', alpha=0.8)
    plt.text(0.32, 0.45, "crossover", color='gray')

    plt.xlabel("anchoring weight")
    plt.ylabel("DINOv2 cosine similarity")
    plt.title("Ecology Reward Collapse Curve")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="center left")
    plt.tight_layout()
    plt.savefig("WRITTING_STUFFS/fig4.4_frontier_ecology.png", dpi=150)
    print("Saved fig4.4_frontier_ecology.png")
    
    print("\nEcology tab:frontier rows:")
    for w in [0.2, 0.4, 0.6, 0.8]:
        idx = list(weights).index(w)
        print(f"w={w}: ref={m_ref[idx]:.4f}, self={m_self[idx]:.4f}, inter={m_inter[idx]:.4f}")

def make_daca_curve():
    df = pd.read_csv("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4/collapse_evidence_fine/collapse_metrics.csv")
    # Read until the empty line to get per-shot data
    with open("data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4/collapse_evidence_fine/collapse_metrics.csv", "r") as f:
        lines = f.readlines()
    
    per_shot_lines = []
    for line in lines[1:]:
        if not line.strip():
            break
        per_shot_lines.append(line.strip().split(','))
    
    df_shots = pd.DataFrame(per_shot_lines, columns=['shot','weight','sim_to_reference','sim_to_own_w0'])
    df_shots['weight'] = df_shots['weight'].astype(float)
    df_shots['sim_to_own_w0'] = df_shots['sim_to_own_w0'].astype(float)

    plt.figure(figsize=(7.2, 4.6))
    
    shots = df_shots['shot'].unique()
    tau = 0.70
    
    for shot in shots:
        shot_data = df_shots[df_shots['shot'] == shot].sort_values('weight')
        ws = shot_data['weight'].values
        cs = shot_data['sim_to_own_w0'].values
        
        plt.plot(ws, cs, alpha=0.5, linewidth=1.5, marker='.')
        
        # Find w* (largest w where cs >= tau)
        valid = shot_data[shot_data['sim_to_own_w0'] >= tau]
        if len(valid) > 0:
            w_star = valid['weight'].max()
            c_star = valid[valid['weight'] == w_star]['sim_to_own_w0'].values[0]
            plt.plot(w_star, c_star, 'ko', markersize=6)
            
    plt.axhline(y=tau, color='r', linestyle='--', label=f'tau = {tau}')
    plt.xlabel("anchoring weight")
    plt.ylabel("content-kept $c_s(w)$")
    plt.title("Per-shot DACA Curve (Geology)")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("WRITTING_STUFFS/fig3.3_daca_curve.png", dpi=150)
    print("Saved fig3.3_daca_curve.png")

if __name__ == "__main__":
    make_frontier_ecology()
    make_daca_curve()

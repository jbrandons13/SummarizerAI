import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open("runs/blockprobe/results.json", "r") as f:
        results = json.load(f)
        
    sites = list(results.keys())
    shots = list(results[sites[0]].keys())
    weights = [0.3, 0.5, 0.8]
    
    # Compute averages over the 6 shots
    avg_results = {site: {"c_s": [], "ref_sim": []} for site in sites}
    
    for site in sites:
        for w in weights:
            c_s_vals = []
            ref_sim_vals = []
            for shot in shots:
                if str(w) in results[site][shot]:
                    c_s_vals.append(results[site][shot][str(w)]["c_s"])
                    ref_sim_vals.append(results[site][shot][str(w)]["ref_sim"])
            avg_results[site]["c_s"].append(np.mean(c_s_vals))
            avg_results[site]["ref_sim"].append(np.mean(ref_sim_vals))
            
    # Plotting
    plt.figure(figsize=(10, 8))
    for site in sites:
        c = avg_results[site]["c_s"]
        r = avg_results[site]["ref_sim"]
        plt.plot(c, r, marker='o', label=site)
        for i, w in enumerate(weights):
            plt.annotate(f"w={w}", (c[i], r[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
            
    plt.xlabel("Mean c_s (Content Preservation vs w=0)")
    plt.ylabel("Mean ref_sim (Style Fidelity vs Reference)")
    plt.title("F2 Block Probe: Style-Content Separation")
    plt.legend()
    plt.grid(True)
    plt.savefig("runs/blockprobe/pareto_plot.png")
    
    # Pareto dominance check vs global
    global_c = np.array(avg_results["global"]["c_s"])
    global_r = np.array(avg_results["global"]["ref_sim"])
    
    print("F2 Block Probe Analysis:")
    print("-" * 40)
    for site in sites:
        if site == "global":
            continue
        c = np.array(avg_results[site]["c_s"])
        r = np.array(avg_results[site]["ref_sim"])
        
        # Check if site curve dominates global curve at any point.
        # We'll just do a simple pointwise check or overall area check.
        # Domination: higher r at matched c, or higher c at matched r.
        # Let's print the (c, r) values to easily compare.
        print(f"\nSite: {site}")
        for i, w in enumerate(weights):
            print(f"  w={w}: c_s={c[i]:.4f}, ref_sim={r[i]:.4f}")
            
    print("\nGlobal:")
    for i, w in enumerate(weights):
        print(f"  w={w}: c_s={global_c[i]:.4f}, ref_sim={global_r[i]:.4f}")

if __name__ == "__main__":
    main()

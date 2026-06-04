import os
import json
import collections
from pathlib import Path
from src.phase4.anchor_policy import (
    AlwaysChainPolicy, NeverChainPolicy, FixedIntervalPolicy,
    SemanticTriggeredPolicy, serialize_decisions
)

def run_all_policies():
    video_id = "lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge"
    base_dir = Path(f"data/intermediate/{video_id}/phase4")
    
    storyboard_path = base_dir / "storyboard.json"
    if not storyboard_path.exists():
        print(f"Error: {storyboard_path} does not exist.")
        return
        
    with open(storyboard_path, 'r', encoding='utf-8') as f:
        storyboard = json.load(f)
        
    # Get config 
    k = 5
    threshold_chain = 0.75
    threshold_soft = 0.55
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    policies = [
        AlwaysChainPolicy(),
        NeverChainPolicy(),
        FixedIntervalPolicy(k=k),
        SemanticTriggeredPolicy(
            embedding_model=embedding_model,
            threshold_chain=threshold_chain,
            threshold_soft=threshold_soft
        )
    ]
    
    configs = {
        "always_chain": {},
        "never_chain": {},
        "fixed_interval": {"k": k},
        "semantic_triggered": {
            "embedding_model": embedding_model,
            "threshold_chain": threshold_chain,
            "threshold_soft": threshold_soft
        }
    }
    
    print(f"--- Anchor Policy Run for {video_id} ---")
    
    distribution = {}
    similarities = []
    
    for policy in policies:
        policy_name = policy.name
        print(f"Running {policy_name}...")
        
        decisions = policy.resolve(storyboard)
        
        # Serialize & Write
        output_dir = base_dir / policy_name
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "storyboard_with_anchors.json"
        
        json_str = serialize_decisions(
            video_id=video_id,
            policy_name=policy_name,
            policy_config=configs[policy_name],
            decisions=decisions
        )
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
            
        # Count stats
        counts = collections.Counter(d.anchor_decision for d in decisions)
        distribution[policy_name] = {
            "RESET": counts.get("RESET", 0),
            "CHAIN": counts.get("CHAIN", 0),
            "SOFT_CHAIN": counts.get("SOFT_CHAIN", 0),
            "Total": len(decisions)
        }
        
        if policy_name == "semantic_triggered":
            similarities = [d.similarity_to_prev for d in decisions if d.similarity_to_prev is not None]

    print("\n--- Decision Distribution ---")
    print(f"{'Policy':<20} | {'RESET':<5} | {'CHAIN':<5} | {'SOFT_CHAIN':<10} | {'Total':<5}")
    print("-" * 57)
    for name, stat in distribution.items():
        print(f"{name:<20} | {stat['RESET']:<5} | {stat['CHAIN']:<5} | {stat['SOFT_CHAIN']:<10} | {stat['Total']:<5}")
        
    print("\n--- Semantic Triggered Similarity Histogram ---")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist = {i: 0 for i in range(len(bins)-1)}
    for sim in similarities:
        for i in range(len(bins)-1):
            if bins[i] <= sim < bins[i+1] or (i == len(bins)-2 and sim >= 1.0):
                hist[i] += 1
                break
                
    for i in range(len(bins)-1):
        lower = bins[i]
        upper = bins[i+1]
        count = hist[i]
        bar = "#" * count
        print(f"[{lower:.1f}, {upper:.1f}): {bar} ({count})")
        
if __name__ == "__main__":
    run_all_policies()

import json
import argparse
import copy
from pathlib import Path

from src.phase4.anchor_policy import (
    AlwaysChainPolicy,
    NeverChainPolicy,
    FixedIntervalPolicy,
    SemanticTriggeredPolicy,
    SegmentBoundaryPolicy,
    ConceptAnchorPolicy
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid-dir", default="data/intermediate/lT_QAkL6lj0_where-do-rocks-come-from-crash-course-ge/phase4", help="Path to video intermediate directory")
    args = parser.parse_args()

    vid_dir = Path(args.vid_dir)
    storyboard_path = vid_dir / "storyboard.json"
    shots_path = vid_dir / "shots.json"

    with open(storyboard_path, 'r') as f:
        storyboard = json.load(f)
    
    with open(shots_path, 'r') as f:
        shots_data = json.load(f)
        
    shots_lookup = {s["shot_id"]: s for s in shots_data.get("shots", [])}

    for shot in storyboard.get("shots", []):
        shot_id = shot["shot_id"]
        if shot_id in shots_lookup:
            segment_ids = shots_lookup[shot_id].get("source_segment_ids", [])
            shot["_segment_id"] = segment_ids[0] if segment_ids else None

    policies = {
        "always_chain": AlwaysChainPolicy(),
        "never_chain": NeverChainPolicy(),
        "fixed_interval_5": FixedIntervalPolicy(5),
        "semantic_triggered": SemanticTriggeredPolicy(),
        "segment_boundary": SegmentBoundaryPolicy(),
        "concept_anchor_canonical": ConceptAnchorPolicy("canonical"),
        "concept_anchor_recent": ConceptAnchorPolicy("recent")
    }

    results = {}

    for name, policy in policies.items():
        sb_copy = copy.deepcopy(storyboard)
        decisions = policy.resolve(sb_copy)
        
        for shot, dec in zip(sb_copy.get("shots", []), decisions):
            shot["anchor_decision"] = dec.anchor_decision
            if dec.anchor_source:
                shot["anchor_source"] = dec.anchor_source
        
        dist = {}
        for d in decisions:
            dist[d.anchor_decision] = dist.get(d.anchor_decision, 0) + 1
            
        results[name] = {
            "decisions": [d.anchor_decision for d in decisions],
            "dist": dist
        }

        out_dir = vid_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "storyboard_with_anchors.json", "w") as f:
            json.dump(sb_copy, f, indent=2)

    print("=== Policy Distribution Report ===")
    for name, data in results.items():
        print(f"[{name}]: {data['dist']}")

    print("\n=== Concept Anchor Sanity Check ===")
    for name, policy_key in [("canonical", "concept_anchor_canonical"), ("recent", "concept_anchor_recent")]:
        decisions = policies[policy_key].resolve(copy.deepcopy(storyboard))
        total_anchors = sum(1 for d in decisions if d.anchor_decision == "CONCEPT_ANCHOR")
        
        non_adjacent = sum(1 for d in decisions if d.anchor_decision == "CONCEPT_ANCHOR" and d.reference_distance is not None and d.reference_distance > 1)
                    
        print(f"{name}: {total_anchors} anchors / {non_adjacent} non-adjacent")

if __name__ == "__main__":
    main()

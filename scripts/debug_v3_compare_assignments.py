import json
from pathlib import Path

def compare_assignments(output_dir: Path):
    """Bandingkan assignment antar config untuk 1 video."""
    configs = ["siglip_temporal", "siglip_temporal_hungarian", "siglip_temporal_dp"]
    assignments = {}

    for c in configs:
        with open(output_dir / f"scene_matches_{c}.json") as f:
            data = json.load(f)
        matches = data.get("matches", data) if isinstance(data, dict) else data
        assignments[c] = [m["matched_scene_id"] for m in matches]

    # Print side-by-side
    print(f"\n{'sent_id':<10}{'Greedy':<12}{'Hungarian':<12}{'DP':<12}")
    print("-" * 46)
    for i in range(len(assignments["siglip_temporal"])):
        g = assignments["siglip_temporal"][i]
        h = assignments["siglip_temporal_hungarian"][i]
        d = assignments["siglip_temporal_dp"][i]
        marker = "" if g == h == d else " ← DIFFERENT"
        print(f"{i:<10}{g:<12}{h:<12}{d:<12}{marker}")

    # Aggregate
    diff_gh = sum(1 for i in range(len(assignments["siglip_temporal"]))
                  if assignments["siglip_temporal"][i] != assignments["siglip_temporal_hungarian"][i])
    diff_gd = sum(1 for i in range(len(assignments["siglip_temporal"]))
                  if assignments["siglip_temporal"][i] != assignments["siglip_temporal_dp"][i])
    diff_hd = sum(1 for i in range(len(assignments["siglip_temporal_hungarian"]))
                  if assignments["siglip_temporal_hungarian"][i] != assignments["siglip_temporal_dp"][i])

    print(f"\nDifferent assignments:")
    print(f"  Greedy vs Hungarian: {diff_gh}")
    print(f"  Greedy vs DP: {diff_gd}")
    print(f"  Hungarian vs DP: {diff_hd}")

if __name__ == "__main__":
    compare_assignments(Path("data/intermediate/501c3e27-b34c-48b4-bffa-de0e61b97ecd"))
